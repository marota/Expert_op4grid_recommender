"""
tests/manoeuvre/test_dataset_timeline.py
-----------------------------------------
Pipeline « dataset topologies historiques » (``manoeuvre.dataset``) :

- segmentation d'une chronologie en blocs de transition (états stables,
  transitoires, manœuvres observées, oscillations repliées, réversibilité) ;
- validation de bout en bout sur une chronologie **reconstruite depuis une
  séquence réelle du dépôt** (CARRIP3, 20 manœuvres) rejouée sur la fixture :
  le bloc retrouvé doit reproduire exactement départ, cible et séquence ;
- tagging structurel (fusion/scission, consignation/remise en service,
  ré-aiguillage) et repli par nommage ;
- extraction aux formats scénario/séquence du dépôt + stats ;
- adaptateur ``dgitt`` sur un CSV événements minimal.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from expert_op4grid_recommender.manoeuvre.dataset import (
    Snapshot,
    TimelinePoste,
    bloc_to_scenario,
    bloc_to_sequence_observee,
    ecrire_dataset,
    poste_from_fixture_json,
    stats_blocs,
    taguer_bloc,
    topologie_id,
)
from expert_op4grid_recommender.manoeuvre.plugins import CibleDetaillee

from .fixture_loader import list_available_fixtures

REPO = pathlib.Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "manoeuvre" / "fixtures"
SEQ_CARRIP3 = REPO / "tests" / "manoeuvre" / "sequences" / "CARRIP3_cible_1noeud.json"

pytestmark = pytest.mark.skipif(
    "CARRIP3" not in list_available_fixtures() or not SEQ_CARRIP3.exists(),
    reason="Fixture/séquence CARRIP3 absentes.",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snap(t: int, etats: dict) -> Snapshot:
    return Snapshot(timestamp=f"2021-01-01T{t:04d}", etats=dict(etats))


def _timeline_depuis_sequence(path: pathlib.Path,
                              stable: int = 3) -> tuple[TimelinePoste, dict]:
    """Chronologie : départ stable, un snapshot par manœuvre, cible stable."""
    data = json.loads(path.read_text())
    etat = {k: bool(v) for k, v in data["depart"].items()}
    snaps, t = [], 0
    for _ in range(stable):
        snaps.append(_snap(t, etat)); t += 1
    for m in sorted(data["manoeuvres"], key=lambda m: m["ordre"]):
        etat[m["switch_id"]] = (m["action"] == "OPEN")
        snaps.append(_snap(t, etat)); t += 1
    for _ in range(stable - 1):
        snaps.append(_snap(t, etat)); t += 1
    return TimelinePoste(data["voltage_level_id"], snaps), data


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def test_topologie_id_canonique():
    a = {"s1": True, "s2": False}
    b = {"s2": False, "s1": True}
    assert topologie_id(a) == topologie_id(b)
    assert topologie_id(a) != topologie_id({"s1": False, "s2": False})


def test_aucun_bloc_sans_changement():
    tl = TimelinePoste("VL", [_snap(i, {"s": False}) for i in range(5)])
    blocs, osc = tl.detecter_blocs()
    assert blocs == [] and osc == []


def test_oscillation_repliee_pas_de_bloc():
    """A stable → bruit bref → A stable : aucune transition, une oscillation."""
    a, b = {"s": False}, {"s": True}
    snaps = ([_snap(i, a) for i in range(3)]
             + [_snap(3, b)]                       # bruit (1 snapshot)
             + [_snap(i, a) for i in range(4, 7)])
    blocs, osc = TimelinePoste("VL", snaps).detecter_blocs(min_stabilite=2)
    assert blocs == []
    assert len(osc) == 1 and osc[0].nb_snapshots == 1


def test_deux_transitions_donnent_deux_blocs_et_reversibilite():
    """A → B → A : deux blocs ; le premier est réversible (retour observé)."""
    a, b = {"s": False, "u": False}, {"s": True, "u": False}
    snaps = ([_snap(i, a) for i in range(3)]
             + [_snap(i, b) for i in range(3, 6)]
             + [_snap(i, a) for i in range(6, 9)])
    blocs, _ = TimelinePoste("VL", snaps).detecter_blocs(min_stabilite=2)
    assert len(blocs) == 2
    assert blocs[0].retour_observe is True       # A revient ensuite
    assert blocs[1].retour_observe is False
    assert blocs[0].manoeuvres_observees == [
        {"timestamp": "2021-01-01T0003", "switch_id": "s", "action": "OPEN"}]


def test_bornes_instables_ignorees():
    """Sans second plateau stable, pas de bloc (pas de borne cible)."""
    a = {"s": False}
    snaps = [_snap(0, a), _snap(1, a), _snap(2, {"s": True})]
    blocs, _ = TimelinePoste("VL", snaps).detecter_blocs(min_stabilite=2)
    assert blocs == []


# ---------------------------------------------------------------------------
# Bout en bout sur une séquence réelle du dépôt (CARRIP3, 20 manœuvres)
# ---------------------------------------------------------------------------

def test_bloc_reproduit_la_sequence_reelle():
    tl, data = _timeline_depuis_sequence(SEQ_CARRIP3)
    blocs, osc = tl.detecter_blocs(min_stabilite=2)
    assert len(blocs) == 1 and osc == []
    b = blocs[0]
    assert b.voltage_level_id == "CARRIP3"
    assert b.etats_depart == {k: bool(v) for k, v in data["depart"].items()}
    assert b.etats_cible == {k: bool(v) for k, v in data["cible"].items()}
    # La séquence observée reproduit la séquence réelle, dans l'ordre.
    attendu = [(m["switch_id"], m["action"])
               for m in sorted(data["manoeuvres"], key=lambda m: m["ordre"])]
    assert [(m["switch_id"], m["action"])
            for m in b.manoeuvres_observees] == attendu
    assert len(b.transitoires) == len(attendu) - 1


def test_tagging_structurel_fusion_carrip3():
    """CARRIP3 cible 1 nœud : fusion de nœuds + remise en service (les départs
    isolés au départ sont reconnectés dans la cible 1 nœud)."""
    tl, _ = _timeline_depuis_sequence(SEQ_CARRIP3)
    blocs, _ = tl.detecter_blocs()
    poste = poste_from_fixture_json(FIXTURES / "CARRIP3.json", "CARRIP3")
    tags = taguer_bloc(blocs[0], poste)
    assert "fusion_noeuds" in tags
    assert "scission_noeud" not in tags
    assert blocs[0].tags == tags                 # mutation consignée


def test_tagging_structurel_consignation():
    """Ouvrir DJ + SA d'une cellule de départ → consignation_ouvrage ;
    le bloc inverse → remise_en_service."""
    poste = poste_from_fixture_json(FIXTURES / "CARRIP3.json", "CARRIP3")
    depart = CibleDetaillee.from_graph(poste.graph, "CARRIP3").etats_organes
    cell = next(c for c in poste.cellules.cellules_depart
                if c.breakers and c.disconnectors)
    cible = dict(depart)
    for sw in list(cell.breakers) + list(cell.disconnectors):
        cible[sw.switch_id] = True
    snaps = ([_snap(i, depart) for i in range(2)]
             + [_snap(i, cible) for i in range(2, 4)]
             + [_snap(i, depart) for i in range(4, 6)])
    blocs, _ = TimelinePoste("CARRIP3", snaps).detecter_blocs(min_stabilite=2)
    assert len(blocs) == 2
    assert "consignation_ouvrage" in taguer_bloc(blocs[0], poste)
    assert "remise_en_service" in taguer_bloc(blocs[1], poste)


def test_tagging_par_nommage_sans_structure():
    """Repli heuristique : COUPL ouvert → scission ; DJ+SA d'un même ouvrage
    ouverts → consignation."""
    depart = {"CARRIP3_CARRI3COUPL.1 DJ_OC": False,
              "CARRIP3_BERT.L31CARRI DJ_OC": False,
              "CARRIP3_BERT.L31CARRI SA1_OC": False}
    cible = {"CARRIP3_CARRI3COUPL.1 DJ_OC": True,
             "CARRIP3_BERT.L31CARRI DJ_OC": True,
             "CARRIP3_BERT.L31CARRI SA1_OC": True}
    snaps = ([_snap(i, depart) for i in range(2)]
             + [_snap(i, cible) for i in range(2, 4)])
    blocs, _ = TimelinePoste("CARRIP3", snaps).detecter_blocs(min_stabilite=2)
    tags = taguer_bloc(blocs[0], poste=None)
    assert "scission_noeud" in tags
    assert "consignation_ouvrage" in tags


def test_tag_reconfiguration_durable():
    tl, _ = _timeline_depuis_sequence(SEQ_CARRIP3, stable=4)
    blocs, _ = tl.detecter_blocs()
    tags = taguer_bloc(blocs[0], poste=None, seuil_durable=3)
    assert "reconfiguration_durable" in tags


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def test_extraction_scenario_et_sequence(tmp_path):
    tl, data = _timeline_depuis_sequence(SEQ_CARRIP3)
    blocs, _ = tl.detecter_blocs()
    poste = poste_from_fixture_json(FIXTURES / "CARRIP3.json", "CARRIP3")
    taguer_bloc(blocs[0], poste)

    sc = bloc_to_scenario(blocs[0], poste)
    assert sc["voltage_level_id"] == "CARRIP3"
    assert sc["depart"] == blocs[0].etats_depart
    assert sc["cible"] == blocs[0].etats_cible
    assert sc["meta"]["tags"] == blocs[0].tags
    assert sc["meta"]["nb_manoeuvres_observees"] == 20
    # Partitions nodales jointes (poste fourni) — la cible vise 1 nœud barré +
    # les éventuels isolés résiduels.
    assert "depart_nodale" in sc and "cible_nodale" in sc

    sq = bloc_to_sequence_observee(blocs[0])
    assert sq["nb_manoeuvres"] == 20
    assert [m["ordre"] for m in sq["manoeuvres"]] == list(range(1, 21))

    ecrits = ecrire_dataset(blocs, tmp_path, {"CARRIP3": poste})
    assert (tmp_path / "scenarios").exists()
    assert any(p.parent.name == "sequences" for p in ecrits)
    # Round-trip : le scénario écrit est rechargeable et cohérent.
    rt = json.loads(ecrits[0].read_text())
    assert rt["cible"] == {k: bool(v) for k, v in data["cible"].items()}

    st = stats_blocs(blocs)
    assert st["nb_blocs"] == 1 and st["blocs_avec_sequence_observee"] == 1
    assert st["par_tag"]


# ---------------------------------------------------------------------------
# Adaptateur dgitt (CSV événements minimal)
# ---------------------------------------------------------------------------

def test_dgitt_charge_un_csv_evenements(tmp_path):
    pd = pytest.importorskip("pandas")
    rows = [
        # état initial complet à t0, puis événements
        ("2021-01-01T00:00", "VLTEST", "DJ1", "closed"),
        ("2021-01-01T00:00", "VLTEST", "SA1", "closed"),
        ("2021-01-01T00:05", "VLTEST", "DJ1", "closed"),
        ("2021-01-01T00:10", "VLTEST", "DJ1", "open"),
        ("2021-01-01T00:15", "VLTEST", "SA1", "open"),
        ("2021-01-01T00:20", "VLTEST", "SA1", "open"),
        ("2021-01-01T00:25", "VLTEST", "DJ1", "open"),
    ]
    pd.DataFrame(rows, columns=["timestamp", "voltage_level_id",
                                "switch_id", "state"]).to_csv(
        tmp_path / "events.csv", index=False)

    from expert_op4grid_recommender.manoeuvre.dataset.dgitt import (
        charger_timelines,
    )
    tls = list(charger_timelines(tmp_path))
    assert len(tls) == 1
    tl = tls[0]
    assert tl.voltage_level_id == "VLTEST"
    blocs, _ = tl.detecter_blocs(min_stabilite=2)
    assert len(blocs) == 1
    b = blocs[0]
    assert b.etats_depart == {"DJ1": False, "SA1": False}
    assert b.etats_cible == {"DJ1": True, "SA1": True}
    assert [(m["switch_id"], m["action"]) for m in b.manoeuvres_observees] == [
        ("DJ1", "OPEN"), ("SA1", "OPEN")]


def test_dgitt_colonnes_inconnues_message_clair(tmp_path):
    pd = pytest.importorskip("pandas")
    pd.DataFrame([[1, 2]], columns=["foo", "bar"]).to_csv(
        tmp_path / "x.csv", index=False)
    from expert_op4grid_recommender.manoeuvre.dataset.dgitt import (
        charger_timelines,
    )
    with pytest.raises(ValueError, match="Colonnes non identifiées"):
        list(charger_timelines(tmp_path))
