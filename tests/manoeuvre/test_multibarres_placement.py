"""
tests/manoeuvre/test_multibarres_placement.py
---------------------------------------------
Caractérisation du **chemin > 2 jeux de barres** (postes 3B/4B) et du **placement
dégradé** — code exercé par aucun golden (les scénarios sauvegardés passent par la
voie *détaillée* / ``_sequence_detaillee_multibarres``, pas par
``determiner_topo_complete_cible`` → ``_placement_automatique`` sur > 2 barres).

Mis à jour à l'**Étape 1+2** (placement généralisé à N jeux de barres) : le
scoping « 2 jeux de barres » de ``_placement_automatique`` a été retiré, donc le
placement n'abandonne plus le 3ᵉ/4ᵉ nœud à l'opérateur. Ces postes peuvent
désormais soit être placés complètement (GUARBP6/MORBRP6), soit être refusés
pour une **vraie raison d'infaisabilité** (CORNIP3 : condition de Hall), mais
**jamais** par scoping. ``_main_busbar_sjb`` reste un utilitaire exporté.

Postes réels à ≥ 3 barres disponibles en fixtures : CORNIP3, GUARBP6, MORBRP6
(en réalité 4 barres « double-barre + 2 barres de transfert »).
"""

from __future__ import annotations

import pytest

from expert_op4grid_recommender.manoeuvre import (
    PosteTopologique,
    TopologieNodale,
    determiner_topo_complete_cible,
    sectionneurs_sous_charge_par_manoeuvre,
)
from expert_op4grid_recommender.manoeuvre.algo import _main_busbar_sjb

from .fixture_loader import build_graph_from_fixture, list_available_fixtures


POSTES_3B = ["CORNIP3", "GUARBP6", "MORBRP6"]

pytestmark = pytest.mark.skipif(
    not all(p in list_available_fixtures() for p in POSTES_3B),
    reason="Fixtures de postes ≥ 3 barres absentes (CORNIP3/GUARBP6/MORBRP6).",
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _poste(name):
    return PosteTopologique.from_graph(build_graph_from_fixture(name), name)


def _switch_states(G):
    """Instantané {switch_id: open} de toutes les arêtes-organes du graphe."""
    return {d["switch_id"]: bool(d.get("open", False))
            for _, _, d in G.edges(data=True) if d.get("switch_id")}


def _known_switch_ids(G):
    return {d.get("switch_id") for _, _, d in G.edges(data=True) if d.get("switch_id")}


def _split_target(poste, name):
    """Cible nodale obtenue en scindant le plus gros nœud courant en deux —
    construit une demande qui sollicite le placement (et sa dégradation 3B/4B)."""
    groups = [sorted(n.equipment_ids) for n in poste.topologie_nodale.noeuds.values()]
    big = max(groups, key=len)
    assert len(big) >= 2, f"{name} : nœud trop petit pour scinder"
    newg = [g for g in groups if g is not big]
    newg.append(big[:1])
    newg.append(big[1:])
    return TopologieNodale.from_node_groups(name, newg)


# --------------------------------------------------------------------------
# _main_busbar_sjb : sélection des 2 jeux de barres principaux
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", POSTES_3B)
def test_main_busbar_sjb_partition(name):
    poste = _poste(name)
    barre_par = poste.tronconnement.barre_par_busbar
    assert len(set(barre_par.values())) > 2, f"{name} doit avoir > 2 barres"

    main_sjb, extra_sjb = _main_busbar_sjb(poste)

    # Partition stricte de l'ensemble des SJB.
    assert main_sjb.isdisjoint(extra_sjb)
    assert main_sjb | extra_sjb == set(barre_par)
    assert extra_sjb, "un poste > 2 barres doit avoir des SJB supplémentaires"

    # main_sjb = exactement les SJB des 2 barres les plus peuplées (tie-break index).
    by_barre = {}
    for s, b in barre_par.items():
        by_barre.setdefault(b, set()).add(s)
    expected_main_barres = set(sorted(by_barre, key=lambda b: (-len(by_barre[b]), b))[:2])
    expected_main_sjb = {s for b in expected_main_barres for s in by_barre[b]}
    assert main_sjb == expected_main_sjb


# --------------------------------------------------------------------------
# determiner_topo_complete_cible : cible identité (déjà satisfaite)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", POSTES_3B)
def test_identity_target_is_trivially_verified(name):
    poste = _poste(name)
    before = _switch_states(poste.graph)

    res = determiner_topo_complete_cible(poste, poste.topologie_nodale)

    assert res.is_verified is True
    assert res.nb_manoeuvres == 0
    assert res.noeuds_non_realisables == []
    assert "satisfait déjà" in res.message
    # Cas trivial : ne mute jamais le graphe du poste.
    assert _switch_states(poste.graph) == before


# --------------------------------------------------------------------------
# determiner_topo_complete_cible : cible scindée -> dégradation gracieuse 3B/4B
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", POSTES_3B)
def test_split_target_plus_de_scoping_2jdb(name):
    """Étape 1+2 : plus aucune dégradation par scoping « 2 jeux de barres ».
    Le placement réalise la cible, ou la refuse pour une vraie infaisabilité."""
    poste = _poste(name)
    before = _switch_states(poste.graph)
    target = _split_target(poste, name)

    res = determiner_topo_complete_cible(poste, target)

    # 1. Le message de scoping a disparu (cœur du changement Étape 1).
    assert "niveaux de barres supplémentaires" not in res.message
    assert "algorithme 2 jeux de barres" not in res.message

    # 2. Si des nœuds restent non réalisés, c'est pour une **vraie**
    #    infaisabilité (condition de Hall / organe), pas par scoping.
    if res.noeuds_non_realisables:
        assert "non réalisable" in res.message

    # 3. Invariant fort exploité par tout le séquenceur : poste.graph non muté.
    assert _switch_states(poste.graph) == before

    # 4. Toute manœuvre émise référence un organe **existant** du poste.
    known = _known_switch_ids(poste.graph)
    assert all(m.switch_id in known for m in res.manoeuvres)

    # 5. Le vérificateur public de règle s'applique et renvoie une liste alignée.
    viol = sectionneurs_sous_charge_par_manoeuvre(poste, res.manoeuvres)
    assert len(viol) == len(res.manoeuvres)
    assert all(v is None or isinstance(v, str) for v in viol)


@pytest.mark.parametrize("name", POSTES_3B)
def test_resume_renders_summary(name):
    poste = _poste(name)
    res = determiner_topo_complete_cible(poste, _split_target(poste, name))

    txt = res.resume()
    assert isinstance(txt, str)
    lines = txt.splitlines()
    assert lines[0].startswith(f"Manœuvres VL '{name}'")
    # Le résumé liste chaque manœuvre puis le message de diagnostic.
    assert f"{res.nb_manoeuvres} OC" in lines[0]
    assert any(res.message[:30] in ln for ln in lines)


def test_idempotent_on_repeated_calls():
    """Rejouer la même demande ne dépend pas d'un état résiduel (le poste n'est
    pas muté entre deux appels) : sorties identiques."""
    name = "MORBRP6"
    poste = _poste(name)
    target = _split_target(poste, name)
    r1 = determiner_topo_complete_cible(poste, target)
    r2 = determiner_topo_complete_cible(poste, target)
    seq1 = [(m.switch_id, m.action) for m in r1.manoeuvres]
    seq2 = [(m.switch_id, m.action) for m in r2.manoeuvres]
    assert seq1 == seq2
    assert r1.is_verified == r2.is_verified
    assert r1.message == r2.message
