"""
tests/manoeuvre/test_ihm_ui_improvements.py
--------------------------------------------
Tests des améliorations d'IHM manœuvre :

1. **Import local désactivé** sur une IHM déportée (Space) : ``/api/pick_grid_file``
   et ``/api/load_grid`` renvoient ``403`` quand ``DATASET["hosted"]`` ;
2. **Modale de config** : ``GET/POST /api/config`` (algos par phase + chemins de
   sauvegarde/rechargement des scénarios/séquences) ;
3. **Date/heure cible** conservée dans le scénario (``meta.cible_dt``) ;
4. **Déclaration d'ouvrages isolés** : un départ en service déclaré isolé est
   réellement **déconnecté** (nœud 0-barre) par ``nodale_to_detaillee``.

Nécessitent ``flask`` et ``pypowsybl`` (sinon test ignoré).
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

pytest.importorskip("flask")
pytest.importorskip("pypowsybl")

import pypowsybl as pp  # noqa: E402

_IHM_PATH = (pathlib.Path(__file__).resolve().parents[2]
             / "scripts" / "manoeuvre_ihm.py")
VL = "S1VL2"


def _load_ihm():
    spec = importlib.util.spec_from_file_location("manoeuvre_ihm_ui_mod", _IHM_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ihm = _load_ihm()


@pytest.fixture()
def session():
    net = pp.network.create_four_substations_node_breaker_network()
    s = ihm.Session(net)
    s.load(VL)
    return s


def _feeders(s):
    return {eq for g in s.groups_of(s.initial) for eq in g}


# --------------------------------------------------------------------------
# 1. Import local désactivé sur IHM déportée (Space)
# --------------------------------------------------------------------------

def test_pick_grid_file_refused_when_hosted(monkeypatch):
    monkeypatch.setitem(ihm.DATASET, "hosted", True)
    r = ihm.app.test_client().get("/api/pick_grid_file")
    assert r.status_code == 403
    assert r.get_json()["hosted"] is True


def test_load_grid_refused_when_hosted(monkeypatch):
    monkeypatch.setitem(ihm.DATASET, "hosted", True)
    r = ihm.app.test_client().post("/api/load_grid", json={"path": "/x.xiidm"})
    assert r.status_code == 403
    assert r.get_json()["hosted"] is True


def test_load_grid_allowed_when_not_hosted(monkeypatch):
    # Non hébergée : l'erreur est « fichier introuvable » (400), pas un refus.
    monkeypatch.setitem(ihm.DATASET, "hosted", False)
    r = ihm.app.test_client().post("/api/load_grid", json={"path": "/nope.xiidm"})
    assert r.status_code == 400
    assert not r.get_json().get("hosted")


# --------------------------------------------------------------------------
# 2. Modale de configuration : /api/config
# --------------------------------------------------------------------------

def test_config_get(session, monkeypatch):
    monkeypatch.setattr(ihm, "SESSION", session)
    d = ihm.app.test_client().get("/api/config").get_json()
    assert "disponibles" in d["algos"] and "selection" in d["algos"]
    assert "libtopo" in d["algos"]["selection"]["sequenceur"]
    assert d["scenarios_dir"] and d["sequences_dir"]
    assert "hosted" in d


def test_config_post_updates_dirs(session, monkeypatch, tmp_path):
    monkeypatch.setattr(ihm, "SESSION", session)
    monkeypatch.setitem(ihm.DATASET, "hosted", False)
    sd, qd = tmp_path / "scen", tmp_path / "seq"
    d = ihm.app.test_client().post("/api/config", json={
        "scenarios_dir": str(sd), "sequences_dir": str(qd)}).get_json()
    assert d["scenarios_dir"] == str(sd)
    assert d["sequences_dir"] == str(qd)
    assert ihm.SCEN_DIR == sd and ihm.SEQ_DIR == qd


def test_config_post_dirs_readonly_when_hosted(session, monkeypatch, tmp_path):
    monkeypatch.setattr(ihm, "SESSION", session)
    monkeypatch.setitem(ihm.DATASET, "hosted", True)
    before_scen, before_seq = ihm.SCEN_DIR, ihm.SEQ_DIR
    ihm.app.test_client().post("/api/config", json={
        "scenarios_dir": str(tmp_path / "x"), "sequences_dir": str(tmp_path / "y")})
    assert ihm.SCEN_DIR == before_scen and ihm.SEQ_DIR == before_seq


# --------------------------------------------------------------------------
# 3. Date/heure de la topologie cible conservée dans le scénario
# --------------------------------------------------------------------------

def test_scenario_keeps_depart_and_cible_dt(session, monkeypatch, tmp_path):
    monkeypatch.setattr(ihm, "SCEN_DIR", tmp_path)
    res = session.save_scenario("sc", depart_dt="2021-01-03T08:00",
                                source="rte", cible_dt="2021-01-03T12:00")
    data = json.loads((tmp_path / f"{res['name']}.json").read_text())
    assert data["meta"]["dt"] == "2021-01-03T08:00"        # topo de départ
    assert data["meta"]["cible_dt"] == "2021-01-03T12:00"  # topo cible (RTE7000)
    # exposée par la liste de scénarios (modale Recharger).
    listed = {s["name"]: s for s in session.list_scenarios()}
    assert listed[res["name"]]["cible_dt"] == "2021-01-03T12:00"


def test_scenario_cible_dt_none_when_modified(session, monkeypatch, tmp_path):
    monkeypatch.setattr(ihm, "SCEN_DIR", tmp_path)
    res = session.save_scenario("sc2", depart_dt="2021-01-03T08:00", source="rte")
    data = json.loads((tmp_path / f"{res['name']}.json").read_text())
    assert data["meta"]["cible_dt"] is None


# --------------------------------------------------------------------------
# 4. Déclaration d'ouvrages isolés -> réellement déconnectés
# --------------------------------------------------------------------------

def test_declared_isolated_feeder_is_disconnected(session):
    feeders = sorted(_feeders(session))
    assert len(feeders) >= 2
    iso = [feeders[0]]
    rest = feeders[1:]
    # Le départ déclaré isolé était en service (S1VL2 pristine = 1 nœud).
    assert session.nodale_state(session.initial)["isolated"] == []
    res = session.nodale_to_detaillee([rest], iso)
    realised = res["nodale"]
    assert iso[0] in realised["isolated"]                 # réellement déconnecté
    # Côté affichage (dropIso du front : isolés retirés des nœuds) → plus un nœud.
    iso_set = set(realised["isolated"])
    flat = {eq for g in realised["groups"] for eq in g if eq not in iso_set}
    assert iso[0] not in flat
    # Et il n'est pas compté dans le nombre de nœuds réels.
    assert res["nb_noeuds"] == len([g for g in realised["groups"]
                                    if any(eq not in iso_set for eq in g)])


def test_isoler_dans_etat_opens_incident_switches(session):
    feeders = sorted(_feeders(session))
    f = feeders[0]
    state = dict(session.initial)
    out = session._isoler_dans_etat(state, [f])
    # L'opération ne mute pas l'entrée et déconnecte f.
    assert out is not state
    assert f in session.nodale_state(out)["isolated"]


# --------------------------------------------------------------------------
# 5. nodale_to_detaillee : diff vert/orange + compte de nœuds correct
# --------------------------------------------------------------------------

def test_to_detaillee_returns_changes_for_highlight(session):
    # Le pont nodal → détaillé doit renvoyer le diff départ→cible pour le
    # surlignage vert/orange (absent auparavant → pas de couleurs après calcul).
    feeders = sorted(_feeders(session))
    iso = [feeders[0]]
    res = session.nodale_to_detaillee([feeders[1:]], iso)
    assert "changes" in res and isinstance(res["changes"], list)
    # l'ouvrage isolé était en service → son organe passe ouvert → diff "opened".
    assert any(c["direction"] == "opened" for c in res["changes"])


def test_to_detaillee_node_count_excludes_isolated(session):
    # 2 isolés + reste sur 1 nœud : obtenu = nœuds RÉELS (sans les isolés), et le
    # message ne gonfle pas le compte (« obtenu 4 » au lieu de « 1 + 2 isolés »).
    feeders = sorted(_feeders(session))
    iso = feeders[:2]
    rest = feeders[2:]
    res = session.nodale_to_detaillee([rest], iso)
    assert res["nb_isoles"] == 2
    assert res["nb_obtenu"] == res["nb_vise"] == 1     # 1 nœud réel visé/obtenu
    assert res["is_verified"] is True
    # côté isolés, le message reste vide (le front compose « N nœud(s) + M isolés »)
    assert "obtenu" not in (res["message"] or "")
    # et aucun nombre supérieur au compte réel n'apparaît dans le message.
    assert "4" not in (res["message"] or "")


def test_to_detaillee_without_iso_keeps_algo_verdict(session):
    p = session.nodale_payload(session.initial)
    res = session.nodale_to_detaillee(p["groups"], [])
    assert res["nb_isoles"] == 0
    assert "changes" in res
    assert res["is_verified"] is True


def test_api_nodale_to_detaillee_endpoint_shape(session, monkeypatch):
    monkeypatch.setattr(ihm, "SESSION", session)
    feeders = sorted(_feeders(session))
    r = ihm.app.test_client().post("/api/nodale_to_detaillee",
                                   json={"groups": [feeders[1:]], "isolated": [feeders[0]]})
    d = r.get_json()
    for k in ("changes", "nb_isoles", "nb_obtenu", "nb_vise", "nodale"):
        assert k in d


# --------------------------------------------------------------------------
# 6. Données d'ordre de la vue nodale (axe gauche→droite = abscisse SLD)
# --------------------------------------------------------------------------

def test_nodale_order_data_present_for_all_feeders(session):
    # La vue nodale trie de gauche à droite par ``order`` (abscisse SLD) et place
    # le côté par ``dirs`` : ces deux champs doivent être fournis pour CHAQUE départ
    # (sinon repli 0/BOTTOM → ordre alphabétique parasite).
    p = session.nodale_payload(session.initial)
    feeders = [e for g in p["groups"] for e in g]
    assert feeders
    for eq in feeders:
        assert isinstance(p["order"].get(eq), (int, float))
        assert p["dirs"].get(eq) in ("TOP", "BOTTOM")
