"""
tests/manoeuvre/test_ihm_explore.py
-----------------------------------
Tests des endpoints « Explorer la journée » de l'IHM de manœuvre
(``/api/explore_poste``, ``/api/explore_retain_target``) et de la présence du
front-end carte dans l'asset HTML.

On construit une ``DayExploration`` synthétique (3 « heures » dérivées du réseau
de test, sans téléchargement HuggingFace) pour exercer la bascule en vue
topologique d'un poste à une heure et la rétention d'une cible.

Nécessite ``flask`` et ``pypowsybl`` (sinon ignoré).
"""
from __future__ import annotations

import copy
import importlib.util
import pathlib

import pytest

pytest.importorskip("flask")
pytest.importorskip("pypowsybl")

import pypowsybl as pp  # noqa: E402

from expert_op4grid_recommender.manoeuvre.dataset import exploration, geographie  # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_IHM_PATH = _ROOT / "scripts" / "manoeuvre_ihm.py"
_ASSET = _ROOT / "scripts" / "manoeuvre_ihm_assets" / "index.html"


def _load_ihm():
    spec = importlib.util.spec_from_file_location("manoeuvre_ihm_explore_mod", _IHM_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ihm = _load_ihm()


def _build_day(mod):
    """Exploration synthétique : 3 heures, un VL perturbé à midi."""
    net = pp.network.create_four_substations_node_breaker_network()
    etats, kinds = exploration.extraire_etats_kinds(net)
    vl_meta, sub_name = exploration.structure_reseau(net)
    vl0 = next(v for v in etats if etats[v])
    h0 = etats
    h12 = copy.deepcopy(etats)
    sid = next(iter(h12[vl0]))
    h12[vl0][sid] = not h12[vl0][sid]
    h23 = copy.deepcopy(etats)
    de = mod.DayExploration("2021-01-01", "repo")
    de.etats = {"00:00": h0, "12:00": h12, "23:00": h23}
    de.heures = [{"requested": h, "ts": h, "iso": "2021-01-01T" + h}
                 for h in ("00:00", "12:00", "23:00")]
    de.kinds, de.vl_meta, de.sub_name = kinds, vl_meta, sub_name
    de.struct = exploration.extraire_structure_topo(net)
    ch = exploration.changements_par_vl([h0, h12, h23], kinds)
    nodaux = exploration.changements_nodaux_par_vl([h0, h12, h23], de.struct)
    exploration.fusionner_nodaux(ch, nodaux)
    de.postes = exploration.agreger_par_poste(ch, vl_meta, sub_name)
    de.top = exploration.classer_postes(de.postes, 10)
    de.classement = exploration.classer_postes(de.postes, 40)
    de.positions, de.coord_source = geographie.resoudre(
        list(de.postes), net=net, autoriser_osm=False)
    return net, de, vl0


@pytest.fixture()
def client():
    net, de, vl0 = _build_day(ihm)
    ihm.SESSION = ihm.Session(net)
    ihm.DAY = de
    c = ihm.app.test_client()
    c._vl0 = vl0  # type: ignore[attr-defined]
    yield c
    ihm.DAY = None


def test_explore_payload_shape():
    _, de, vl0 = _build_day(ihm)
    payload = ihm._explore_payload(de)
    assert payload["ok"] and payload["date"] == "2021-01-01"
    assert payload["types_oc"] == list(exploration.TYPES_OC)
    assert payload["n_actifs"] >= 1
    assert isinstance(payload["classement"], list) and payload["classement"]
    # le poste actif figure en tête de classement.
    assert payload["classement"][0]["total"] >= 1
    # le décompte des re-groupements de nœuds est exposé (carte + classement).
    assert "nodal" in payload["classement"][0]
    assert all("nodal" in p for p in payload["postes"])
    # champs de résolution des coordonnées présents (carte / téléchargement).
    assert "coord_stats" in payload and "coord_file" in payload


def test_explore_coords_file_endpoint(tmp_path, monkeypatch):
    client = ihm.app.test_client()
    snap = tmp_path / "postes_rte_geo.json"
    monkeypatch.setattr(ihm, "GEO_SNAPSHOT", snap)
    # absent → 404
    assert client.get("/api/explore_coords_file").status_code == 404
    # présent → 200 + pièce jointe
    snap.write_text('{"S1": {"lat": 48.0, "lon": 2.0}}', encoding="utf-8")
    r = client.get("/api/explore_coords_file")
    assert r.status_code == 200
    assert "attachment" in r.headers.get("Content-Disposition", "")
    assert "S1" in r.get_data(as_text=True)


def test_explore_basemap_endpoint():
    r = ihm.app.test_client().get("/api/explore_basemap").get_json()
    assert r["depts"] and r["neighbors"]
    assert len(r["depts"][0][0]) == 2   # anneaux de points écran [x, y]


def test_explore_poste_par_vl(client):
    vl0 = client._vl0
    r = client.post("/api/explore_poste", json={"vl": vl0, "hour": "12:00"}).get_json()
    assert r["ok"] and r["vl"] == vl0 and r["hour"] == "12:00"
    assert r["initial_svg"] and r["svg"] and "switches" in r
    assert len(r["heures"]) == 3
    assert isinstance(r["sub_vls"], list)


def test_explore_poste_par_sub_defaut_vl_actif(client):
    vl0 = client._vl0
    sub = ihm.DAY.vl_meta[vl0]["substation"]
    r = client.post("/api/explore_poste", json={"sub": sub, "hour": "00:00"}).get_json()
    assert r["ok"] and r["sub"] == sub and r["vl"]


def test_explore_retain_target(client):
    vl0 = client._vl0
    client.post("/api/explore_poste", json={"vl": vl0, "hour": "12:00"})
    r = client.post("/api/explore_retain_target", json={"vl": vl0, "hour": "00:00"}).get_json()
    assert r["ok"] and r["hour"] == "00:00" and r["svg"]


def test_explore_changes_diff(client):
    """Différence départ/cible : nulle à iso-heure, non vide après rétention
    d'une cible à une autre heure ; chaque entrée a un sens et un ``svgId``."""
    vl0 = client._vl0
    r = client.post("/api/explore_poste", json={"vl": vl0, "hour": "12:00"}).get_json()
    # départ == cible (même heure) → aucun organe en écart.
    assert r["changes"] == []
    # cible = 00:00 (un organe basculé vs 12:00) → exactement un écart.
    r2 = client.post("/api/explore_retain_target",
                     json={"vl": vl0, "hour": "00:00"}).get_json()
    assert len(r2["changes"]) == 1
    ch = r2["changes"][0]
    assert ch["direction"] in ("closed", "opened")
    assert ch["svgId"] and ch["id"]


def test_nb_noeuds_reels_exclut_ouvrages_isoles():
    """Affichage : ``_nb_noeuds_reels`` ne compte que les nœuds avec barre — les
    ouvrages isolés (déconnectés) ne sont pas des nœuds (le moteur, lui, garde
    ``TopologieNodale.nb_noeuds``, isolés inclus)."""
    from expert_op4grid_recommender.manoeuvre.topologie import TopologieNodale
    net = pp.network.create_four_substations_node_breaker_network()
    s = ihm.Session(net)
    strict = False
    for vl in list(net.get_voltage_levels().index):
        try:
            s.load(vl)
        except Exception:
            continue
        if not s.current:
            continue
        allopen = {sid: True for sid in s.current}   # tout ouvrir → départs isolés
        s.apply(allopen)
        G = s._graph(allopen)
        reels = ihm._nb_noeuds_reels(G)
        full = TopologieNodale.from_graph(G, vl).nb_noeuds
        assert reels <= full
        if reels < full:
            strict = True
    assert strict   # au moins un VL où des ouvrages isolés ne sont pas comptés


def test_explore_poste_guard_sans_day():
    ihm.DAY = None
    ihm.SESSION = ihm.Session(pp.network.create_four_substations_node_breaker_network())
    r = ihm.app.test_client().post("/api/explore_poste", json={"vl": "x"})
    assert r.status_code == 400 and r.get_json()["ok"] is False


# --- front-end carte présent dans l'asset -----------------------------------

def test_asset_contient_la_carte():
    txt = _ASSET.read_text(encoding="utf-8")
    for token in ("exploreDay", "renderExplore", "buildMap", "mapToTopo",
                  "id=\"mapPane\"", "id=\"exploreBar\"", "id=\"btnExplore\"",
                  "Explorer la journée", "convexHull", "frhull",
                  "id=\"mapHours\"", "renderMapHours",
                  # filtrage des tensions par la légende
                  "voltToggle", "voltAll", "voltBand", "voltOff",
                  # mise en évidence des écarts départ/cible
                  "highlightChanges", "octog-closed", "octog-opened",
                  # décompte des re-groupements de nœuds (scission/fusion)
                  "re-groupé", "nodbadge",
                  # nom de scénario formaté + sélection poste hors carte
                  "defaultScenName", "topoDepart", "topoCible", "observee",
                  # recherche de poste utilisable pendant l'exploration
                  "selectPoste"):
        assert token in txt, f"jeton front-end manquant : {token}"
