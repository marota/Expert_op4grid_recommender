"""
tests/manoeuvre/test_geographie.py
----------------------------------
Tests du résolveur de coordonnées ``manoeuvre.dataset.geographie`` : plan de masse
committé (`positions_from_layout`), appariement OSM/ODRE (formes de géo variées),
projection Mercator, chargement d'instantané committé, et chaîne de résolution.

Aucun accès réseau : OSM/ODRE sont simulés par des enregistrements en mémoire ;
l'extension ``substationPosition`` (absente du dataset RTE 7000 réel) est
couverte si pypowsybl est disponible.
"""
from __future__ import annotations

import json

import pytest

from expert_op4grid_recommender.manoeuvre.dataset import geographie as g


def test_positions_from_layout():
    # plan de masse par VL → coord par poste (VL de plus haute tension retenu).
    layout = {"AP7": [10.0, -20.0], "AP6": [11.0, -21.0], "BP3": [30.0, -40.0]}
    vl_meta = {
        "AP7": {"substation": "A", "nominal_v": 400.0},
        "AP6": {"substation": "A", "nominal_v": 225.0},
        "BP3": {"substation": "B", "nominal_v": 63.0},
        "CP3": {"substation": "C", "nominal_v": 63.0},   # absent du layout
    }
    pos, stats = g.positions_from_layout(layout, vl_meta)
    assert pos["A"] == {"x": 10.0, "y": -20.0, "source": "layout"}   # 400 kV gagne
    assert pos["B"]["x"] == 30.0
    assert "C" not in pos
    assert stats["n_substations"] == 3 and stats["n_apparies"] == 2


def test_charger_layout_bundle():
    # le plan de masse RTE committé est présent et non vide.
    layout = g.charger_layout()
    assert isinstance(layout, dict) and len(layout) > 1000
    assert g.charger_layout("/inexistant.json") == {}


def test_charger_basemap_bundle():
    # le fond de carte committé (départements + voisins) est présent et structuré.
    bm = g.charger_basemap()
    assert isinstance(bm, dict) and bm.get("depts") and bm.get("neighbors")
    ring = bm["depts"][0]
    assert len(ring) >= 3 and len(ring[0]) == 2   # anneaux de points [x, y]
    assert g.charger_basemap("/inexistant.json") == {}


def test_merc():
    x, y = g.merc(0.0, 0.0)
    assert abs(x) < 1e-6 and abs(y) < 1e-6
    assert g.merc(180.0, 0.0)[0] > 2e7          # ~ demi-circonférence


def test_normaliser_mnemonique():
    assert g.normaliser_mnemonique(".CTLH") == "CTLH"
    assert g.normaliser_mnemonique(".G.RO") == "GRO"
    assert g.normaliser_mnemonique(" carri-p ") == "CARRIP"
    assert g.normaliser_mnemonique("") == ""


def test_extraire_geo_formes():
    assert g._extraire_geo({"geo_point_2d": [48.7, 6.2]}) == (48.7, 6.2)
    assert g._extraire_geo({"geo_point_2d": {"lat": 45.1, "lon": 5.7}}) == (45.1, 5.7)
    assert g._extraire_geo({"geo_point_2d": "43.3,-1.4"}) == (43.3, -1.4)
    # GeoJSON [lon, lat] sous coordinates → (lat, lon)
    assert g._extraire_geo({"geom": {"coordinates": [5.7, 45.1]}}) == (45.1, 5.7)
    assert g._extraire_geo({"autre": 1}) is None


def test_apparier_odre_exact_et_prefixe():
    subs = [".CTLH", ".G.RO", ".NAVA", "CARRIP"]
    recs = [
        {"code_poste": "CTLH", "nom_poste": "CHATEAU", "tension": "63 kV",
         "geo_point_2d": [48.7, 6.2]},
        {"code": "GRO", "libelle": "GRO", "geo_point_2d": {"lat": 45.1, "lon": 5.7}},
        {"code": "NAVA", "nom": "NAVARRE", "geo_point_2d": "43.3,-1.4"},
        {"code": "ZZZZ", "geo_point_2d": [1.0, 1.0]},
    ]
    pos, stats = g.apparier_odre(recs, subs)
    assert pos[".CTLH"]["lat"] == 48.7 and pos[".CTLH"]["source"] == "osm"
    assert pos[".G.RO"]["lat"] == 45.1            # .G.RO → GRO (normalisé)
    assert pos[".NAVA"]["lon"] == -1.4
    assert "CARRIP" not in pos                    # aucun code correspondant
    assert stats["n_apparies"] == 3
    assert stats["n_substations"] == 4
    assert 0.0 < stats["taux"] <= 1.0


def test_apparier_par_nom_et_prefixe():
    # ODRE n'expose qu'un **nom** (CONCARNEAU) là où le réseau porte le mnémonique
    # (CONCA) : appariement par nom + repli préfixe.
    recs = [{"nom_poste": "CONCARNEAU", "geo_point_2d": [47.9, -3.9]},
            {"libelle": "TIVERVAL", "geo_point_2d": [48.7, 1.9]}]
    pos, stats = g.apparier_odre(recs, ["CONCA", "TIVER", "ZZZZZ"])
    assert pos["CONCA"]["lat"] == 47.9
    assert pos["TIVER"]["lat"] == 48.7
    assert "ZZZZZ" not in pos
    assert stats["n_index_nom"] == 2 and stats["n_apparies"] == 2
    # diagnostic présent pour le cas « 0 apparié ».
    assert "sample_fields" in stats and "sample_subs" in stats


def test_apparier_diagnostic_zero():
    recs = [{"code_poste": "9999", "geo_point_2d": [1.0, 2.0]}]
    pos, stats = g.apparier_odre(recs, ["CONCA", "TIVER"])
    assert pos == {} and stats["n_apparies"] == 0 and stats["n_records"] == 1
    assert stats["sample_codes"] == ["9999"]
    assert set(stats["sample_subs"]) == {"CONCA", "TIVER"}


def test_apparier_sans_prefixe():
    # CARRIP vs code "CARRI" : seulement via préfixe (désactivable).
    recs = [{"code": "CARRI", "geo_point_2d": [47.0, 1.0]}]
    pos_pref, _ = g.apparier_odre(recs, ["CARRIP"], prefix_fallback=True)
    pos_strict, _ = g.apparier_odre(recs, ["CARRIP"], prefix_fallback=False)
    assert "CARRIP" in pos_pref
    assert "CARRIP" not in pos_strict


def test_charger_snapshot(tmp_path):
    p = tmp_path / "snap.json"
    p.write_text(json.dumps({".CTLH": {"lat": 48.7, "lon": 6.2, "nom": "X"},
                             ".BAD": {"nom": "no coords"}}))
    snap = g.charger_snapshot(p)
    assert snap[".CTLH"]["source"] == "snapshot" and snap[".CTLH"]["lat"] == 48.7
    assert ".BAD" not in snap                     # pas de coordonnées → ignoré
    assert g.charger_snapshot(tmp_path / "absent.json") == {}


def test_resoudre_chaine_snapshot_puis_aucune(tmp_path):
    p = tmp_path / "snap.json"
    p.write_text(json.dumps({"S1": {"lat": 1.0, "lon": 2.0}}))
    pos, src = g.resoudre(["S1", "S2"], net=None, snapshot_path=p,
                          autoriser_osm=False)
    assert src == "snapshot" and "S1" in pos and "S2" not in pos
    # aucune source disponible → ('aucune', {})
    pos2, src2 = g.resoudre(["S1"], net=None, snapshot_path=tmp_path / "x.json",
                            autoriser_osm=False)
    assert src2 == "aucune" and pos2 == {}


def test_fetch_odre_geojson(monkeypatch, tmp_path):
    # ODRE n'expose la géométrie que via l'export geojson : on aplatit chaque
    # feature en record + geo_point_2d (centroïde) à côté des properties.
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [2.35, 48.85]},
         "properties": {"code_poste": "PARIS", "nom_poste": "PARIS"}},
        {"type": "Feature",
         "geometry": {"type": "Polygon",
                      "coordinates": [[[4.8, 45.7], [4.9, 45.7], [4.9, 45.8], [4.8, 45.8], [4.8, 45.7]]]},
         "properties": {"code_poste": "LYON", "nom_poste": "LYON"}},
        {"type": "Feature", "geometry": None, "properties": {"code_poste": "NOGEO"}}]}
    monkeypatch.setattr(g, "_http_get", lambda *a, **k: json.dumps(fc).encode("utf-8"))
    recs = g.fetch_odre_records(cache_dir=tmp_path)
    assert recs[0]["geo_point_2d"] == [48.85, 2.35]          # Point → [lat, lon]
    assert abs(recs[1]["geo_point_2d"][0] - 45.75) < 0.05    # Polygon → centroïde
    assert "geo_point_2d" not in recs[2]                     # géométrie nulle
    assert (tmp_path / "odre_postes_geo.json").exists()      # nom de cache distinct


def test_apparier_code_brut_avec_point():
    # code RTE à point (B.MAN) : match **verbatim** (la normalisation donnerait
    # BMAN et pourrait collisionner).
    recs = [{"code_poste": "B.MAN", "nom_poste": "BEAUMANOIR", "geo_point_2d": [48.1, -2.0]}]
    pos, stats = g.apparier_odre(recs, ["B.MAN", "B.MAU"])
    assert pos["B.MAN"]["lat"] == 48.1 and stats["n_exact"] == 1
    assert "B.MAU" not in pos


def test_centroid():
    assert g._centroid({"type": "Point", "coordinates": [2.0, 48.0]}) == (2.0, 48.0)
    c = g._centroid({"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2]]]})
    assert c == (1.0, 1.0)
    assert g._centroid(None) is None and g._centroid({"type": "Point"}) is None


def test_ecrire_snapshot_roundtrip(tmp_path):
    p = tmp_path / "geo.json"
    g.ecrire_snapshot(p, {"S1": {"lat": 48.0, "lon": 2.0, "nom": "X",
                                 "tension": "400 kV", "source": "odre"}})
    snap = g.charger_snapshot(p)
    assert snap["S1"]["lat"] == 48.0 and snap["S1"]["nom"] == "X"
    assert snap["S1"]["source"] == "snapshot"   # rechargé → source snapshot


def test_resoudre_osm_persiste_et_stats(tmp_path, monkeypatch):
    # OSM simulé ; resoudre doit apparier, peupler stats_out et persister le
    # snapshot (→ source 'osm' la 1ʳᵉ fois, puis 'snapshot' au tour suivant).
    recs = [{"code_poste": "S1", "nom_poste": "UN", "geo_point_2d": [48.0, 2.0]}]
    monkeypatch.setattr(g, "fetch_osm_substations", lambda *a, **k: recs)
    persist = tmp_path / "geo.json"
    stats: dict = {}
    pos, src = g.resoudre(["S1", "S2"], net=None,
                          snapshot_path=tmp_path / "absent.json",
                          persist_path=persist, stats_out=stats)
    assert src == "osm" and pos["S1"]["lat"] == 48.0 and "S2" not in pos
    assert stats["n_apparies"] == 1 and "taux" in stats
    assert persist.exists()
    # 2ᵉ résolution : le snapshot persisté prime, sans repasser par OSM.
    monkeypatch.setattr(g, "fetch_osm_substations",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no net")))
    pos2, src2 = g.resoudre(["S1"], net=None, snapshot_path=persist)
    assert src2 == "snapshot" and pos2["S1"]["lat"] == 48.0


def test_resoudre_osm_injoignable_diagnostic(tmp_path, monkeypatch):
    # Échec OSM → source 'aucune' + geo_error renseigné (pas d'exception).
    monkeypatch.setattr(g, "fetch_osm_substations",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("blocked")))
    stats: dict = {}
    pos, src = g.resoudre(["S1"], net=None, snapshot_path=tmp_path / "x.json",
                          stats_out=stats)
    assert src == "aucune" and pos == {} and "blocked" in stats["geo_error"]


def test_fetch_osm_substations(monkeypatch, tmp_path):
    # Réponse Overpass simulée : node (lat/lon direct) + way (center) ;
    # ref:FR:RTE = code_poste, geo_point_2d = [lat, lon].
    overpass = {"elements": [
        {"type": "node", "id": 1, "lat": 48.85, "lon": 2.35,
         "tags": {"power": "substation", "ref:FR:RTE": "PARIS",
                  "ref:FR:RTE_nom": "PARIS", "voltage": "400000"}},
        {"type": "way", "id": 2, "center": {"lat": 45.76, "lon": 4.83},
         "tags": {"power": "substation", "ref:FR:RTE": "LYON", "name": "Poste de Lyon"}},
        {"type": "node", "id": 3, "lat": 1.0, "lon": 1.0,
         "tags": {"power": "substation"}}]}   # sans ref:FR:RTE → ignoré
    monkeypatch.setattr(g, "_http_get", None)  # garde-fou : ne doit pas l'utiliser

    class _Resp:
        def __init__(self, b): self._b = b
        def read(self): return self._b
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(g.urllib.request, "urlopen",
                        lambda *a, **k: _Resp(json.dumps(overpass).encode("utf-8")))
    recs = g.fetch_osm_substations(cache_dir=tmp_path)
    assert {r["code_poste"] for r in recs} == {"PARIS", "LYON"}
    paris = next(r for r in recs if r["code_poste"] == "PARIS")
    assert paris["geo_point_2d"] == [48.85, 2.35] and paris["nom_poste"] == "PARIS"
    lyon = next(r for r in recs if r["code_poste"] == "LYON")
    assert lyon["geo_point_2d"] == [45.76, 4.83]
    assert (tmp_path / "osm_postes.json").exists()


def test_403_non_retriable():
    # 403 = refus (politique de sortie) : exclu des codes retentés.
    assert 403 not in g._RETRIABLE
    assert 429 in g._RETRIABLE


def test_positions_xiidm_absentes_par_defaut():
    pp = pytest.importorskip("pypowsybl")
    net = pp.network.create_four_substations_node_breaker_network()
    # réseau de test sans extension substationPosition → dict vide, pas d'erreur.
    assert g.positions_xiidm(net) == {}


def test_positions_xiidm_presentes():
    pp = pytest.importorskip("pypowsybl")
    net = pp.network.create_four_substations_node_breaker_network()
    net.create_extensions("substationPosition", id="S1", latitude=48.85,
                          longitude=2.35)
    pos = g.positions_xiidm(net)
    assert pos["S1"] == {"lat": 48.85, "lon": 2.35, "source": "xiidm"}
