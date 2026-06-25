"""
tests/manoeuvre/test_geographie.py
----------------------------------
Tests du résolveur de coordonnées ``manoeuvre.dataset.geographie`` :
normalisation des mnémoniques, appariement ODRE (formes de géo variées),
chargement d'instantané committé, et chaîne de résolution.

Aucun accès réseau : ODRE est simulé par des enregistrements en mémoire ;
l'extension ``substationPosition`` (absente du dataset RTE 7000 réel) est
couverte si pypowsybl est disponible.
"""
from __future__ import annotations

import json

import pytest

from expert_op4grid_recommender.manoeuvre.dataset import geographie as g


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
    assert pos[".CTLH"]["lat"] == 48.7 and pos[".CTLH"]["source"] == "odre"
    assert pos[".G.RO"]["lat"] == 45.1            # .G.RO → GRO (normalisé)
    assert pos[".NAVA"]["lon"] == -1.4
    assert "CARRIP" not in pos                    # aucun code ODRE correspondant
    assert stats["n_apparies"] == 3
    assert stats["n_substations"] == 4
    assert 0.0 < stats["taux"] <= 1.0


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
                          autoriser_odre=False)
    assert src == "snapshot" and "S1" in pos and "S2" not in pos
    # aucune source disponible → ('aucune', {})
    pos2, src2 = g.resoudre(["S1"], net=None, snapshot_path=tmp_path / "x.json",
                            autoriser_odre=False)
    assert src2 == "aucune" and pos2 == {}


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
