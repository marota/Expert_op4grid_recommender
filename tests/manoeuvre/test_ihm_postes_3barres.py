"""
tests/manoeuvre/test_ihm_postes_3barres.py
------------------------------------------
Couverture des évolutions IHM de la PR « postes à N jeux de barres » :

- les **7 postes 400 kV à 3 jeux de barres** identifiés sont épinglés
  (``POSTES_TEST``) ;
- ``Session.all_postes`` = **tous** les voltage levels NODE_BREAKER de la
  situation (pour la recherche / l'inspection de n'importe quel poste) ;
- ``GET /api/postes`` renvoie ``{postes, all}`` ;
- ``POST /api/load_grid`` charge **dynamiquement** une autre situation réseau
  (succès + erreur 400 propre, session inchangée en cas d'échec).

Réseau de référence : ``create_four_substations_node_breaker_network`` (rapide,
sans dépendance à un fichier grille). Nécessite ``flask`` + ``pypowsybl``.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

pytest.importorskip("flask")
pytest.importorskip("pypowsybl")

import pypowsybl as pp  # noqa: E402

_IHM_PATH = (pathlib.Path(__file__).resolve().parents[2]
             / "scripts" / "manoeuvre_ihm.py")

# Les 7 postes 400 kV à 3 JdB identifiés (réseau France 28/08/2024).
POSTES_3JDB = ["SSV.OP7", "TAVELP7", "TRI.PP7", "ARGOEP7", "CHESNP7",
               "COR.PP7", "CERGYP7"]


def _load_ihm():
    spec = importlib.util.spec_from_file_location("manoeuvre_ihm_3b_mod", _IHM_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ihm = _load_ihm()


def _net():
    return pp.network.create_four_substations_node_breaker_network()


# --------------------------------------------------------------------------
# 1. Les 7 postes 3 JdB sont épinglés
# --------------------------------------------------------------------------

@pytest.mark.parametrize("vl", POSTES_3JDB)
def test_poste_3jdb_epingle(vl):
    assert vl in ihm.POSTES_TEST, f"{vl} doit être dans POSTES_TEST"


# --------------------------------------------------------------------------
# 2. Session.all_postes : tous les VL NODE_BREAKER, épinglés en tête
# --------------------------------------------------------------------------

def test_session_all_postes_node_breaker():
    net = _net()
    s = ihm.Session(net)
    assert s.all_postes, "all_postes ne doit pas être vide"
    assert set(s.postes) <= set(s.all_postes)
    # Les postes épinglés présents apparaissent en tête.
    assert s.all_postes[:len(s.postes)] == s.postes
    # Tous les all_postes sont des VL avec sections de barres (NODE_BREAKER).
    bbs = net.get_busbar_sections(all_attributes=True)
    nb_vls = set(bbs["voltage_level_id"])
    assert set(s.all_postes) <= nb_vls
    # pas de doublon
    assert len(s.all_postes) == len(set(s.all_postes))


# --------------------------------------------------------------------------
# 3. /api/postes renvoie postes + all
# --------------------------------------------------------------------------

def test_api_postes_renvoie_all(monkeypatch):
    s = ihm.Session(_net())
    monkeypatch.setattr(ihm, "SESSION", s)
    r = ihm.app.test_client().get("/api/postes").get_json()
    assert "postes" in r and "all" in r
    assert set(r["postes"]) <= set(r["all"])
    assert r["all"] == s.all_postes


# --------------------------------------------------------------------------
# 4. /api/load_grid : bascule dynamique de situation réseau
# --------------------------------------------------------------------------

def test_load_grid_bascule_situation(monkeypatch, tmp_path):
    s0 = ihm.Session(_net())
    monkeypatch.setattr(ihm, "SESSION", s0)
    grid = tmp_path / "situation.xiidm"
    s0.net.save(str(grid), format="XIIDM")

    r = ihm.app.test_client().post(
        "/api/load_grid", json={"path": str(grid)}).get_json()

    assert r["ok"] is True
    assert r["all"] and set(r["postes"]) <= set(r["all"])
    # La session globale a été remplacée par la nouvelle situation.
    assert ihm.SESSION is not s0
    assert ihm.SESSION.all_postes == r["all"]


def test_load_grid_chemin_invalide(monkeypatch):
    s0 = ihm.Session(_net())
    monkeypatch.setattr(ihm, "SESSION", s0)

    resp = ihm.app.test_client().post(
        "/api/load_grid", json={"path": "/chemin/inexistant_xyz.xiidm"})

    assert resp.status_code == 400
    body = resp.get_json()
    assert body["ok"] is False and "error" in body
    # En cas d'échec, la session courante n'est PAS modifiée.
    assert ihm.SESSION is s0


# --------------------------------------------------------------------------
# 5. Catalogue par typologie (sections du sélecteur de poste)
# --------------------------------------------------------------------------

# Les 10 postes « caractéristiques particulières » (VL réels) doivent figurer
# au catalogue, comme les 7 postes 3 JdB.
POSTES_PARTICULIERS = [".OBER 7", ".VANY 7", ".ZAND 7", "MUHLBP7", ".LAUF 7",
                       "P.GASP6", "CPNIEP6", "ROMAIP6", "REICHP3", ".MUHL 6"]


def test_catalog_structure_et_couverture():
    """``Session.catalog()`` : sections {title, postes:[{vl, available}], n_available}
    couvrant **tous** les postes 3 JdB et les 10 postes particuliers."""
    cat = ihm.Session(_net()).catalog()
    assert isinstance(cat, list) and cat
    tous = set()
    for sec in cat:
        assert {"title", "postes", "n_available"} <= set(sec)
        for p in sec["postes"]:
            assert {"vl", "available"} <= set(p)
            tous.add(p["vl"])
        assert sec["n_available"] == sum(1 for p in sec["postes"] if p["available"])
    for vl in POSTES_3JDB + POSTES_PARTICULIERS:
        assert vl in tous, f"{vl} absent du catalogue par typologie"


def test_catalog_disponibilite_reflete_la_situation(monkeypatch):
    """``available`` est vrai ssi le VL est présent dans la situation chargée."""
    s = ihm.Session(_net())
    # Le réseau de test ne contient aucun poste du catalogue → tout indisponible.
    cat = s.catalog()
    assert all(not p["available"] for sec in cat for p in sec["postes"])
    assert all(sec["n_available"] == 0 for sec in cat)
    # On simule une situation contenant 2 postes du catalogue.
    s.vls = {".OBER 7", "P.GASP6", "AUTRE"}
    dispo = {p["vl"] for sec in s.catalog() for p in sec["postes"] if p["available"]}
    assert dispo == {".OBER 7", "P.GASP6"}


def test_api_postes_inclut_catalog(monkeypatch):
    s = ihm.Session(_net())
    monkeypatch.setattr(ihm, "SESSION", s)
    r = ihm.app.test_client().get("/api/postes").get_json()
    assert "catalog" in r and isinstance(r["catalog"], list)
    assert r["catalog"] == s.catalog()
