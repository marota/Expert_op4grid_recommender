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
