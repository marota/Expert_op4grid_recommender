"""
tests/manoeuvre/test_ihm_cache_and_api.py
-----------------------------------------
Filets de sécurité pour la revue qualité de l'IHM (``scripts/manoeuvre_ihm.py``) :

- **#4** — le vérificateur de règle « sectionneur sous charge » est désormais
  **public** (``manoeuvre.sectionneurs_sous_charge_par_manoeuvre``), l'alias privé
  historique étant conservé pour compatibilité ;
- **#5** — ``Session.applied(state)`` restaure l'état d'affichage courant en
  sortie de bloc, **y compris sur exception** ;
- **#6** — graphe / topologie / load flow d'un état sont **mémoïsés par état** et
  invalidés au chargement d'un poste, **sans changer les sorties**.

Nécessitent ``flask`` et ``pypowsybl`` (sinon test ignoré).
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
VL = "S1VL2"


def _load_ihm():
    spec = importlib.util.spec_from_file_location("manoeuvre_ihm_cache_mod", _IHM_PATH)
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


# --------------------------------------------------------------------------
# #4 — vérificateur de règle public
# --------------------------------------------------------------------------

def test_public_checker_exported_from_package():
    from expert_op4grid_recommender import manoeuvre
    assert hasattr(manoeuvre, "sectionneurs_sous_charge_par_manoeuvre")
    assert "sectionneurs_sous_charge_par_manoeuvre" in manoeuvre.__all__


def test_public_checker_is_private_alias():
    from expert_op4grid_recommender.manoeuvre.algo import (
        sectionneurs_sous_charge_par_manoeuvre as pub,
        _sectionneurs_sous_charge_par_manoeuvre as priv,
    )
    assert pub is priv


def test_ihm_uses_public_checker():
    # L'IHM ne franchit plus la frontière privée : elle référence le symbole public.
    assert hasattr(ihm, "sectionneurs_sous_charge_par_manoeuvre")
    assert not hasattr(ihm, "_sectionneurs_sous_charge_par_manoeuvre")


# --------------------------------------------------------------------------
# #5 — gestionnaire de contexte applied(state)
# --------------------------------------------------------------------------

def test_applied_restores_current_on_success(session):
    before = dict(session.current)
    with session.applied(session.initial):
        pass
    # Réseau restauré sur l'affichage courant.
    g = ihm.build_vl_graph(session.net, VL)
    restored = ihm.TopologieNodale.from_graph(g, VL)
    cur = session._topo(session.current)
    assert restored.meme_topologie(cur)
    assert session.current == before


def test_applied_restores_current_on_exception(session):
    before = dict(session.current)
    with pytest.raises(RuntimeError):
        with session.applied(session.initial):
            raise RuntimeError("boom")
    assert session.current == before
    # L'état réseau effectif correspond bien à ``current`` (et non à ``initial``).
    g = ihm.build_vl_graph(session.net, VL)
    assert ihm.TopologieNodale.from_graph(g, VL).meme_topologie(
        session._topo(session.current))


# --------------------------------------------------------------------------
# #6 — mémoïsation par état (graphe / topo / flux)
# --------------------------------------------------------------------------

def test_caches_start_empty_then_fill(session):
    assert session._graph_cache == {}
    assert session._topo_cache == {}
    assert session._flow_cache == {}
    session.nodale_payload(session.initial)
    assert session._graph_cache
    assert session._topo_cache
    assert session._flow_cache


def test_repeated_payload_is_cache_hit_and_identical(session):
    first = session.nodale_payload(session.initial)
    n_graph = len(session._graph_cache)
    n_topo = len(session._topo_cache)
    second = session.nodale_payload(session.initial)
    # Même état → pas de nouvelle entrée de cache, sortie identique.
    assert len(session._graph_cache) == n_graph
    assert len(session._topo_cache) == n_topo
    assert first == second


def test_load_clears_caches(session):
    session.nodale_payload(session.initial)
    assert session._graph_cache and session._topo_cache and session._flow_cache
    session.load(VL)
    assert session._graph_cache == {}
    assert session._topo_cache == {}
    assert session._flow_cache == {}


def test_cache_keyed_by_state_distinguishes_topologies(session):
    # Tout ouvrir produit un état distinct → clé de cache distincte, sortie distincte.
    base = session.nodale_state(session.initial)
    n_keys = len(session._graph_cache)
    for sid in list(session.switches_df(VL).index):
        session.current[sid] = True
    opened = session.nodale_state(session.current)
    assert len(session._graph_cache) > n_keys          # nouvelle entrée
    assert opened["isolated"]                            # tout déconnecté
    assert base["isolated"] == []


def test_memoized_topo_matches_fresh_build(session):
    # La topo mémoïsée doit être équivalente à une reconstruction directe.
    session.apply(session.initial)
    fresh = ihm.TopologieNodale.from_graph(
        ihm.build_vl_graph(session.net, VL), VL)
    cached = session._topo(session.initial)
    assert cached.meme_topologie(fresh)
