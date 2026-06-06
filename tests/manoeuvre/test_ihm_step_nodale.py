"""
tests/manoeuvre/test_ihm_step_nodale.py
---------------------------------------
La **topologie nodale suit l'étape** de la séquence (IHM) : ``step_view(i)`` /
``GET /api/step`` renvoient la **partition nodale de l'état détaillé de l'étape**,
pour que le volet « cible » de l'IHM se mette à jour au fil de l'animation
(départ → … → cible).

Réseau de référence : ``create_four_substations_node_breaker_network`` (VL S1VL2).
Nécessite ``flask`` + ``pypowsybl``.
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
    spec = importlib.util.spec_from_file_location("manoeuvre_ihm_step_mod", _IHM_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ihm = _load_ihm()


def _session_avec_cible_scindee():
    """Session S1VL2 avec une cible **scindée en 2 nœuds** et sa séquence calculée."""
    s = ihm.Session(pp.network.create_four_substations_node_breaker_network())
    s.load(VL)
    flat = sorted(eq for g in s.groups_of(s.initial) for eq in g)
    assert len(flat) >= 2, "le poste de test doit avoir ≥ 2 départs"
    half = len(flat) // 2
    s.nodale_to_detaillee([flat[:half], flat[half:]])
    s.sequence("smooth")
    return s


def test_step_view_vide_renvoie_six_valeurs():
    """Contrat : ``step_view`` renvoie 6 valeurs même sans séquence (nodale=None)
    → pas d'``unpacking`` cassé côté endpoint."""
    s = ihm.Session(pp.network.create_four_substations_node_breaker_network())
    s.load(VL)
    out = s.step_view(0)
    assert len(out) == 6
    assert out[-1] is None


def test_step_view_renvoie_la_nodale_de_letape():
    s = _session_avec_cible_scindee()
    # Étape 0 = état de départ → nodale == nodale de l'état de départ.
    *_rest, nod0 = s.step_view(0)
    assert nod0 is not None and "groups" in nod0
    assert nod0 == s.nodale_state(s.seq_states[0])
    # Dernière étape → nodale == nodale de l'état final de la séquence.
    last = len(s.seq_states) - 1
    *_rest2, nodL = s.step_view(last)
    assert nodL == s.nodale_state(s.seq_states[last])
    # La partition **suit** l'étape : si la cible scinde réellement le nœud, le
    # dernier état a strictement plus de nœuds connectés que le départ.
    if s._topo(s.current).nb_noeuds > s._topo(s.initial).nb_noeuds:
        assert len(nodL["groups"]) > len(nod0["groups"])


def test_api_step_inclut_la_nodale(monkeypatch):
    s = _session_avec_cible_scindee()
    monkeypatch.setattr(ihm, "SESSION", s)
    client = ihm.app.test_client()
    r0 = client.get("/api/step?i=0").get_json()
    last = len(s.seq_states) - 1
    rL = client.get(f"/api/step?i={last}").get_json()
    for r in (r0, rL):
        assert "nodale" in r and isinstance(r["nodale"], dict)
        assert "groups" in r["nodale"]
    # cohérence avec le nb de nœuds renvoyé pour la vue détaillée.
    assert rL["nodale"] == s.nodale_state(s.seq_states[last])
