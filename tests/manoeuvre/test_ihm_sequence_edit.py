"""
tests/manoeuvre/test_ihm_sequence_edit.py
-----------------------------------------
Tests de la **logique pure** d'édition de séquence de l'IHM
(``scripts/manoeuvre_ihm.py``) : rejeu des états, dérivation d'une manœuvre
manuelle, et invariants d'insertion (« conserver la suite ») / suppression.

Les parties Flask / SVG / pypowsybl ne sont pas testées ici (l'IHM nécessite un
réseau ``--grid`` et tourne en mode interactif) ; seules les fonctions pures
``_replay_states`` et ``_manual_manoeuvre`` sont exercées, sans instancier
``Session`` (donc sans réseau).
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

# L'IHM importe Flask au chargement du module → optionnel.
pytest.importorskip("flask")
pytest.importorskip("pypowsybl")

_IHM_PATH = (pathlib.Path(__file__).resolve().parents[2]
             / "scripts" / "manoeuvre_ihm.py")


def _load_ihm():
    spec = importlib.util.spec_from_file_location("manoeuvre_ihm_mod", _IHM_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ihm = _load_ihm()


def _open(states_dict):
    """Ensemble des organes ouverts (pour comparer des états)."""
    return {k for k, v in states_dict.items() if v}


# ---------------------------------------------------------------------------
# _replay_states
# ---------------------------------------------------------------------------

def test_replay_states_etats_successifs():
    initial = {"a": False, "b": False, "c": False}
    man = [
        {"switch_id": "a", "action": "OPEN"},
        {"switch_id": "b", "action": "OPEN"},
    ]
    states = ihm._replay_states(initial, man)
    assert len(states) == 3                     # départ + 2 manœuvres
    assert _open(states[0]) == set()
    assert _open(states[1]) == {"a"}
    assert _open(states[2]) == {"a", "b"}
    # immutabilité : l'état initial n'est pas modifié
    assert initial == {"a": False, "b": False, "c": False}


def test_replay_states_close_reouvre():
    initial = {"a": True}
    man = [{"switch_id": "a", "action": "CLOSE"},
           {"switch_id": "a", "action": "OPEN"}]
    states = ihm._replay_states(initial, man)
    assert [_open(s) for s in states] == [{"a"}, set(), {"a"}]


# ---------------------------------------------------------------------------
# _manual_manoeuvre
# ---------------------------------------------------------------------------

def test_manual_manoeuvre_bascule_depuis_etat_affiche():
    # organe fermé -> OPEN ; ouvert -> CLOSE
    assert ihm._manual_manoeuvre({"x": False}, "x")["action"] == "OPEN"
    assert ihm._manual_manoeuvre({"x": True}, "x")["action"] == "CLOSE"
    m = ihm._manual_manoeuvre({"x": False}, "x")
    assert m["switch_id"] == "x"
    assert "manuelle" in m["raison"]
    assert m["boucle"] is None


def test_manual_manoeuvre_organe_inconnu():
    assert ihm._manual_manoeuvre({"x": False}, "y") is None


# ---------------------------------------------------------------------------
# Invariant d'insertion : « insérer après l'étape, conserver la suite »
# ---------------------------------------------------------------------------

def test_insertion_conserve_la_suite():
    initial = {"a": False, "b": False, "c": False, "d": False}
    man = [
        {"switch_id": "a", "action": "OPEN"},   # état 1
        {"switch_id": "b", "action": "OPEN"},   # état 2
        {"switch_id": "c", "action": "OPEN"},   # état 3
    ]
    states = ihm._replay_states(initial, man)

    step = 1                                     # on visualise l'état 1 ({a})
    mm = ihm._manual_manoeuvre(states[step], "d")  # d fermé -> OPEN
    new_man = man[:step] + [mm] + man[step:]     # insertion en position step
    new_states = ihm._replay_states(initial, new_man)

    # une manœuvre de plus
    assert len(new_states) == len(states) + 1
    # les états jusqu'à l'étape affichée sont inchangés
    assert _open(new_states[0]) == _open(states[0])
    assert _open(new_states[1]) == _open(states[1])
    # le nouvel état (step+1) bascule bien 'd' relativement à l'état affiché
    assert _open(new_states[step + 1]) == {"a", "d"}
    # la suite est conservée : l'état final cumule tout (a,b,c,d)
    assert _open(new_states[-1]) == {"a", "b", "c", "d"}


def test_suppression_retire_l_effet():
    initial = {"a": False, "b": False, "c": False}
    man = [
        {"switch_id": "a", "action": "OPEN"},
        {"switch_id": "b", "action": "OPEN"},
        {"switch_id": "c", "action": "OPEN"},
    ]
    index = 2                                    # supprimer la 2e manœuvre (b)
    new_man = man[:index - 1] + man[index:]
    new_states = ihm._replay_states(initial, new_man)
    assert len(new_states) == len(man)           # 3 états (départ + 2)
    assert _open(new_states[-1]) == {"a", "c"}   # 'b' n'est plus manœuvré


# ---------------------------------------------------------------------------
# _delete_indices (suppression multiple / bloc)
# ---------------------------------------------------------------------------

def _man(*sids):
    return [{"switch_id": s, "action": "OPEN"} for s in sids]


def test_delete_indices_selection_eparse():
    man = _man("a", "b", "c", "d", "e")
    out = ihm._delete_indices(man, [2, 4])       # retire b et d
    assert [m["switch_id"] for m in out] == ["a", "c", "e"]


def test_delete_indices_bloc_contigu():
    man = _man("a", "b", "c", "d", "e")
    out = ihm._delete_indices(man, [2, 3, 4])    # bloc b-c-d
    assert [m["switch_id"] for m in out] == ["a", "e"]


def test_delete_indices_ignore_doublons_et_hors_bornes():
    man = _man("a", "b", "c")
    out = ihm._delete_indices(man, [2, 2, 0, 99, -1])  # seul 2 (b) est valide
    assert [m["switch_id"] for m in out] == ["a", "c"]


def test_delete_indices_ne_mute_pas_l_original():
    man = _man("a", "b", "c")
    ihm._delete_indices(man, [1, 2, 3])
    assert [m["switch_id"] for m in man] == ["a", "b", "c"]  # liste source intacte


def test_delete_indices_ordre_insensible():
    man = _man("a", "b", "c", "d")
    assert ihm._delete_indices(man, [4, 1, 3]) == ihm._delete_indices(man, [1, 3, 4])
