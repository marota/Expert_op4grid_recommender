"""
tests/manoeuvre/test_exploration.py
-----------------------------------
Tests du module ``manoeuvre.dataset.exploration`` (exploration de l'intérêt
d'une journée : changements d'OC par poste sur 3 situations).

Le cœur d'agrégation est **Python pur** : testé sans pypowsybl. L'extraction
(``extraire_etats_kinds`` / ``structure_reseau``) est testée sur le réseau de
référence pypowsybl si disponible.
"""
from __future__ import annotations

import pytest

from expert_op4grid_recommender.manoeuvre.dataset import exploration as ex


# --- cœur pur ---------------------------------------------------------------

KINDS = {"a_dj": "BREAKER", "a_sa": "DISCONNECTOR", "a_sa2": "DISCONNECTOR",
         "a_int": "LOAD_BREAK_SWITCH", "b_dj": "BREAKER"}


def _situations():
    s0 = {"A": {"a_dj": False, "a_sa": False, "a_sa2": False, "a_int": False},
          "B": {"b_dj": False}}
    s1 = {"A": {"a_dj": True, "a_sa": False, "a_sa2": True, "a_int": True},
          "B": {"b_dj": False}}
    s2 = {"A": {"a_dj": True, "a_sa": False, "a_sa2": False, "a_int": True},
          "B": {"b_dj": False}}
    return [s0, s1, s2]


def test_a_change():
    assert ex._a_change([False, True, False]) is True
    assert ex._a_change([True, True, True]) is False
    assert ex._a_change([None, True, None]) is False   # une seule valeur connue
    assert ex._a_change([None, True, False]) is True


def test_changements_par_vl_compte_par_type():
    ch = ex.changements_par_vl(_situations(), KINDS)
    # A : a_dj (F→T), a_sa2 (F→T→F), a_int (F→T) changent ; a_sa constant.
    assert ch["A"]["total"] == 3
    assert ch["A"]["BREAKER"] == 1
    assert ch["A"]["DISCONNECTOR"] == 1
    assert ch["A"]["LOAD_BREAK_SWITCH"] == 1
    assert set(ch["A"]["changed"]) == {"a_dj", "a_sa2", "a_int"}
    assert ch["A"]["n_oc"] == 4
    # B : aucun changement.
    assert ch["B"]["total"] == 0


def test_changements_vl_apparu_partiellement():
    # un organe absent d'une situation ne compte pas comme un changement.
    s = [{"V": {"x": False}}, {"V": {}}, {"V": {"x": False}}]
    ch = ex.changements_par_vl(s, {"x": "BREAKER"})
    assert ch["V"]["total"] == 0


def test_agreger_par_poste_et_classement():
    ch = ex.changements_par_vl(_situations(), KINDS)
    vl_meta = {"A": {"substation": "S1", "nominal_v": 400.0, "name": "A"},
               "B": {"substation": "S2", "nominal_v": 63.0, "name": "B"}}
    postes = ex.agreger_par_poste(ch, vl_meta, {"S1": "POSTE_A", "S2": "POSTE_B"})
    assert postes["S1"]["name"] == "POSTE_A"
    assert postes["S1"]["total"] == 3
    assert postes["S1"]["BREAKER"] == 1
    assert postes["S1"]["nominal_v_max"] == 400.0
    assert ex.classer_postes(postes, 10) == ["S1"]       # S2 inactif exclu
    assert ex.vl_le_plus_actif(postes["S1"]) == "A"


def test_agreger_multi_vl_par_poste():
    # deux VL d'un même poste : changements additionnés, tension max retenue.
    ch = {"P6": {"total": 2, "n_oc": 5, "changed": [], "BREAKER": 2,
                 "DISCONNECTOR": 0, "LOAD_BREAK_SWITCH": 0},
          "P7": {"total": 5, "n_oc": 9, "changed": [], "BREAKER": 1,
                 "DISCONNECTOR": 4, "LOAD_BREAK_SWITCH": 0}}
    vl_meta = {"P6": {"substation": "SUB", "nominal_v": 225.0, "name": "P6"},
               "P7": {"substation": "SUB", "nominal_v": 400.0, "name": "P7"}}
    postes = ex.agreger_par_poste(ch, vl_meta)
    p = postes["SUB"]
    assert p["total"] == 7 and p["nominal_v_max"] == 400.0
    assert [v["vl"] for v in p["vls"]] == ["P7", "P6"]    # tri par changements desc
    assert ex.vl_le_plus_actif(p) == "P7"


def test_classer_top_n_ordre():
    postes = {
        "X": {"total": 4, "nominal_v_max": 63.0, "name": "X"},
        "Y": {"total": 9, "nominal_v_max": 225.0, "name": "Y"},
        "Z": {"total": 9, "nominal_v_max": 400.0, "name": "Z"},
        "W": {"total": 0, "nominal_v_max": 90.0, "name": "W"},
    }
    # tri : total desc, puis tension desc → Z (400) avant Y (225) à total égal.
    assert ex.classer_postes(postes, 10) == ["Z", "Y", "X"]
    assert ex.classer_postes(postes, 2) == ["Z", "Y"]


# --- extraction (pypowsybl) -------------------------------------------------

def test_extraction_reseau_reference():
    pp = pytest.importorskip("pypowsybl")
    net = pp.network.create_four_substations_node_breaker_network()
    etats, kinds = ex.extraire_etats_kinds(net)
    assert etats and kinds
    # toutes les valeurs d'état sont des booléens ; kinds connus.
    a_vl = next(iter(etats))
    assert all(isinstance(v, bool) for v in etats[a_vl].values())
    assert set(kinds.values()) <= {"BREAKER", "DISCONNECTOR", "LOAD_BREAK_SWITCH"}
    vl_meta, sub_name = ex.structure_reseau(net)
    assert vl_meta and all("substation" in m and "nominal_v" in m
                           for m in vl_meta.values())
    # chaque VL extrait appartient à une substation connue.
    assert all(vl in vl_meta for vl in etats)
