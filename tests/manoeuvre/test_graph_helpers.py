"""
tests/manoeuvre/test_graph_helpers.py
-------------------------------------
Tests unitaires de ``graph._safe_get`` — l'utilitaire qui appelle les getters
pypowsybl avec ``all_attributes=True`` (cf. convention critique du module) et
gère le repli pour les getters anciens / les erreurs. Couvre les trois branches
de robustesse, indépendamment de pypowsybl (getters factices).
"""

from __future__ import annotations

from expert_op4grid_recommender.manoeuvre.graph import _safe_get


def test_safe_get_passes_all_attributes_and_returns_value():
    seen = {}

    def getter(all_attributes=False):
        seen["all_attributes"] = all_attributes
        return "DF"

    assert _safe_get(getter) == "DF"
    assert seen["all_attributes"] is True   # le flag est bien transmis


def test_safe_get_retries_without_all_attributes_on_typeerror():
    """Getter ancien : ``all_attributes`` lève TypeError → repli sans le flag."""
    calls = []

    def legacy_getter(all_attributes=False):
        calls.append(all_attributes)
        if all_attributes:
            raise TypeError("unexpected keyword argument 'all_attributes'")
        return "LEGACY_DF"

    assert _safe_get(legacy_getter) == "LEGACY_DF"
    assert calls == [True, False]           # tenté avec, puis sans


def test_safe_get_returns_none_when_retry_also_fails():
    def broken(all_attributes=False):
        if all_attributes:
            raise TypeError("no kw")
        raise RuntimeError("toujours cassé")

    assert _safe_get(broken) is None


def test_safe_get_returns_none_on_non_typeerror():
    def raising(all_attributes=False):
        raise ValueError("getter en échec")

    assert _safe_get(raising) is None
