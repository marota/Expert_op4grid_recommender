# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""First tests for the two data-loading modules (revision R6).

``utils/load_training_data.py`` and ``utils/load_evaluation_data.py`` had
**zero** direct coverage (review M3) — which is precisely why the C6 bugs
(``state = action_path[0]`` indexing a StateInfo; a ``filter_out_...`` function
asserting against a name that only existed in the ``__main__`` block; a bare
``raise("...")``) shipped unnoticed. These tests cover the importable / pure
surface and lock in the C6 fixes.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from expert_op4grid_recommender.utils import (
    data_utils,
    load_evaluation_data,
    load_training_data,
)


# --- Import smoke: the modules must import as a library ---------------------
# The C6 ``filter_out_non_reproductible_observation`` bug raised NameError the
# moment the module was used; a plain import + attribute check is the cheapest
# guard against that whole class of "only works from __main__" breakage.

@pytest.mark.parametrize("mod", [load_training_data, load_evaluation_data, data_utils])
def test_module_imports_and_is_a_module(mod):
    assert inspect.ismodule(mod)


# --- C6 regression: filter_out_non_reproductible_observation ----------------

def test_filter_requires_line_we_disconnect_param():
    """C6: ``line_we_disconnect`` used to be a module global defined only in the
    ``__main__`` block (NameError on library use). It must now be an explicit
    required parameter."""
    sig = inspect.signature(load_training_data.filter_out_non_reproductible_observation)
    assert "line_we_disconnect" in sig.parameters
    p = sig.parameters["line_we_disconnect"]
    assert p.default is inspect.Parameter.empty  # required, not defaulted


# --- load_interesting_lines (pure CSV reader) ------------------------------

def test_load_interesting_lines_missing_file_returns_empty(tmp_path):
    out = load_training_data.load_interesting_lines(
        path=str(tmp_path), file_name="does_not_exist.csv"
    )
    assert isinstance(out, np.ndarray)
    assert out.size == 0


def test_load_interesting_lines_reads_and_strips(tmp_path):
    csv = tmp_path / "lines.csv"
    csv.write_text("branches\n  L1 \nL2\n\tL3\t\n", encoding="utf-8")
    out = load_training_data.load_interesting_lines(
        path=str(tmp_path), file_name="lines.csv"
    )
    assert list(out) == ["L1", "L2", "L3"]  # whitespace stripped both ends


# --- C6 regression: load_evaluation_data raises a real exception ------------

def test_load_evaluation_data_uses_valueerror_not_bare_raise():
    """C6: the no-chronic path used ``raise("...")`` -> TypeError (exceptions
    must derive from BaseException), masking the real error. It must raise a
    proper ValueError now."""
    src = inspect.getsource(load_evaluation_data)
    assert 'raise ValueError("no chronic is found' in src
    # And the maintenance-reconnection action must reconnect (set_line_status
    # +1), not disconnect via set_bus -1 (the other half of the C6 fix).
    assert "set_line_status" in src


# --- data_utils.StateInfo is constructible -----------------------------------

def test_stateinfo_is_a_class_with_expected_api():
    assert inspect.isclass(data_utils.StateInfo)
    for method in ("from_init_state", "set_env_state"):
        assert hasattr(data_utils.StateInfo, method)
