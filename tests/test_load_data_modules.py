# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""First test references for ``utils/load_training_data`` and
``utils/load_evaluation_data``.

The 2026-07 review (M3) flagged that these two modules had **zero** test
references — which is why the C6 correctness bugs (indexing a ``StateInfo``, a
``__main__``-only global ``NameError``, ``raise("string")``, a "reconnect"
action that disconnected) shipped unnoticed. These modules pull in the full
grid2op / pypowsybl stack at import, so the checks below are guarded by
``importorskip`` and skip cleanly where that stack is absent (e.g. the
grid2op-less CI leg) while acting as regression guards everywhere it is present.

The assertions are deliberately structural (signatures / source contracts) so
they do not need a live grid: each one pins exactly the contract a C6 fix
restored, so reverting a fix fails a test.
"""
import inspect

import pytest

# Both modules import the pypowsybl/grid2op backend transitively at module load.
pytest.importorskip("pypowsybl")

from expert_op4grid_recommender.utils import load_training_data as ltd  # noqa: E402
from expert_op4grid_recommender.utils import load_evaluation_data as led  # noqa: E402
from expert_op4grid_recommender.utils.data_utils import StateInfo  # noqa: E402


class TestLoadTrainingDataC6Contracts:
    def test_filter_out_non_reproductible_observation_takes_line_we_disconnect(self):
        """C6: ``line_we_disconnect`` must be a real parameter.

        It used to be read from a module-global defined only under
        ``if __name__ == '__main__'``, so importing and calling the function as
        a library raised ``NameError``.
        """
        params = inspect.signature(ltd.filter_out_non_reproductible_observation).parameters
        assert "line_we_disconnect" in params, list(params)

    def test_set_state_accepts_stateinfo_without_indexing(self):
        """C6: ``set_state`` must accept a ``StateInfo`` directly.

        The bug was ``state = action_path[0]`` (indexing a ``StateInfo`` that
        has no ``__getitem__`` → ``TypeError``); the fix branches on
        ``isinstance(action_path, StateInfo)``.
        """
        params = inspect.signature(ltd.set_state).parameters
        assert "action_path" in params, list(params)
        src = inspect.getsource(ltd.set_state)
        assert "isinstance(action_path, StateInfo)" in src, (
            "set_state must special-case StateInfo instead of indexing them"
        )
        # StateInfo must genuinely be non-subscriptable, which is what made the
        # old ``action_path[0]`` a hard TypeError.
        assert not hasattr(StateInfo, "__getitem__")


class TestLoadEvaluationDataC6Contracts:
    def test_missing_chronic_raises_real_exception(self):
        """C6: a not-found chronic must ``raise ValueError(...)`` (a real
        exception), not the old ``raise("string")`` which is a ``TypeError``
        that masked the real error.
        """
        src = inspect.getsource(led.get_first_obs_on_chronic)
        assert 'raise("' not in src and "raise('" not in src, (
            "must not `raise(<str>)` — exceptions must derive from BaseException"
        )
        assert "raise ValueError" in src

    def test_reconnect_action_reconnects_not_disconnects(self):
        """C6: the "reconnect maintenance lines" action must reconnect lines
        (``set_line_status`` status +1), not the old ``set_bus`` -1 which
        disconnected them.
        """
        src = inspect.getsource(led.run_remedial_action)
        assert "set_line_status" in src, "reconnect must go through set_line_status"
        assert "(line_reco, 1)" in src, (
            "maintenance lines must be reconnected with status 1, not disconnected"
        )
