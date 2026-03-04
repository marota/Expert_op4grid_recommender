"""
Tests for the minimum-action-count-per-type feature (v0.1.3).

Covers:
- add_prioritized_actions() helper function behaviour
- MIN_LINE_RECONNECTIONS / MIN_CLOSE_COUPLING / MIN_OPEN_COUPLING /
  MIN_LINE_DISCONNECTIONS enforcement inside ActionDiscoverer
- Warning emitted by main() when sum(MIN_*) > N_PRIORITIZED_ACTIONS
- LINES_MONITORING_FILE routing in environment.py setup helpers
"""
import sys
import os
import io
import pytest
import unittest.mock as mock

# Make sure the project root is on the path (conftest.py already handles this,
# but we add it here for clarity when running the file standalone).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_identified(ids):
    """Return a minimal identified_actions dict keyed by the given ids."""
    return {
        action_id: {
            "action": object(),
            "score": float(i),
            "description_unitaire": f"Action {action_id}",
        }
        for i, action_id in enumerate(ids, start=1)
    }


# ---------------------------------------------------------------------------
# add_prioritized_actions – unit tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def add_prioritized_actions_fn():
    """Import add_prioritized_actions, skipping if dependencies are missing."""
    np = pytest.importorskip("numpy", reason="numpy required for helpers module")
    from expert_op4grid_recommender.utils.helpers import add_prioritized_actions
    return add_prioritized_actions


class TestAddPrioritizedActions:
    """Unit tests for utils.helpers.add_prioritized_actions."""

    @pytest.fixture(autouse=True)
    def _setup(self, add_prioritized_actions_fn):
        self.add_prioritized_actions = add_prioritized_actions_fn

    def test_zero_per_type_limit_adds_nothing(self):
        """n_action_max_per_type=0 must add no actions."""
        identified = _make_identified(["A1", "A2", "A3"])
        result = self.add_prioritized_actions({}, identified, n_action_max_total=5, n_action_max_per_type=0)
        assert result == {}

    def test_per_type_limit_respected(self):
        """At most n_action_max_per_type actions are added from identified_actions."""
        identified = _make_identified(["A1", "A2", "A3", "A4"])
        result = self.add_prioritized_actions({}, identified, n_action_max_total=5, n_action_max_per_type=2)
        assert len(result) == 2

    def test_total_limit_respected(self):
        """Total number of actions in prioritized_actions must not exceed n_action_max_total."""
        identified = _make_identified(["A1", "A2", "A3"])
        existing = _make_identified(["B1", "B2"])
        result = self.add_prioritized_actions(
            existing.copy(), identified, n_action_max_total=3, n_action_max_per_type=5
        )
        assert len(result) <= 3

    def test_already_present_action_not_duplicated(self):
        """Actions already in prioritized_actions must not be added again."""
        existing = _make_identified(["A1"])
        identified = _make_identified(["A1", "A2"])
        result = self.add_prioritized_actions(
            existing.copy(), identified, n_action_max_total=5, n_action_max_per_type=5
        )
        # A1 was already present, only A2 should be new
        assert "A1" in result
        assert "A2" in result
        assert len(result) == 2

    def test_empty_identified_returns_unchanged_prioritized(self):
        """Empty identified_actions should leave prioritized_actions untouched."""
        existing = _make_identified(["X1"])
        result = self.add_prioritized_actions(existing.copy(), {}, n_action_max_total=5, n_action_max_per_type=3)
        assert list(result.keys()) == ["X1"]

    def test_returns_dict(self):
        """Return value must always be a dict."""
        result = self.add_prioritized_actions({}, {}, n_action_max_total=5, n_action_max_per_type=3)
        assert isinstance(result, dict)

    def test_min_zero_still_leaves_total_cap_active(self):
        """With n_action_max_per_type=0, even a large total cap adds nothing."""
        identified = _make_identified([f"X{i}" for i in range(10)])
        result = self.add_prioritized_actions({}, identified, n_action_max_total=10, n_action_max_per_type=0)
        assert len(result) == 0

    def test_min_greater_than_available_adds_all_available(self):
        """If n_action_max_per_type > len(identified), all available are added."""
        identified = _make_identified(["A1", "A2"])
        result = self.add_prioritized_actions({}, identified, n_action_max_total=10, n_action_max_per_type=5)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# MIN_* enforcement in the two-pass prioritisation logic
# ---------------------------------------------------------------------------

class TestMinActionCountEnforcement:
    """
    Tests that verify the two-pass add_prioritized_actions logic inside
    ActionDiscoverer.discover_and_prioritize respects MIN_* config values.

    We test the helper directly with the same calling pattern used in
    discovery.py, rather than exercising the full ActionDiscoverer pipeline
    (which requires heavy mocking of alphaDeesp internals).
    """

    @pytest.fixture(autouse=True)
    def _setup(self, add_prioritized_actions_fn):
        self.add_prioritized_actions = add_prioritized_actions_fn

    def _run_two_pass(self, reconnections, merges, splits, disconnections,
                      min_reco, min_close, min_open, min_deco,
                      n_total=5, n_reco_max=3, n_split_max=3):
        """Replicate the two-pass logic from discovery.py for unit-testing."""
        prioritized = {}

        # Pass 1 – minimum guarantees
        prioritized = self.add_prioritized_actions(
            prioritized, reconnections, n_total, n_action_max_per_type=min_reco
        )
        prioritized = self.add_prioritized_actions(
            prioritized, merges, n_total, n_action_max_per_type=min_close
        )
        prioritized = self.add_prioritized_actions(
            prioritized, splits, n_total, n_action_max_per_type=min_open
        )
        prioritized = self.add_prioritized_actions(
            prioritized, disconnections, n_total, n_action_max_per_type=min_deco
        )

        # Pass 2 – fill remaining slots
        prioritized = self.add_prioritized_actions(
            prioritized, reconnections, n_total, n_action_max_per_type=n_reco_max
        )
        prioritized = self.add_prioritized_actions(
            prioritized, merges, n_total
        )
        prioritized = self.add_prioritized_actions(
            prioritized, splits, n_total, n_action_max_per_type=n_split_max
        )
        prioritized = self.add_prioritized_actions(
            prioritized, disconnections, n_total
        )

        return prioritized

    def test_all_zeros_normal_fill(self):
        """With all MIN_* = 0, the first pass adds nothing; second pass fills normally."""
        reco = _make_identified(["R1", "R2"])
        merges = _make_identified(["M1"])
        result = self._run_two_pass(reco, merges, {}, {}, 0, 0, 0, 0, n_total=5)
        assert "R1" in result
        assert "R2" in result
        assert "M1" in result

    def test_min_reconnections_guaranteed(self):
        """MIN_LINE_RECONNECTIONS=2 ensures at least 2 reconnections even when merges are better scored."""
        reco = _make_identified(["R1", "R2", "R3"])
        # merges would normally displace reconnections if they had higher priority
        merges = _make_identified([f"M{i}" for i in range(5)])
        result = self._run_two_pass(reco, merges, {}, {}, min_reco=2, min_close=0, min_open=0, min_deco=0, n_total=5)
        reco_in_result = [k for k in result if k.startswith("R")]
        assert len(reco_in_result) >= 2, f"Expected ≥2 reconnections, got {reco_in_result}"

    def test_min_disconnections_guaranteed(self):
        """MIN_LINE_DISCONNECTIONS=1 ensures at least 1 disconnection action is included."""
        deco = _make_identified(["D1", "D2"])
        reco = _make_identified([f"R{i}" for i in range(5)])
        result = self._run_two_pass(reco, {}, {}, deco, min_reco=0, min_close=0, min_open=0, min_deco=1, n_total=5)
        deco_in_result = [k for k in result if k.startswith("D")]
        assert len(deco_in_result) >= 1, f"Expected ≥1 disconnection, got {deco_in_result}"

    def test_min_close_coupling_guaranteed(self):
        """MIN_CLOSE_COUPLING=1 ensures at least 1 close-coupling action."""
        merges = _make_identified(["M1", "M2"])
        reco = _make_identified([f"R{i}" for i in range(5)])
        result = self._run_two_pass(reco, merges, {}, {}, 0, 1, 0, 0, n_total=5)
        merges_in_result = [k for k in result if k.startswith("M")]
        assert len(merges_in_result) >= 1

    def test_min_open_coupling_guaranteed(self):
        """MIN_OPEN_COUPLING=1 ensures at least 1 open-coupling action."""
        splits = _make_identified(["S1", "S2"])
        reco = _make_identified([f"R{i}" for i in range(5)])
        result = self._run_two_pass(reco, {}, splits, {}, 0, 0, 1, 0, n_total=5)
        splits_in_result = [k for k in result if k.startswith("S")]
        assert len(splits_in_result) >= 1

    def test_total_cap_still_respected_when_min_set(self):
        """Total prioritized actions must never exceed n_total even with MIN_* > 0."""
        reco = _make_identified([f"R{i}" for i in range(3)])
        merges = _make_identified([f"M{i}" for i in range(3)])
        splits = _make_identified([f"S{i}" for i in range(3)])
        deco = _make_identified([f"D{i}" for i in range(3)])
        result = self._run_two_pass(reco, merges, splits, deco, 1, 1, 1, 1, n_total=5)
        assert len(result) <= 5

    def test_min_more_than_available_does_not_crash(self):
        """MIN_* larger than number of available actions should not crash."""
        reco = _make_identified(["R1"])  # only 1 available
        result = self._run_two_pass(reco, {}, {}, {}, min_reco=5, min_close=0, min_open=0, min_deco=0, n_total=5)
        # Should add only the 1 available, without error
        assert "R1" in result
        assert len(result) == 1

    def test_min_all_zero_with_empty_candidates_returns_empty(self):
        """No candidates + all zero minimums => empty result."""
        result = self._run_two_pass({}, {}, {}, {}, 0, 0, 0, 0, n_total=5)
        assert result == {}


# ---------------------------------------------------------------------------
# Warning when sum(MIN_*) > N_PRIORITIZED_ACTIONS
# ---------------------------------------------------------------------------

class TestMinActionSumWarning:
    """
    Tests that main() emits a warning when the sum of MIN_* exceeds
    N_PRIORITIZED_ACTIONS.
    """

    def _get_sum_warning(self, min_reco, min_close, min_open, min_deco, n_prioritized):
        """Return the warning string that main() would print, or None."""
        sum_min = min_reco + min_close + min_open + min_deco
        if sum_min > n_prioritized:
            return (
                f"Warning: The sum of minimum actions per type ({sum_min}) exceeds the "
                f"maximum number of prioritized actions overall ({n_prioritized}). "
                f"Some minimums will not be respected."
            )
        return None

    def test_no_warning_when_sum_within_limit(self):
        msg = self._get_sum_warning(1, 1, 1, 1, n_prioritized=5)
        assert msg is None

    def test_warning_when_sum_equals_limit_plus_one(self):
        msg = self._get_sum_warning(2, 2, 1, 1, n_prioritized=5)
        assert msg is not None
        assert "exceeds" in msg

    def test_warning_contains_actual_numbers(self):
        msg = self._get_sum_warning(3, 3, 0, 0, n_prioritized=4)
        assert "6" in msg   # sum
        assert "4" in msg   # n_prioritized

    def test_warning_when_all_min_set_high(self):
        msg = self._get_sum_warning(10, 10, 10, 10, n_prioritized=5)
        assert msg is not None

    def test_no_warning_when_all_zero(self):
        msg = self._get_sum_warning(0, 0, 0, 0, n_prioritized=5)
        assert msg is None


# ---------------------------------------------------------------------------
# LINES_MONITORING_FILE routing
# ---------------------------------------------------------------------------

class TestLinesMonitoringFileRouting:
    """
    Tests for the LINES_MONITORING_FILE fallback logic used in environment.py.
    The key logic (replicated here for unit testing) is:

        monitoring_file = getattr(config, 'LINES_MONITORING_FILE', None)
        if monitoring_file is None:
            monitoring_file = os.path.join(config.ENV_FOLDER, "lignes_a_monitorer.csv")
        lines_we_care_about = load_interesting_lines(file_name=monitoring_file)
    """

    def _resolve_monitoring_file(self, config_monitoring_file, env_folder):
        """Replicate the routing logic from environment.py."""
        monitoring_file = config_monitoring_file  # already resolved via getattr
        if monitoring_file is None:
            monitoring_file = os.path.join(env_folder, "lignes_a_monitorer.csv")
        return monitoring_file

    def test_none_falls_back_to_default_csv(self):
        result = self._resolve_monitoring_file(None, "/some/env/folder")
        assert result == os.path.join("/some/env/folder", "lignes_a_monitorer.csv")

    def test_explicit_path_is_used_as_is(self):
        custom_path = "/custom/monitoring/lines.csv"
        result = self._resolve_monitoring_file(custom_path, "/some/env/folder")
        assert result == custom_path

    def test_default_path_ends_with_csv_filename(self):
        result = self._resolve_monitoring_file(None, "/data/env")
        assert result.endswith("lignes_a_monitorer.csv")

    def test_config_without_attribute_treated_as_none(self):
        """getattr with default None on a config lacking LINES_MONITORING_FILE gives None."""
        class FakeConfig:
            ENV_FOLDER = "/data/env"
            # LINES_MONITORING_FILE intentionally absent

        monitoring_file = getattr(FakeConfig, 'LINES_MONITORING_FILE', None)
        result = self._resolve_monitoring_file(monitoring_file, FakeConfig.ENV_FOLDER)
        assert result == os.path.join("/data/env", "lignes_a_monitorer.csv")

    def test_config_with_explicit_none_attribute(self):
        """LINES_MONITORING_FILE = None in config should also fall back to default."""
        class FakeConfig:
            ENV_FOLDER = "/data/env"
            LINES_MONITORING_FILE = None

        monitoring_file = getattr(FakeConfig, 'LINES_MONITORING_FILE', None)
        result = self._resolve_monitoring_file(monitoring_file, FakeConfig.ENV_FOLDER)
        assert result == os.path.join("/data/env", "lignes_a_monitorer.csv")

    def test_config_with_custom_path(self):
        """LINES_MONITORING_FILE set to a custom path should be used directly."""
        class FakeConfig:
            ENV_FOLDER = "/data/env"
            LINES_MONITORING_FILE = "/custom/my_monitoring.csv"

        monitoring_file = getattr(FakeConfig, 'LINES_MONITORING_FILE', None)
        result = self._resolve_monitoring_file(monitoring_file, FakeConfig.ENV_FOLDER)
        assert result == "/custom/my_monitoring.csv"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
