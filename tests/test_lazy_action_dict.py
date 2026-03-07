# tests/test_lazy_action_dict.py
"""
Tests for the LazyActionDict class and enrich_actions_lazy function.

Verifies:
- Lazy computation of 'content' from 'switches' on first access
- Backward compatibility when 'content' is already present
- Correct behavior for actions without switches
- Single computation (idempotency)
- Integration with NetworkTopologyCache
"""

import pytest
from unittest.mock import MagicMock, patch


class TestLazyActionDict:
    """Tests for the LazyActionDict dict subclass."""

    def test_content_in_contains_returns_true(self):
        """'content' in lazy_dict should return True even when not yet computed."""
        from expert_op4grid_recommender.data_loader import LazyActionDict

        lazy = LazyActionDict({"switches": {"sw1": False}}, topology_cache=MagicMock())
        assert "content" in lazy

    def test_content_not_computed_until_accessed(self):
        """Content should not be computed until explicitly accessed."""
        from expert_op4grid_recommender.data_loader import LazyActionDict

        cache = MagicMock()
        lazy = LazyActionDict({"switches": {"sw1": False}}, topology_cache=cache)

        assert not lazy._content_computed
        cache.compute_bus_assignments.assert_not_called()

    def test_getitem_triggers_computation(self):
        """Accessing ['content'] should trigger lazy computation."""
        from expert_op4grid_recommender.data_loader import LazyActionDict

        cache = MagicMock()
        cache._switch_to_vl = {"sw1": "VL1"}
        cache.compute_bus_assignments.return_value = {"node_a": 1}
        cache.get_element_bus_assignments.return_value = {"lines_or_id": {"L1": 1}}

        lazy = LazyActionDict(
            {"switches": {"sw1": False}, "VoltageLevelId": "VL1"},
            topology_cache=cache
        )

        content = lazy["content"]

        assert lazy._content_computed
        cache.compute_bus_assignments.assert_called_once()
        cache.get_element_bus_assignments.assert_called_once()
        assert "set_bus" in content
        assert content["set_bus"] == {"lines_or_id": {"L1": 1}}
        assert content["switches"] == {"sw1": False}

    def test_get_triggers_computation(self):
        """Accessing .get('content') should trigger lazy computation."""
        from expert_op4grid_recommender.data_loader import LazyActionDict

        cache = MagicMock()
        cache._switch_to_vl = {"sw1": "VL1"}
        cache.compute_bus_assignments.return_value = {}
        cache.get_element_bus_assignments.return_value = {}

        lazy = LazyActionDict({"switches": {"sw1": False}}, topology_cache=cache)

        content = lazy.get("content", {})

        assert lazy._content_computed
        assert isinstance(content, dict)

    def test_computation_happens_only_once(self):
        """Content computation should happen exactly once, even with multiple accesses."""
        from expert_op4grid_recommender.data_loader import LazyActionDict

        cache = MagicMock()
        cache._switch_to_vl = {"sw1": "VL1"}
        cache.compute_bus_assignments.return_value = {}
        cache.get_element_bus_assignments.return_value = {}

        lazy = LazyActionDict({"switches": {"sw1": False}}, topology_cache=cache)

        lazy["content"]
        lazy["content"]
        lazy.get("content")

        assert cache.compute_bus_assignments.call_count == 1

    def test_preexisting_content_not_recomputed(self):
        """If content is already in the data dict, it should not be recomputed."""
        from expert_op4grid_recommender.data_loader import LazyActionDict

        existing_content = {"set_bus": {"lines_or_id": {"L1": 2}}}
        cache = MagicMock()

        lazy = LazyActionDict(
            {"content": existing_content, "switches": {"sw1": False}},
            topology_cache=cache
        )

        assert lazy._content_computed  # Already marked as computed
        result = lazy["content"]
        assert result == existing_content
        cache.compute_bus_assignments.assert_not_called()

    def test_disco_action_from_action_id(self):
        """disco_* actions should derive content from action ID."""
        from expert_op4grid_recommender.data_loader import LazyActionDict

        lazy = LazyActionDict(
            {"description_unitaire": "Ouverture de la ligne 'AISERL31MAGNY'"},
            topology_cache=MagicMock(),
            action_id="disco_AISERL31MAGNY"
        )

        content = lazy["content"]

        assert content["set_bus"]["lines_or_id"] == {"AISERL31MAGNY": -1}
        assert content["set_bus"]["lines_ex_id"] == {"AISERL31MAGNY": -1}
        assert content["set_bus"]["loads_id"] == {}
        assert content["set_bus"]["generators_id"] == {}

    def test_disco_action_from_description(self):
        """disco actions should fall back to description if action_id has no disco_ prefix."""
        from expert_op4grid_recommender.data_loader import LazyActionDict

        lazy = LazyActionDict(
            {"description_unitaire": "Ouverture de la ligne 'CPVANL61ZMAGN'"},
            topology_cache=MagicMock(),
            action_id="some_other_id"
        )

        content = lazy["content"]

        assert content["set_bus"]["lines_or_id"] == {"CPVANL61ZMAGN": -1}
        assert content["set_bus"]["lines_ex_id"] == {"CPVANL61ZMAGN": -1}

    def test_no_switches_no_disco_returns_empty_set_bus(self):
        """Actions without switches and not disco_* should get empty set_bus."""
        from expert_op4grid_recommender.data_loader import LazyActionDict

        cache = MagicMock()
        lazy = LazyActionDict(
            {"description": "some non-disco action"},
            topology_cache=cache,
            action_id="unknown_action"
        )

        content = lazy["content"]

        assert content == {"set_bus": {}}
        cache.compute_bus_assignments.assert_not_called()

    def test_no_cache_returns_empty_set_bus(self):
        """Switch actions without a topology cache should get empty set_bus."""
        from expert_op4grid_recommender.data_loader import LazyActionDict

        lazy = LazyActionDict({"switches": {"sw1": False}}, topology_cache=None)

        content = lazy["content"]

        assert content == {"set_bus": {}}

    def test_other_keys_work_normally(self):
        """Non-content keys should work as normal dict operations."""
        from expert_op4grid_recommender.data_loader import LazyActionDict

        lazy = LazyActionDict(
            {"description": "test", "switches": {"sw1": False}},
            topology_cache=MagicMock()
        )

        assert lazy["description"] == "test"
        assert lazy.get("description") == "test"
        assert lazy.get("nonexistent", "default") == "default"
        assert "description" in lazy
        assert "nonexistent" not in lazy

    def test_vl_ids_derived_from_switch_to_vl(self):
        """Impacted VL IDs should be derived from _switch_to_vl mapping."""
        from expert_op4grid_recommender.data_loader import LazyActionDict

        cache = MagicMock()
        cache._switch_to_vl = {"sw1": "VL_A", "sw2": "VL_B"}
        cache.compute_bus_assignments.return_value = {}
        cache.get_element_bus_assignments.return_value = {}

        lazy = LazyActionDict(
            {"switches": {"sw1": False, "sw2": True}},
            topology_cache=cache
        )

        lazy["content"]

        # Check that compute_bus_assignments was called with both VLs
        call_args = cache.compute_bus_assignments.call_args
        assert call_args[0][0] == {"sw1": False, "sw2": True}
        assert call_args[0][1] == {"VL_A", "VL_B"}

    def test_unknown_switch_ids_produce_warning(self):
        """Switches not in _switch_to_vl should produce empty content with warning."""
        from expert_op4grid_recommender.data_loader import LazyActionDict

        cache = MagicMock()
        cache._switch_to_vl = {}  # No switches known

        lazy = LazyActionDict(
            {"switches": {"unknown_sw": False}},
            topology_cache=cache
        )

        content = lazy["content"]

        assert content == {"set_bus": {}}
        cache.compute_bus_assignments.assert_not_called()

    def test_cache_exception_returns_empty_set_bus(self):
        """If cache computation fails, content should have empty set_bus."""
        from expert_op4grid_recommender.data_loader import LazyActionDict

        cache = MagicMock()
        cache._switch_to_vl = {"sw1": "VL1"}
        cache.compute_bus_assignments.side_effect = RuntimeError("computation failed")

        lazy = LazyActionDict(
            {"switches": {"sw1": False}},
            topology_cache=cache
        )

        content = lazy["content"]

        assert content == {"set_bus": {}, "switches": {"sw1": False}}


class TestEnrichActionsLazy:
    """Tests for the enrich_actions_lazy function."""

    def test_wraps_all_actions(self):
        """All actions should be wrapped as LazyActionDict instances."""
        from expert_op4grid_recommender.data_loader import LazyActionDict

        with patch('expert_op4grid_recommender.data_loader.enrich_actions_lazy') as mock_fn:
            # Test the real function instead
            pass

        from expert_op4grid_recommender.data_loader import enrich_actions_lazy

        mock_network = MagicMock()
        # Mock NetworkTopologyCache constructor
        with patch('expert_op4grid_recommender.utils.conversion_actions_repas.NetworkTopologyCache') as MockCache:
            MockCache.return_value = MagicMock()

            dict_actions = {
                "a1": {"switches": {"sw1": False}},
                "a2": {"switches": {"sw2": True}},
            }

            result = enrich_actions_lazy(dict_actions, mock_network)

            assert len(result) == 2
            for action_id in result:
                assert isinstance(result[action_id], LazyActionDict)

    def test_shared_cache_across_actions(self):
        """All actions should share the same NetworkTopologyCache instance."""
        from expert_op4grid_recommender.data_loader import enrich_actions_lazy, LazyActionDict

        mock_network = MagicMock()
        with patch('expert_op4grid_recommender.utils.conversion_actions_repas.NetworkTopologyCache') as MockCache:
            cache_instance = MagicMock()
            MockCache.return_value = cache_instance

            dict_actions = {
                "a1": {"switches": {"sw1": False}},
                "a2": {"switches": {"sw2": True}},
            }

            result = enrich_actions_lazy(dict_actions, mock_network)

            # Both actions should reference the same cache
            assert result["a1"]._topology_cache is result["a2"]._topology_cache
            assert result["a1"]._topology_cache is cache_instance

    def test_preserves_existing_content(self):
        """Actions that already have content should retain it."""
        from expert_op4grid_recommender.data_loader import enrich_actions_lazy

        mock_network = MagicMock()
        with patch('expert_op4grid_recommender.utils.conversion_actions_repas.NetworkTopologyCache') as MockCache:
            MockCache.return_value = MagicMock()

            existing_content = {"set_bus": {"lines_or_id": {"L1": 1}}}
            dict_actions = {
                "a1": {"content": existing_content, "switches": {"sw1": False}},
            }

            result = enrich_actions_lazy(dict_actions, mock_network)

            assert result["a1"]._content_computed
            assert result["a1"]["content"] == existing_content

    def test_action_id_passed_to_lazy_dict(self):
        """Each LazyActionDict should receive its action_id."""
        from expert_op4grid_recommender.data_loader import enrich_actions_lazy

        mock_network = MagicMock()
        with patch('expert_op4grid_recommender.utils.conversion_actions_repas.NetworkTopologyCache') as MockCache:
            MockCache.return_value = MagicMock()

            dict_actions = {
                "disco_LINEA": {"description_unitaire": "Ouverture de la ligne 'LINEA'"},
                "switch_action": {"switches": {"sw1": False}},
            }

            result = enrich_actions_lazy(dict_actions, mock_network)

            assert result["disco_LINEA"]._action_id == "disco_LINEA"
            assert result["switch_action"]._action_id == "switch_action"

    def test_disco_action_content_via_enrich(self):
        """disco_* actions should produce correct content after enrich_actions_lazy."""
        from expert_op4grid_recommender.data_loader import enrich_actions_lazy

        mock_network = MagicMock()
        with patch('expert_op4grid_recommender.utils.conversion_actions_repas.NetworkTopologyCache') as MockCache:
            MockCache.return_value = MagicMock()

            dict_actions = {
                "disco_MYLINE": {"description_unitaire": "Ouverture de la ligne 'MYLINE'"},
            }

            result = enrich_actions_lazy(dict_actions, mock_network)
            content = result["disco_MYLINE"]["content"]

            assert content["set_bus"]["lines_or_id"] == {"MYLINE": -1}
            assert content["set_bus"]["lines_ex_id"] == {"MYLINE": -1}
