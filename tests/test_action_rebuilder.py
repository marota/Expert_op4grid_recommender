#!/usr/bin/python3
# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""
Tests for the action_rebuilder module.

This module tests:
- make_description_unitaire
- make_raw_all_actions_dict
- build_action_dict_for_snapshot_from_scratch
- rebuild_action_dict_for_snapshot
- run_rebuild_actions
"""

import pytest
import os
import sys
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime

# --- Test Setup: Add Project Root to Python path ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from expert_op4grid_recommender.utils.action_rebuilder import (
    make_description_unitaire,
    make_raw_all_actions_dict,
    build_action_dict_for_snapshot_from_scratch,
    rebuild_action_dict_for_snapshot,
    run_rebuild_actions
)


# --- Mock Classes ---

class MockRepasAction:
    """
    Mock of a REPAS action object.
    
    Simulates the structure of actions parsed from REPAS JSON files.
    """
    def __init__(self, action_id, switches_by_voltage_level):
        self._id = action_id
        self._switches_by_voltage_level = switches_by_voltage_level


class MockNetwork:
    """
    Mock of a pypowsybl network object.
    """
    def __init__(self, case_date=None):
        self.case_date = case_date or datetime(2024, 8, 28, 1, 0)
    
    def get_voltage_levels(self):
        return Mock()
    
    def get_lines(self):
        return Mock()


# --- Fixtures ---

@pytest.fixture
def single_voltage_level_action():
    """Creates a mock REPAS action with a single voltage level."""
    return MockRepasAction(
        action_id="ACTION_001",
        switches_by_voltage_level={
            "SUBSTATION_A": {"SUBSTATION_A_SWITCH_1 DJ_OC": True}
        }
    )


@pytest.fixture
def multi_voltage_level_action():
    """Creates a mock REPAS action with multiple voltage levels."""
    return MockRepasAction(
        action_id="ACTION_002",
        switches_by_voltage_level={
            "SUBSTATION_A": {"SUBSTATION_A_COUPL_1 DJ_OC": True},
            "SUBSTATION_B": {"SUBSTATION_B_COUPL_2 DJ_OC": False}
        }
    )


@pytest.fixture
def coupling_action():
    """Creates a mock REPAS action involving a coupling breaker."""
    return MockRepasAction(
        action_id="ACTION_COUPL",
        switches_by_voltage_level={
            "SUBSTATION_A": {"SUBSTATION_A_COUPL_DJ DJ_OC": True}
        }
    )


@pytest.fixture
def tro_action():
    """Creates a mock REPAS action involving a TRO (transformer) breaker."""
    return MockRepasAction(
        action_id="ACTION_TRO",
        switches_by_voltage_level={
            "SUBSTATION_A": {"SUBSTATION_A_TRO.1 DJ_OC": False}
        }
    )


@pytest.fixture
def regular_action():
    """Creates a mock REPAS action without coupling or TRO."""
    return MockRepasAction(
        action_id="ACTION_REG",
        switches_by_voltage_level={
            "SUBSTATION_A": {"SUBSTATION_A_LINE_1 DJ_OC": True}
        }
    )


@pytest.fixture
def mock_network():
    """Creates a mock pypowsybl network."""
    return MockNetwork()


@pytest.fixture
def sample_dict_action():
    """Creates a sample action dictionary as would be loaded from JSON."""
    return {
        "ACTION_COUPL_SUBSTATION_A": {
            "description": "Ouverture COUPL dans SUBSTATION_A",
            "description_unitaire": "Ouverture COUPL switch dans le poste SUBSTATION_A",
            "VoltageLevelId": "SUBSTATION_A",
            "content": {"set_bus": {}}
        },
        "ACTION_REG_SUBSTATION_B": {
            "description": "Ouverture ligne dans SUBSTATION_B",
            "description_unitaire": "Ouverture LINE_1 dans le poste SUBSTATION_B",
            "VoltageLevelId": "SUBSTATION_B",
            "content": {"set_bus": {}}
        }
    }


# --- Tests for make_description_unitaire ---

class TestMakeDescriptionUnitaire:
    """Tests for the make_description_unitaire function."""
    
    def test_ouverture_description(self):
        """Test that 'Ouverture' is used when switch value is True."""
        switches = {"SUBSTATION_A": {"SWITCH_1 DJ_OC": True}}
        result = make_description_unitaire(switches)
        
        assert "Ouverture" in result
        assert "SWITCH_1 DJ_OC" in result
        assert "SUBSTATION_A" in result
    
    def test_fermeture_description(self):
        """Test that 'Fermeture' is used when switch value is False."""
        switches = {"SUBSTATION_B": {"SWITCH_2 DJ_OC": False}}
        result = make_description_unitaire(switches)
        
        assert "Fermeture" in result
        assert "SWITCH_2 DJ_OC" in result
        assert "SUBSTATION_B" in result
    
    def test_description_format(self):
        """Test the exact format of the description string."""
        switches = {"POSTE_X": {"POSTE_X_COUPL DJ_OC": True}}
        result = make_description_unitaire(switches)
        
        expected = "Ouverture POSTE_X_COUPL DJ_OC dans le poste POSTE_X"
        assert result == expected
    
    def test_single_entry_in_voltage_level(self):
        """Test with a single switch in a voltage level."""
        switches = {"VL1": {"SW1": False}}
        result = make_description_unitaire(switches)
        
        assert result == "Fermeture SW1 dans le poste VL1"


# --- Tests for make_raw_all_actions_dict ---

class TestMakeRawAllActionsDict:
    """Tests for the make_raw_all_actions_dict function."""
    
    def test_single_voltage_level_action(self, single_voltage_level_action):
        """Test that single voltage level actions are added directly."""
        all_actions = [single_voltage_level_action]
        result = make_raw_all_actions_dict(all_actions)
        
        assert "ACTION_001" in result
        assert len(result) == 1
        assert result["ACTION_001"]._id == "ACTION_001"
    
    def test_multi_voltage_level_action_split(self, multi_voltage_level_action):
        """Test that multi voltage level actions are split into separate entries."""
        all_actions = [multi_voltage_level_action]
        result = make_raw_all_actions_dict(all_actions)
        
        assert len(result) == 2
        assert "ACTION_002_SUBSTATION_A" in result
        assert "ACTION_002_SUBSTATION_B" in result
    
    def test_split_action_has_single_voltage_level(self, multi_voltage_level_action):
        """Test that split actions only contain their respective voltage level."""
        all_actions = [multi_voltage_level_action]
        result = make_raw_all_actions_dict(all_actions)
        
        action_a = result["ACTION_002_SUBSTATION_A"]
        assert len(action_a._switches_by_voltage_level) == 1
        assert "SUBSTATION_A" in action_a._switches_by_voltage_level
        
        action_b = result["ACTION_002_SUBSTATION_B"]
        assert len(action_b._switches_by_voltage_level) == 1
        assert "SUBSTATION_B" in action_b._switches_by_voltage_level
    
    def test_mixed_actions(self, single_voltage_level_action, multi_voltage_level_action):
        """Test with a mix of single and multi voltage level actions."""
        all_actions = [single_voltage_level_action, multi_voltage_level_action]
        result = make_raw_all_actions_dict(all_actions)
        
        assert len(result) == 3
        assert "ACTION_001" in result
        assert "ACTION_002_SUBSTATION_A" in result
        assert "ACTION_002_SUBSTATION_B" in result
    
    def test_empty_actions_list(self):
        """Test with an empty list of actions."""
        result = make_raw_all_actions_dict([])
        assert result == {}
    
    def test_original_action_not_modified(self, multi_voltage_level_action):
        """Test that the original action is not modified (deep copy is used)."""
        original_switches = dict(multi_voltage_level_action._switches_by_voltage_level)
        all_actions = [multi_voltage_level_action]
        
        make_raw_all_actions_dict(all_actions)
        
        # Original should still have both voltage levels
        assert len(multi_voltage_level_action._switches_by_voltage_level) == 2
        assert multi_voltage_level_action._switches_by_voltage_level == original_switches


# --- Tests for build_action_dict_for_snapshot_from_scratch ---

class TestBuildActionDictForSnapshotFromScratch:
    """Tests for the build_action_dict_for_snapshot_from_scratch function."""
    
    @patch('expert_op4grid_recommender.utils.action_rebuilder.convert_repas_actions_to_grid2op_actions')
    def test_filters_coupling_actions(self, mock_convert, mock_network, coupling_action, regular_action):
        """Test that only COUPL actions are sent for conversion."""
        mock_convert.return_value = {"converted_action": {}}
        all_actions = [coupling_action, regular_action]
        
        build_action_dict_for_snapshot_from_scratch(mock_network, all_actions)
        
        # Check that convert was called with only the coupling action
        call_args = mock_convert.call_args[0]
        actions_to_convert = call_args[1]
        assert len(actions_to_convert) == 1
        assert actions_to_convert[0]._id == "ACTION_COUPL"
    
    @patch('expert_op4grid_recommender.utils.action_rebuilder.convert_repas_actions_to_grid2op_actions')
    def test_filters_tro_actions(self, mock_convert, mock_network, tro_action, regular_action):
        """Test that TRO actions are also sent for conversion."""
        mock_convert.return_value = {"converted_action": {}}
        all_actions = [tro_action, regular_action]
        
        build_action_dict_for_snapshot_from_scratch(mock_network, all_actions)
        
        call_args = mock_convert.call_args[0]
        actions_to_convert = call_args[1]
        assert len(actions_to_convert) == 1
        assert actions_to_convert[0]._id == "ACTION_TRO"
    
    @patch('expert_op4grid_recommender.utils.action_rebuilder.convert_repas_actions_to_grid2op_actions')
    def test_returns_converted_actions(self, mock_convert, mock_network, coupling_action):
        """Test that the function returns the converted actions."""
        expected_result = {"ACTION_COUPL_converted": {"content": {}}}
        mock_convert.return_value = expected_result
        
        result = build_action_dict_for_snapshot_from_scratch(mock_network, [coupling_action])
        
        assert result == expected_result
    
    @patch('expert_op4grid_recommender.utils.action_rebuilder.create_dict_disco_reco_lines_disco')
    @patch('expert_op4grid_recommender.utils.action_rebuilder.convert_repas_actions_to_grid2op_actions')
    def test_adds_reco_disco_actions_when_flag_true(self, mock_convert, mock_create_disco, mock_network, coupling_action):
        """Test that disco/reco actions are added when flag is True."""
        mock_convert.return_value = {"converted": {}}
        mock_create_disco.return_value = {"disco_action": {}}
        
        result = build_action_dict_for_snapshot_from_scratch(
            mock_network, [coupling_action], 
            add_reco_disco_actions=True,
            filter_voltage_levels=["VL1"]
        )
        
        mock_create_disco.assert_called_once_with(mock_network, filter_voltage_levels=["VL1"])
        assert "converted" in result
        assert "disco_action" in result
    
    @patch('expert_op4grid_recommender.utils.action_rebuilder.create_dict_disco_reco_lines_disco')
    @patch('expert_op4grid_recommender.utils.action_rebuilder.convert_repas_actions_to_grid2op_actions')
    def test_no_reco_disco_actions_when_flag_false(self, mock_convert, mock_create_disco, mock_network, coupling_action):
        """Test that disco/reco actions are not added when flag is False."""
        mock_convert.return_value = {"converted": {}}
        
        build_action_dict_for_snapshot_from_scratch(mock_network, [coupling_action], add_reco_disco_actions=False)
        
        mock_create_disco.assert_not_called()


# --- Tests for rebuild_action_dict_for_snapshot ---

class TestRebuildActionDictForSnapshot:
    """Tests for the rebuild_action_dict_for_snapshot function."""
    
    @patch('expert_op4grid_recommender.utils.action_rebuilder.convert_repas_actions_to_grid2op_actions')
    def test_keeps_non_coupling_actions(self, mock_convert, mock_network, regular_action):
        """Test that non-COUPL/TRO actions are kept as-is."""
        mock_convert.return_value = {}
        
        dict_action = {
            "ACTION_REG_SUBSTATION_A": {
                "description": "Regular action",
                "description_unitaire": "Regular action description",
                "content": {"set_bus": {}}
            }
        }
        
        result = rebuild_action_dict_for_snapshot(mock_network, [regular_action], dict_action)
        
        assert "ACTION_REG_SUBSTATION_A" in result
        assert result["ACTION_REG_SUBSTATION_A"]["description"] == "Regular action"
    
    @patch('expert_op4grid_recommender.utils.action_rebuilder.convert_repas_actions_to_grid2op_actions')
    def test_converts_coupling_actions(self, mock_convert, mock_network, coupling_action):
        """Test that COUPL actions are sent for conversion."""
        mock_convert.return_value = {"ACTION_COUPL_SUBSTATION_A": {"converted": True}}
        
        dict_action = {
            "ACTION_COUPL_SUBSTATION_A": {
                "description": "COUPL action",
                "description_unitaire": "Ouverture COUPL dans SUBSTATION_A",
                "VoltageLevelId": "SUBSTATION_A",
                "content": {}
            }
        }
        
        result = rebuild_action_dict_for_snapshot(mock_network, [coupling_action], dict_action)
        
        assert mock_convert.called
        assert "ACTION_COUPL_SUBSTATION_A" in result
    
    @patch('expert_op4grid_recommender.utils.action_rebuilder.convert_repas_actions_to_grid2op_actions')
    def test_converts_tro_actions(self, mock_convert, mock_network, tro_action):
        """Test that TRO actions are sent for conversion."""
        mock_convert.return_value = {"ACTION_TRO_SUBSTATION_A": {"converted": True}}
        
        dict_action = {
            "ACTION_TRO_SUBSTATION_A": {
                "description": "TRO action",
                "description_unitaire": "Ouverture TRO.1 dans SUBSTATION_A",
                "VoltageLevelId": "SUBSTATION_A",
                "content": {}
            }
        }
        
        result = rebuild_action_dict_for_snapshot(mock_network, [tro_action], dict_action)
        
        assert mock_convert.called
    
    @patch('expert_op4grid_recommender.utils.action_rebuilder.convert_repas_actions_to_grid2op_actions')
    def test_preserves_description_unitaire(self, mock_convert, mock_network, coupling_action):
        """Test that description_unitaire is preserved from original dict_action."""
        mock_convert.return_value = {
            "ACTION_COUPL_SUBSTATION_A": {"converted": True}
        }
        
        dict_action = {
            "ACTION_COUPL_SUBSTATION_A": {
                "description": "COUPL action",
                "description_unitaire": "Custom description unitaire",
                "VoltageLevelId": "SUBSTATION_A",
                "content": {}
            }
        }
        
        result = rebuild_action_dict_for_snapshot(mock_network, [coupling_action], dict_action)
        
        assert result["ACTION_COUPL_SUBSTATION_A"]["description_unitaire"] == "Custom description unitaire"
    
    @patch('expert_op4grid_recommender.utils.action_rebuilder.convert_repas_actions_to_grid2op_actions')
    def test_warning_for_missing_action(self, mock_convert, mock_network, capsys):
        """Test that a warning is printed when action is not found in REPAS actions."""
        mock_convert.return_value = {}
        
        dict_action = {
            "MISSING_ACTION_VL1": {
                "description": "COUPL missing",
                "description_unitaire": "Ouverture COUPL missing",
                "VoltageLevelId": "VL1",
                "content": {}
            }
        }
        
        rebuild_action_dict_for_snapshot(mock_network, [], dict_action)
        
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "not found" in captured.out


# --- Tests for run_rebuild_actions ---

class TestRunRebuildActions:
    """Tests for the run_rebuild_actions function."""
    
    @patch('expert_op4grid_recommender.utils.action_rebuilder.build_action_dict_for_snapshot_from_scratch')
    @patch('expert_op4grid_recommender.utils.action_rebuilder.repas.parse_json')
    @patch('builtins.open', create=True)
    def test_from_scratch_mode(self, mock_open, mock_parse, mock_build, mock_network, tmp_path):
        """Test that from_scratch mode calls build_action_dict_for_snapshot_from_scratch."""
        mock_parse.return_value = [Mock()]
        mock_build.return_value = {"action": {}}
        mock_open.return_value.__enter__ = Mock(return_value=Mock())
        mock_open.return_value.__exit__ = Mock(return_value=False)
        
        # Create temp directory structure
        os.makedirs(os.path.join("data", "action_space"), exist_ok=True)
        
        try:
            run_rebuild_actions(
                mock_network,
                do_from_scratch=True,
                repas_file_path="fake_repas.json"
            )
            
            mock_build.assert_called_once()
        finally:
            # Cleanup
            if os.path.exists(os.path.join("data", "action_space")):
                import shutil
                shutil.rmtree("data", ignore_errors=True)
    
    @patch('expert_op4grid_recommender.utils.action_rebuilder.rebuild_action_dict_for_snapshot')
    @patch('expert_op4grid_recommender.utils.action_rebuilder.repas.parse_json')
    @patch('builtins.open', create=True)
    def test_rebuild_mode(self, mock_open, mock_parse, mock_rebuild, mock_network, tmp_path):
        """Test that non-from_scratch mode calls rebuild_action_dict_for_snapshot."""
        mock_parse.return_value = [Mock()]
        mock_rebuild.return_value = {"action": {}}
        mock_open.return_value.__enter__ = Mock(return_value=Mock())
        mock_open.return_value.__exit__ = Mock(return_value=False)
        
        os.makedirs(os.path.join("data", "action_space"), exist_ok=True)
        
        try:
            run_rebuild_actions(
                mock_network,
                do_from_scratch=False,
                repas_file_path="fake_repas.json",
                dict_action_to_filter_on={"existing": {}}
            )
            
            mock_rebuild.assert_called_once()
        finally:
            import shutil
            shutil.rmtree("data", ignore_errors=True)
    
    @patch('expert_op4grid_recommender.utils.action_rebuilder.repas.parse_json')
    def test_voltage_filter_threshold(self, mock_parse, mock_network):
        """Test that voltage filter threshold is passed to parse_json."""
        mock_parse.return_value = []
        
        os.makedirs(os.path.join("data", "action_space"), exist_ok=True)
        
        try:
            run_rebuild_actions(
                mock_network,
                do_from_scratch=True,
                repas_file_path="fake_repas.json",
                voltage_filter_threshold=225
            )
            
            # Check that parse_json was called with a filter function
            call_args = mock_parse.call_args
            filter_func = call_args[0][2]
            
            # Test the filter function
            assert filter_func(("id", {"nominal_v": 100})) == True  # 100 < 225
            assert filter_func(("id", {"nominal_v": 300})) == False  # 300 >= 225
        finally:
            import shutil
            shutil.rmtree("data", ignore_errors=True)
    
    @patch('expert_op4grid_recommender.utils.action_rebuilder.repas.parse_json')
    def test_returns_original_on_failure(self, mock_parse, mock_network):
        """Test that original dict is returned on failure."""
        mock_parse.side_effect = Exception("Parse error")
        
        original_dict = {"original": {"content": {}}}
        
        result = run_rebuild_actions(
            mock_network,
            do_from_scratch=False,
            repas_file_path="fake_repas.json",
            dict_action_to_filter_on=original_dict
        )
        
        assert result == original_dict
    
    def test_default_dict_action_is_empty(self, mock_network):
        """Test that default dict_action_to_filter_on is empty dict."""
        with patch('expert_op4grid_recommender.utils.action_rebuilder.repas.parse_json') as mock_parse:
            mock_parse.side_effect = Exception("Parse error")
            
            result = run_rebuild_actions(
                mock_network,
                do_from_scratch=True,
                repas_file_path="fake_repas.json"
            )
            
            assert result == {}
    
    def test_prints_rebuild_message(self, mock_network, capsys):
        """Test that informative message is printed at start."""
        with patch('expert_op4grid_recommender.utils.action_rebuilder.repas.parse_json') as mock_parse:
            mock_parse.side_effect = Exception("Stop early")
            
            run_rebuild_actions(
                mock_network,
                do_from_scratch=True,
                repas_file_path="test_repas.json",
                voltage_filter_threshold=400
            )
            
            captured = capsys.readouterr()
            assert "Rebuilding action dictionary" in captured.out
            assert "test_repas.json" in captured.out
            assert "400" in captured.out


# --- Integration Tests (marked as slow) ---

@pytest.mark.slow
class TestActionRebuilderIntegration:
    """Integration tests that require actual files and environment."""
    
    def test_with_real_repas_file(self):
        """Test with actual REPAS file if available."""
        repas_file = Path(__file__).parent.parent / "data" / "action_space" / "allLogics.2024.12.10.json"
        
        if not repas_file.exists():
            pytest.skip("REPAS file not found for integration test")
        
        # This would require a real network, so we skip for now
        pytest.skip("Full integration test requires pypowsybl network")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
