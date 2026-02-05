#!/usr/bin/python3
# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""
Tests for the action_rebuilder module and conversion_actions_repas module.

This module tests:
- make_description_unitaire
- make_raw_all_actions_dict
- build_action_dict_for_snapshot_from_scratch
- rebuild_action_dict_for_snapshot
- run_rebuild_actions
- UnionFind data structure
- _reindex_bus_numbers_per_vl function
- NetworkTopologyCache class
- convert_to_grid2op_action (analytical version)
- convert_to_grid2op_action_batch
- convert_to_grid2op_action_with_variant (legacy version)
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# --- Test Setup: Add Project Root to Python path ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from expert_op4grid_recommender.utils.action_rebuilder import (
    make_description_unitaire,
    make_raw_all_actions_dict,
    build_action_dict_for_snapshot_from_scratch,
    rebuild_action_dict_for_snapshot,
    run_rebuild_actions
)

from expert_op4grid_recommender.utils.conversion_actions_repas import (
    UnionFind,
    _reindex_bus_numbers_per_vl,
    NetworkTopologyCache,
    convert_to_grid2op_action,
    convert_to_grid2op_action_batch,
    convert_to_grid2op_action_with_variant,
    convert_repas_actions_to_grid2op_actions,
    get_all_switch_descriptions,
)


# =============================================================================
# Mock Classes
# =============================================================================

class MockRepasAction:
    """
    Mock of a REPAS action object.
    
    Simulates the structure of actions parsed from REPAS JSON files.
    """
    def __init__(self, action_id, switches_by_voltage_level, description="Test action"):
        self._id = action_id
        self._switches_by_voltage_level = switches_by_voltage_level
        self._description = description
        self._loads_by_id = {}
        self._generators_by_id = {}


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


def create_mock_network_with_topology():
    """
    Create a mock network with realistic topology data for testing.
    
    Creates a simple network with:
    - 2 voltage levels (VL1, VL2)
    - 4 switches in VL1 (2 open, 2 closed)
    - 2 branches connecting VL1 to VL2
    - 1 load in VL1
    - 1 generator in VL1
    """
    mock_network = MagicMock()
    
    # Switches DataFrame
    switches_data = {
        'voltage_level_id': ['VL1', 'VL1', 'VL1', 'VL1'],
        'bus_breaker_bus1_id': ['VL1_NODE1', 'VL1_NODE2', 'VL1_NODE1', 'VL1_NODE3'],
        'bus_breaker_bus2_id': ['VL1_NODE2', 'VL1_NODE3', 'VL1_NODE3', 'VL1_NODE4'],
        'open': [False, True, False, True]  # SW1 closed, SW2 open, SW3 closed, SW4 open
    }
    switches_df = pd.DataFrame(switches_data, index=['VL1_SW1', 'VL1_SW2', 'VL1_SW3', 'VL1_SW4'])
    mock_network.get_switches.return_value = switches_df
    
    # Loads DataFrame
    loads_data = {
        'bus_id': ['VL1_NODE1'],
        'voltage_level_id': ['VL1']
    }
    loads_df = pd.DataFrame(loads_data, index=['LOAD1'])
    mock_network.get_loads.return_value = loads_df
    
    # Generators DataFrame
    generators_data = {
        'bus_id': ['VL1_NODE4'],
        'voltage_level_id': ['VL1']
    }
    generators_df = pd.DataFrame(generators_data, index=['GEN1'])
    mock_network.get_generators.return_value = generators_df
    
    # Shunts DataFrame (empty)
    shunts_df = pd.DataFrame(columns=['bus_id', 'voltage_level_id'])
    shunts_df.index.name = 'id'
    mock_network.get_shunt_compensators.return_value = shunts_df
    
    # Branches DataFrame
    branches_data = {
        'bus1_id': ['VL1_NODE1', 'VL1_NODE4'],
        'bus2_id': ['VL2_NODE1', 'VL2_NODE2'],
        'voltage_level1_id': ['VL1', 'VL1'],
        'voltage_level2_id': ['VL2', 'VL2']
    }
    branches_df = pd.DataFrame(branches_data, index=['LINE1', 'LINE2'])
    mock_network.get_branches.return_value = branches_df
    
    return mock_network


# =============================================================================
# Fixtures
# =============================================================================

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
        action_id="ACTION-COUPL",
        switches_by_voltage_level={
            "SUBSTATION-A": {"SUBSTATION_A_COUPL_DJ DJ_OC": True}
        }
    )


@pytest.fixture
def tro_action():
    """Creates a mock REPAS action involving a TRO (transformer) breaker."""
    return MockRepasAction(
        action_id="ACTION-TRO",
        switches_by_voltage_level={
            "SUBSTATION-A": {"SUBSTATION_A_TRO.1 DJ_OC": False}
        }
    )


@pytest.fixture
def regular_action():
    """Creates a mock REPAS action without coupling or TRO."""
    return MockRepasAction(
        action_id="ACTION-REG",
        switches_by_voltage_level={
            "SUBSTATION-A": {"SUBSTATION_A_LINE_1 DJ_OC": True}
        }
    )


@pytest.fixture
def mock_network():
    """Creates a mock pypowsybl network."""
    return MockNetwork()


@pytest.fixture
def mock_network_with_topology():
    """Creates a mock pypowsybl network with full topology data."""
    return create_mock_network_with_topology()


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


# =============================================================================
# Tests for UnionFind
# =============================================================================

class TestUnionFind:
    """Tests for the UnionFind data structure."""
    
    def test_initialization(self):
        """Test that UnionFind initializes correctly with elements."""
        elements = ['A', 'B', 'C']
        uf = UnionFind(elements)
        
        assert len(uf.parent) == 3
        assert all(uf.parent[e] == e for e in elements)
        assert all(uf.rank[e] == 0 for e in elements)
    
    def test_find_without_union(self):
        """Test find operation on elements that haven't been united."""
        uf = UnionFind(['A', 'B', 'C'])
        
        assert uf.find('A') == 'A'
        assert uf.find('B') == 'B'
        assert uf.find('C') == 'C'
    
    def test_union_two_elements(self):
        """Test union of two elements."""
        uf = UnionFind(['A', 'B', 'C'])
        uf.union('A', 'B')
        
        # A and B should have the same root
        assert uf.find('A') == uf.find('B')
        # C should still be separate
        assert uf.find('C') == 'C'
        assert uf.find('A') != uf.find('C')
    
    def test_union_chain(self):
        """Test chaining unions."""
        uf = UnionFind(['A', 'B', 'C', 'D'])
        uf.union('A', 'B')
        uf.union('B', 'C')
        
        # A, B, C should all have the same root
        assert uf.find('A') == uf.find('B') == uf.find('C')
        # D should still be separate
        assert uf.find('D') == 'D'
    
    def test_union_already_same_set(self):
        """Test union of elements already in the same set."""
        uf = UnionFind(['A', 'B'])
        uf.union('A', 'B')
        uf.union('A', 'B')  # Should not cause issues
        
        assert uf.find('A') == uf.find('B')
    
    def test_get_component_mapping_single_component(self):
        """Test component mapping when all elements are connected."""
        uf = UnionFind(['A', 'B', 'C'])
        uf.union('A', 'B')
        uf.union('B', 'C')
        
        mapping = uf.get_component_mapping()
        
        # All should have the same component number
        assert mapping['A'] == mapping['B'] == mapping['C']
        assert mapping['A'] == 1  # First component is 1
    
    def test_get_component_mapping_multiple_components(self):
        """Test component mapping with multiple disconnected components."""
        uf = UnionFind(['A', 'B', 'C', 'D'])
        uf.union('A', 'B')
        uf.union('C', 'D')
        
        mapping = uf.get_component_mapping()
        
        # A and B should be in one component
        assert mapping['A'] == mapping['B']
        # C and D should be in another component
        assert mapping['C'] == mapping['D']
        # The two components should be different
        assert mapping['A'] != mapping['C']
    
    def test_get_component_mapping_all_separate(self):
        """Test component mapping when no unions have been made."""
        uf = UnionFind(['A', 'B', 'C'])
        
        mapping = uf.get_component_mapping()
        
        # Each element should be in its own component
        assert len(set(mapping.values())) == 3
    
    def test_path_compression(self):
        """Test that path compression works (find should update parent)."""
        uf = UnionFind(['A', 'B', 'C', 'D'])
        uf.union('A', 'B')
        uf.union('B', 'C')
        uf.union('C', 'D')
        
        # After finding D, the path should be compressed
        root = uf.find('D')
        # D's parent should now point directly to root (or close to it)
        assert uf.parent['D'] == root
    
    def test_empty_union_find(self):
        """Test UnionFind with empty element list."""
        uf = UnionFind([])
        
        assert len(uf.parent) == 0
        assert uf.get_component_mapping() == {}


# =============================================================================
# Tests for _reindex_bus_numbers_per_vl
# =============================================================================

class TestReindexBusNumbersPerVl:
    """Tests for the _reindex_bus_numbers_per_vl function."""
    
    def test_basic_reindexing_single_vl(self):
        """Test basic reindexing of non-sequential bus numbers in single VL."""
        set_bus = {
            'lines_or_id': {'L1': 2, 'L2': 4, 'L3': 2},
            'lines_ex_id': {},
            'loads_id': {'LOAD1': 4},
            'generators_id': {},
            'shunts_id': {}
        }
        # All elements in same voltage level
        element_to_vl = {'L1': 'VL1', 'L2': 'VL1', 'L3': 'VL1', 'LOAD1': 'VL1'}
        
        result = _reindex_bus_numbers_per_vl(set_bus, element_to_vl)
        
        # 2 -> 1, 4 -> 2
        assert result['lines_or_id']['L1'] == 1
        assert result['lines_or_id']['L2'] == 2
        assert result['lines_or_id']['L3'] == 1
        assert result['loads_id']['LOAD1'] == 2
    
    def test_preserves_disconnected(self):
        """Test that disconnected elements (bus=-1) are preserved."""
        set_bus = {
            'lines_or_id': {'L1': 2, 'L2': -1},
            'lines_ex_id': {'L3': -1},
            'loads_id': {},
            'generators_id': {},
            'shunts_id': {}
        }
        element_to_vl = {'L1': 'VL1', 'L2': 'VL1', 'L3': 'VL1'}
        
        result = _reindex_bus_numbers_per_vl(set_bus, element_to_vl)
        
        assert result['lines_or_id']['L2'] == -1
        assert result['lines_ex_id']['L3'] == -1
    
    def test_already_sequential(self):
        """Test that already sequential numbers are preserved."""
        set_bus = {
            'lines_or_id': {'L1': 1, 'L2': 2, 'L3': 1},
            'lines_ex_id': {},
            'loads_id': {},
            'generators_id': {},
            'shunts_id': {}
        }
        element_to_vl = {'L1': 'VL1', 'L2': 'VL1', 'L3': 'VL1'}
        
        result = _reindex_bus_numbers_per_vl(set_bus, element_to_vl)
        
        assert result['lines_or_id']['L1'] == 1
        assert result['lines_or_id']['L2'] == 2
        assert result['lines_or_id']['L3'] == 1
    
    def test_single_bus_number(self):
        """Test with only one unique bus number (reindexed to 1)."""
        set_bus = {
            'lines_or_id': {'L1': 5, 'L2': 5},
            'lines_ex_id': {},
            'loads_id': {},
            'generators_id': {},
            'shunts_id': {}
        }
        element_to_vl = {'L1': 'VL1', 'L2': 'VL1'}
        
        result = _reindex_bus_numbers_per_vl(set_bus, element_to_vl)
        
        # Single bus 5 -> 1
        expected_set_bus = {
            'lines_or_id': {'L1': 1, 'L2': 1},
            'lines_ex_id': {},
            'loads_id': {},
            'generators_id': {},
            'shunts_id': {}
        }
        assert result == expected_set_bus
    
    def test_empty_set_bus(self):
        """Test with empty element dictionaries."""
        set_bus = {
            'lines_or_id': {},
            'lines_ex_id': {},
            'loads_id': {},
            'generators_id': {},
            'shunts_id': {}
        }
        element_to_vl = {}
        
        result = _reindex_bus_numbers_per_vl(set_bus, element_to_vl)
        
        assert result == set_bus
    
    def test_large_gap_in_numbers(self):
        """Test with large gaps in bus numbers."""
        set_bus = {
            'lines_or_id': {'L1': 1, 'L2': 100, 'L3': 50},
            'lines_ex_id': {},
            'loads_id': {},
            'generators_id': {},
            'shunts_id': {}
        }
        element_to_vl = {'L1': 'VL1', 'L2': 'VL1', 'L3': 'VL1'}
        
        result = _reindex_bus_numbers_per_vl(set_bus, element_to_vl)
        
        # 1 -> 1, 50 -> 2, 100 -> 3
        assert result['lines_or_id']['L1'] == 1
        assert result['lines_or_id']['L3'] == 2
        assert result['lines_or_id']['L2'] == 3
    
    def test_mixed_positive_and_negative(self):
        """Test with a mix of positive bus numbers and disconnected (-1)."""
        set_bus = {
            'lines_or_id': {'L1': 3, 'L2': -1, 'L3': 7},
            'lines_ex_id': {'L4': 3, 'L5': -1},
            'loads_id': {'LOAD1': 7},
            'generators_id': {'GEN1': -1},
            'shunts_id': {}
        }
        element_to_vl = {
            'L1': 'VL1', 'L2': 'VL1', 'L3': 'VL1', 
            'L4': 'VL1', 'L5': 'VL1', 
            'LOAD1': 'VL1', 'GEN1': 'VL1'
        }
        
        result = _reindex_bus_numbers_per_vl(set_bus, element_to_vl)
        
        # 3 -> 1, 7 -> 2; -1 stays -1
        assert result['lines_or_id']['L1'] == 1
        assert result['lines_or_id']['L2'] == -1
        assert result['lines_or_id']['L3'] == 2
        assert result['lines_ex_id']['L4'] == 1
        assert result['lines_ex_id']['L5'] == -1
        assert result['loads_id']['LOAD1'] == 2
        assert result['generators_id']['GEN1'] == -1
    
    def test_preserves_zero_bus(self):
        """Test that bus=0 (invalid) is also preserved."""
        set_bus = {
            'lines_or_id': {'L1': 2, 'L2': 0},
            'lines_ex_id': {},
            'loads_id': {},
            'generators_id': {},
            'shunts_id': {}
        }
        element_to_vl = {'L1': 'VL1', 'L2': 'VL1'}
        
        result = _reindex_bus_numbers_per_vl(set_bus, element_to_vl)
        
        assert result['lines_or_id']['L2'] == 0
    
    def test_multiple_voltage_levels_independent_reindexing(self):
        """Test that different voltage levels are reindexed independently."""
        set_bus = {
            'lines_or_id': {'L1': 5, 'L2': 10},  # L1 in VL1, L2 in VL2
            'lines_ex_id': {},
            'loads_id': {'LOAD1': 5, 'LOAD2': 10},  # LOAD1 in VL1, LOAD2 in VL2
            'generators_id': {},
            'shunts_id': {}
        }
        element_to_vl = {'L1': 'VL1', 'L2': 'VL2', 'LOAD1': 'VL1', 'LOAD2': 'VL2'}
        
        result = _reindex_bus_numbers_per_vl(set_bus, element_to_vl)
        
        # VL1: 5 -> 1; VL2: 10 -> 1 (independent reindexing)
        assert result['lines_or_id']['L1'] == 1
        assert result['lines_or_id']['L2'] == 1
        assert result['loads_id']['LOAD1'] == 1
        assert result['loads_id']['LOAD2'] == 1
    
    def test_element_not_in_vl_mapping_unchanged(self):
        """Test that elements not in element_to_vl mapping are unchanged."""
        set_bus = {
            'lines_or_id': {'L1': 5, 'L2': 10},
            'lines_ex_id': {},
            'loads_id': {},
            'generators_id': {},
            'shunts_id': {}
        }
        # Only L1 is in the VL mapping
        element_to_vl = {'L1': 'VL1'}
        
        result = _reindex_bus_numbers_per_vl(set_bus, element_to_vl)
        
        # L1 should be reindexed, L2 should be unchanged
        assert result['lines_or_id']['L1'] == 1
        assert result['lines_or_id']['L2'] == 10


# =============================================================================
# Tests for NetworkTopologyCache
# =============================================================================

class TestNetworkTopologyCache:
    """Tests for the NetworkTopologyCache class."""
    
    def test_initialization(self, mock_network_with_topology):
        """Test that cache initializes correctly from network."""
        cache = NetworkTopologyCache(mock_network_with_topology)
        
        assert cache.network is mock_network_with_topology
        assert 'VL1' in cache._vl_nodes
        assert len(cache._vl_nodes['VL1']) == 4  # 4 unique nodes
    
    def test_switch_data_cached(self, mock_network_with_topology):
        """Test that switch data is properly cached."""
        cache = NetworkTopologyCache(mock_network_with_topology)
        
        # Should have 4 switches in VL1
        assert len(cache._vl_switches['VL1']) == 4
        
        # Check initial switch states
        assert cache._vl_switch_states['VL1_SW1'] == False  # closed
        assert cache._vl_switch_states['VL1_SW2'] == True   # open
    
    def test_element_mappings_cached(self, mock_network_with_topology):
        """Test that element-to-node mappings are cached."""
        cache = NetworkTopologyCache(mock_network_with_topology)
        
        assert 'LOAD1' in cache._load_to_node
        assert cache._load_to_node['LOAD1'] == 'VL1_NODE1'
        
        assert 'GEN1' in cache._gen_to_node
        assert cache._gen_to_node['GEN1'] == 'VL1_NODE4'
        
        assert 'LINE1' in cache._branch_or_to_node
        assert cache._branch_or_to_node['LINE1'] == 'VL1_NODE1'
    
    def test_compute_bus_assignments_no_changes(self, mock_network_with_topology):
        """Test bus assignment computation with no switch changes."""
        cache = NetworkTopologyCache(mock_network_with_topology)
        
        # No switch changes, use initial state
        node_to_bus = cache.compute_bus_assignments({}, {'VL1'})
        
        assert 'VL1' in node_to_bus
        # With initial state: SW1 closed, SW2 open, SW3 closed, SW4 open
        # NODE1-NODE2 connected (SW1), NODE1-NODE3 connected (SW3)
        # So NODE1, NODE2, NODE3 are in one component; NODE4 is separate
    
    def test_compute_bus_assignments_with_switch_change(self, mock_network_with_topology):
        """Test bus assignment computation with switch changes."""
        cache = NetworkTopologyCache(mock_network_with_topology)
        
        # Close SW4 to connect NODE3-NODE4
        switch_changes = {'VL1_SW4': False}  # False = closed
        node_to_bus = cache.compute_bus_assignments(switch_changes, {'VL1'})
        
        assert 'VL1' in node_to_bus
        # Now all nodes should be connected through the chain
    
    def test_compute_bus_assignments_empty_vl(self, mock_network_with_topology):
        """Test bus assignment for a voltage level with no nodes."""
        cache = NetworkTopologyCache(mock_network_with_topology)
        
        # VL2 has no switches in our mock
        node_to_bus = cache.compute_bus_assignments({}, {'VL2'})
        
        assert 'VL2' in node_to_bus
        assert node_to_bus['VL2'] == {'VL2_NODE1': -1, 'VL2_NODE2': -1} #assets are disconnected, hence -1 values
    
    def test_get_element_bus_assignments(self, mock_network_with_topology):
        """Test conversion from node-to-bus to element-to-bus mappings."""
        cache = NetworkTopologyCache(mock_network_with_topology)
        
        # Create a mock node_to_bus mapping
        node_to_bus = {
            'VL1': {
                'VL1_NODE1': 1,
                'VL1_NODE2': 1,
                'VL1_NODE3': 1,
                'VL1_NODE4': 2
            }
        }
        
        set_bus = cache.get_element_bus_assignments(node_to_bus, {'VL1'})
        
        assert 'lines_or_id' in set_bus
        assert 'loads_id' in set_bus
        assert 'generators_id' in set_bus
        
        # LINE1 origin is at VL1_NODE1 -> bus 1
        assert set_bus['lines_or_id']['LINE1'] == 1
        # LINE2 origin is at VL1_NODE4 -> bus 2
        assert set_bus['lines_or_id']['LINE2'] == 2
        # LOAD1 is at VL1_NODE1 -> bus 1
        assert set_bus['loads_id']['LOAD1'] == 1
        # GEN1 is at VL1_NODE4 -> bus 2
        assert set_bus['generators_id']['GEN1'] == 2
    
    def test_get_element_bus_assignments_reindexes(self, mock_network_with_topology):
        """Test that element bus assignments are reindexed to sequential numbers."""
        cache = NetworkTopologyCache(mock_network_with_topology)
        
        # Create a mock node_to_bus mapping with non-sequential bus numbers
        node_to_bus = {
            'VL1': {
                'VL1_NODE1': 3,
                'VL1_NODE2': 3,
                'VL1_NODE3': 3,
                'VL1_NODE4': 7
            }
        }
        
        set_bus = cache.get_element_bus_assignments(node_to_bus, {'VL1'})
        
        # Should be reindexed: 3 -> 1, 7 -> 2
        assert set_bus['lines_or_id']['LINE1'] == 1
        assert set_bus['lines_or_id']['LINE2'] == 2
        assert set_bus['loads_id']['LOAD1'] == 1
        assert set_bus['generators_id']['GEN1'] == 2


# =============================================================================
# Tests for convert_to_grid2op_action (analytical version)
# =============================================================================

class TestConvertToGrid2opAction:
    """Tests for the analytical convert_to_grid2op_action function."""
    
    def test_basic_conversion(self, mock_network_with_topology):
        """Test basic action conversion."""
        action = MockRepasAction(
            action_id="TEST_ACTION",
            switches_by_voltage_level={'VL1': {'VL1_SW2': False}},  # Close SW2
            description="Test action description"
        )
        
        result = convert_to_grid2op_action(mock_network_with_topology, action)
        
        assert 'description' in result
        assert result['description'] == "Test action description"
        assert 'content' in result
        assert 'set_bus' in result['content']
    
    def test_conversion_with_cache(self, mock_network_with_topology):
        """Test action conversion with pre-built cache."""
        cache = NetworkTopologyCache(mock_network_with_topology)
        
        action = MockRepasAction(
            action_id="TEST_ACTION",
            switches_by_voltage_level={'VL1': {'VL1_SW2': False}}
        )
        
        result = convert_to_grid2op_action(
            mock_network_with_topology, action, topology_cache=cache
        )
        
        assert 'content' in result
        assert 'set_bus' in result['content']
    
    def test_conversion_verbose(self, mock_network_with_topology, capsys):
        """Test verbose output during conversion."""
        action = MockRepasAction(
            action_id="VERBOSE_TEST",
            switches_by_voltage_level={'VL1': {'VL1_SW1': True}}
        )
        
        convert_to_grid2op_action(
            mock_network_with_topology, action, verbose=True
        )
        
        captured = capsys.readouterr()
        assert "VERBOSE_TEST" in captured.out
    
    def test_conversion_returns_correct_structure(self, mock_network_with_topology):
        """Test that conversion returns correct dictionary structure."""
        action = MockRepasAction(
            action_id="STRUCT_TEST",
            switches_by_voltage_level={'VL1': {'VL1_SW1': True}}
        )
        
        result = convert_to_grid2op_action(mock_network_with_topology, action)
        
        set_bus = result['content']['set_bus']
        assert 'lines_or_id' in set_bus
        assert 'lines_ex_id' in set_bus
        assert 'loads_id' in set_bus
        assert 'generators_id' in set_bus
        assert 'shunts_id' in set_bus
    
    def test_conversion_includes_switches_at_root(self, mock_network_with_topology):
        """Test that conversion includes switches at root level (not inside content)."""
        action = MockRepasAction(
            action_id="SWITCH_TEST",
            switches_by_voltage_level={'VL1': {'VL1_COUPL_DJ': True, 'VL1_LINE_SW': False}}
        )
        
        result = convert_to_grid2op_action(mock_network_with_topology, action)
        
        # New convention: switches field is at root level, not inside content
        assert 'switches' in result
        switches = result['switches']
        # Full switch IDs are used (as expected by pypowsybl)
        assert 'VL1_COUPL_DJ' in switches
        assert 'VL1_LINE_SW' in switches
        assert switches['VL1_COUPL_DJ'] == True
        assert switches['VL1_LINE_SW'] == False
    
    def test_conversion_multiple_voltage_levels(self, mock_network_with_topology):
        """Test conversion with switches in multiple voltage levels."""
        action = MockRepasAction(
            action_id="MULTI_VL_TEST",
            switches_by_voltage_level={
                'VL1': {'VL1_SW1': True},
                'VL2': {'VL2_SW2': False}
            }
        )
        
        result = convert_to_grid2op_action(mock_network_with_topology, action)
        
        # New convention: switches at root level with full IDs
        assert 'switches' in result
        switches = result['switches']
        assert 'VL1_SW1' in switches
        assert 'VL2_SW2' in switches


# =============================================================================
# Tests for convert_to_grid2op_action_batch
# =============================================================================

class TestConvertToGrid2opActionBatch:
    """Tests for the batch conversion function."""
    
    def test_empty_actions_list(self, mock_network_with_topology):
        """Test batch conversion with empty list."""
        result = convert_to_grid2op_action_batch(mock_network_with_topology, [])
        
        assert result == []
    
    def test_single_action_batch(self, mock_network_with_topology):
        """Test batch conversion with single action."""
        action = MockRepasAction(
            action_id="SINGLE",
            switches_by_voltage_level={'VL1': {'VL1_SW1': True}}
        )
        
        results = convert_to_grid2op_action_batch(mock_network_with_topology, [action])
        
        assert len(results) == 1
        assert 'content' in results[0]
    
    def test_multiple_actions_batch(self, mock_network_with_topology):
        """Test batch conversion with multiple actions."""
        actions = [
            MockRepasAction(
                action_id=f"ACTION_{i}",
                switches_by_voltage_level={'VL1': {'VL1_SW1': bool(i % 2)}}
            )
            for i in range(5)
        ]
        
        results = convert_to_grid2op_action_batch(mock_network_with_topology, actions)
        
        assert len(results) == 5
        for result in results:
            assert 'content' in result
            assert 'set_bus' in result['content']
    
    def test_batch_verbose(self, mock_network_with_topology, capsys):
        """Test verbose output during batch conversion."""
        actions = [
            MockRepasAction(
                action_id=f"ACTION_{i}",
                switches_by_voltage_level={'VL1': {'VL1_SW1': True}}
            )
            for i in range(100)
        ]
        
        convert_to_grid2op_action_batch(mock_network_with_topology, actions, verbose=True)
        
        captured = capsys.readouterr()
        assert "Building topology cache" in captured.out
        assert "50" in captured.out  # Progress message at 50
        assert "100" in captured.out  # Progress message at 100
    
    def test_batch_preserves_order(self, mock_network_with_topology):
        """Test that batch conversion preserves action order."""
        actions = [
            MockRepasAction(
                action_id=f"ACTION_{i}",
                switches_by_voltage_level={'VL1': {'VL1_SW1': True}},
                description=f"Description {i}"
            )
            for i in range(3)
        ]
        
        results = convert_to_grid2op_action_batch(mock_network_with_topology, actions)
        
        for i, result in enumerate(results):
            assert result['description'] == f"Description {i}"


# =============================================================================
# Tests for get_all_switch_descriptions
# =============================================================================

class TestGetAllSwitchDescriptions:
    """Tests for the get_all_switch_descriptions function."""
    
    def test_single_switch_ouverture(self):
        """Test description for single switch opening."""
        switches = {'VL1': {'SWITCH_1': True}}
        
        result = get_all_switch_descriptions(switches)
        
        assert "Ouverture SWITCH_1" in result
        assert "VL1" in result
    
    def test_single_switch_fermeture(self):
        """Test description for single switch closing."""
        switches = {'VL1': {'SWITCH_1': False}}
        
        result = get_all_switch_descriptions(switches)
        
        assert "Fermeture SWITCH_1" in result
        assert "VL1" in result
    
    def test_multiple_switches_same_vl(self):
        """Test description for multiple switches in same voltage level."""
        switches = {'VL1': {'SWITCH_1': True, 'SWITCH_2': False}}
        
        result = get_all_switch_descriptions(switches)
        
        assert "Ouverture SWITCH_1" in result
        assert "Fermeture SWITCH_2" in result
        assert " et " in result
        assert "VL1" in result
    
    def test_empty_switches(self):
        """Test description for empty switches dictionary."""
        switches = {}
        
        result = get_all_switch_descriptions(switches)
        
        assert result == ""
    
    def test_empty_voltage_level(self):
        """Test description for voltage level with no switches."""
        switches = {'VL1': {}}
        
        result = get_all_switch_descriptions(switches)
        
        assert result == ""


# =============================================================================
# Tests for convert_repas_actions_to_grid2op_actions
# =============================================================================

class TestConvertRepasActionsToGrid2opActions:
    """Tests for the high-level conversion function."""
    
    def test_empty_actions_list(self, mock_network_with_topology):
        """Test conversion with empty actions list."""
        result = convert_repas_actions_to_grid2op_actions(
            mock_network_with_topology, []
        )
        
        assert result == {}
    
    def test_single_action_conversion(self, mock_network_with_topology):
        """Test conversion of single action."""
        action = MockRepasAction(
            action_id="TEST",
            switches_by_voltage_level={'VL1': {'VL1_SW1': True}}
        )
        
        result = convert_repas_actions_to_grid2op_actions(
            mock_network_with_topology, [action], use_batch=False, use_analytical=True
        )
        
        assert len(result) == 1
        assert 'TEST_VL1' in result
        assert 'description_unitaire' in result['TEST_VL1']
        assert 'VoltageLevelId' in result['TEST_VL1']
    
    def test_adds_voltage_level_id(self, mock_network_with_topology):
        """Test that VoltageLevelId is added for single VL actions."""
        action = MockRepasAction(
            action_id="TEST",
            switches_by_voltage_level={'VL1': {'VL1_SW1': True}}
        )
        
        result = convert_repas_actions_to_grid2op_actions(
            mock_network_with_topology, [action], use_analytical=True
        )
        
        assert result['TEST_VL1']['VoltageLevelId'] == 'VL1'
    
    def test_batch_mode(self, mock_network_with_topology):
        """Test batch mode conversion."""
        actions = [
            MockRepasAction(
                action_id=f"ACTION_{i}",
                switches_by_voltage_level={'VL1': {'VL1_SW1': True}}
            )
            for i in range(3)
        ]
        
        result = convert_repas_actions_to_grid2op_actions(
            mock_network_with_topology, actions, use_batch=True, use_analytical=True
        )
        
        assert len(result) == 3
    
    def test_non_batch_mode(self, mock_network_with_topology):
        """Test non-batch mode conversion."""
        actions = [
            MockRepasAction(
                action_id=f"ACTION_{i}",
                switches_by_voltage_level={'VL1': {'VL1_SW1': True}}
            )
            for i in range(3)
        ]
        
        result = convert_repas_actions_to_grid2op_actions(
            mock_network_with_topology, actions, use_batch=False, use_analytical=True
        )
        
        assert len(result) == 3


# =============================================================================
# Tests for make_description_unitaire (from action_rebuilder)
# =============================================================================

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


# =============================================================================
# Tests for make_raw_all_actions_dict (from action_rebuilder)
# =============================================================================

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


# =============================================================================
# Tests for build_action_dict_for_snapshot_from_scratch
# =============================================================================

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
        assert actions_to_convert[0]._id == "ACTION-COUPL"
    
    @patch('expert_op4grid_recommender.utils.action_rebuilder.convert_repas_actions_to_grid2op_actions')
    def test_filters_tro_actions(self, mock_convert, mock_network, tro_action, regular_action):
        """Test that TRO actions are also sent for conversion."""
        mock_convert.return_value = {"converted_action": {}}
        all_actions = [tro_action, regular_action]
        
        build_action_dict_for_snapshot_from_scratch(mock_network, all_actions)
        
        call_args = mock_convert.call_args[0]
        actions_to_convert = call_args[1]
        assert len(actions_to_convert) == 1
        assert actions_to_convert[0]._id == "ACTION-TRO"
    
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


# =============================================================================
# Tests for rebuild_action_dict_for_snapshot
# =============================================================================

class TestRebuildActionDictForSnapshot:
    """Tests for the rebuild_action_dict_for_snapshot function."""
    
    @patch('expert_op4grid_recommender.utils.action_rebuilder.convert_repas_actions_to_grid2op_actions')
    def test_keeps_non_coupling_actions(self, mock_convert, mock_network, regular_action):
        """Test that non-COUPL/TRO actions are kept as-is."""
        mock_convert.return_value = {}
        
        dict_action = {
            "ACTION-REG_SUBSTATION-A": {
                "description": "Regular action",
                "description_unitaire": "Regular action description",
                "content": {"set_bus": {}}
            }
        }
        
        result = rebuild_action_dict_for_snapshot(mock_network, [regular_action], dict_action)
        
        assert "ACTION-REG_SUBSTATION-A" in result
        assert result["ACTION-REG_SUBSTATION-A"]["description"] == "Regular action"
    
    @patch('expert_op4grid_recommender.utils.action_rebuilder.convert_repas_actions_to_grid2op_actions')
    def test_converts_coupling_actions(self, mock_convert, mock_network, coupling_action):
        """Test that COUPL actions are sent for conversion."""
        mock_convert.return_value = {"ACTION-COUPL_SUBSTATION-A": {"converted": True}}
        
        dict_action = {
            "ACTION-COUPL_SUBSTATION-A": {
                "description": "COUPL action",
                "description_unitaire": "Ouverture COUPL dans SUBSTATION_A",
                "VoltageLevelId": "SUBSTATION_A",
                "content": {}
            }
        }
        
        result = rebuild_action_dict_for_snapshot(mock_network, [coupling_action], dict_action)
        
        assert mock_convert.called
        assert "ACTION-COUPL_SUBSTATION-A" in result
    
    @patch('expert_op4grid_recommender.utils.action_rebuilder.convert_repas_actions_to_grid2op_actions')
    def test_converts_tro_actions(self, mock_convert, mock_network, tro_action):
        """Test that TRO actions are sent for conversion."""
        mock_convert.return_value = {"ACTION-TRO_SUBSTATION-A": {"converted": True}}
        
        dict_action = {
            "ACTION-TRO_SUBSTATION-A": {
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
            "ACTION-COUPL_SUBSTATION-A": {"converted": True}
        }
        
        dict_action = {
            "ACTION-COUPL_SUBSTATION-A": {
                "description": "COUPL action",
                "description_unitaire": "Custom description unitaire",
                "VoltageLevelId": "SUBSTATION_A",
                "content": {}
            }
        }
        
        result = rebuild_action_dict_for_snapshot(mock_network, [coupling_action], dict_action)
        
        assert result["ACTION-COUPL_SUBSTATION-A"]["description_unitaire"] == "Custom description unitaire"
    
    @patch('expert_op4grid_recommender.utils.action_rebuilder.convert_repas_actions_to_grid2op_actions')
    def test_warning_for_missing_action(self, mock_convert, mock_network, capsys):
        """Test that a warning is printed when action is not found in REPAS actions."""
        mock_convert.return_value = {}
        
        dict_action = {
            "MISSING-ACTION_VL1": {
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


# =============================================================================
# Tests for run_rebuild_actions
# =============================================================================

class TestRunRebuildActions:
    """Tests for the run_rebuild_actions function."""
    
    @patch('expert_op4grid_recommender.utils.action_rebuilder.json.dump')
    @patch('expert_op4grid_recommender.utils.action_rebuilder.build_action_dict_for_snapshot_from_scratch')
    @patch('expert_op4grid_recommender.utils.action_rebuilder.repas.parse_json')
    @patch('expert_op4grid_recommender.utils.action_rebuilder.os.path.join')
    @patch('builtins.open', create=True)
    def test_from_scratch_mode(self, mock_open, mock_path_join, mock_parse, mock_build, mock_json_dump, mock_network, tmp_path):
        """Test that from_scratch mode calls build_action_dict_for_snapshot_from_scratch."""
        mock_parse.return_value = [Mock()]
        mock_build.return_value = {"action": {}}
        mock_path_join.return_value = str(tmp_path / "output.json")
        mock_open.return_value.__enter__ = Mock(return_value=Mock())
        mock_open.return_value.__exit__ = Mock(return_value=False)
        
        run_rebuild_actions(
            mock_network,
            do_from_scratch=True,
            repas_file_path="fake_repas.json"
        )
        
        mock_build.assert_called_once()
    
    @patch('expert_op4grid_recommender.utils.action_rebuilder.json.dump')
    @patch('expert_op4grid_recommender.utils.action_rebuilder.rebuild_action_dict_for_snapshot')
    @patch('expert_op4grid_recommender.utils.action_rebuilder.repas.parse_json')
    @patch('expert_op4grid_recommender.utils.action_rebuilder.os.path.join')
    @patch('builtins.open', create=True)
    def test_rebuild_mode(self, mock_open, mock_path_join, mock_parse, mock_rebuild, mock_json_dump, mock_network, tmp_path):
        """Test that non-from_scratch mode calls rebuild_action_dict_for_snapshot."""
        mock_parse.return_value = [Mock()]
        mock_rebuild.return_value = {"action": {}}
        mock_path_join.return_value = str(tmp_path / "output.json")
        mock_open.return_value.__enter__ = Mock(return_value=Mock())
        mock_open.return_value.__exit__ = Mock(return_value=False)
        
        run_rebuild_actions(
            mock_network,
            do_from_scratch=False,
            repas_file_path="fake_repas.json",
            dict_action_to_filter_on={"existing": {}}
        )
        
        mock_rebuild.assert_called_once()
    
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


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.slow
class TestConversionIntegration:
    """Integration tests for the full conversion pipeline."""
    
    def test_full_analytical_conversion_pipeline(self, mock_network_with_topology):
        """Test the full analytical conversion pipeline."""
        # Create a realistic action
        action = MockRepasAction(
            action_id="INTEGRATION_TEST",
            switches_by_voltage_level={
                'VL1': {'VL1_SW2': False, 'VL1_SW4': False}  # Close both open switches
            },
            description="Integration test action"
        )
        
        # Convert using analytical method
        result = convert_repas_actions_to_grid2op_actions(
            mock_network_with_topology, 
            [action],
            use_batch=True,
            use_analytical=True
        )
        
        # Verify structure
        assert len(result) == 1
        action_key = 'INTEGRATION_TEST_VL1'
        assert action_key in result
        
        converted = result[action_key]
        assert 'description' in converted
        assert 'description_unitaire' in converted
        assert 'VoltageLevelId' in converted
        assert 'content' in converted
        assert 'set_bus' in converted['content']
        
        set_bus = converted['content']['set_bus']
        assert 'lines_or_id' in set_bus
        assert 'loads_id' in set_bus
        assert 'generators_id' in set_bus
    
    def test_bus_numbers_are_sequential(self, mock_network_with_topology):
        """Test that resulting bus numbers are sequential starting at 1."""
        action = MockRepasAction(
            action_id="SEQ_TEST",
            switches_by_voltage_level={
                'VL1': {'VL1_SW2': True}  # Open SW2
            }
        )
        
        result = convert_repas_actions_to_grid2op_actions(
            mock_network_with_topology,
            [action],
            use_analytical=True
        )
        
        set_bus = result['SEQ_TEST_VL1']['content']['set_bus']
        
        # Collect all bus numbers (excluding -1)
        all_buses = set()
        for element_dict in set_bus.values():
            for bus in element_dict.values():
                if bus > 0:
                    all_buses.add(bus)
        
        if all_buses:
            # Bus numbers should be sequential starting at 1
            expected = set(range(1, len(all_buses) + 1))
            assert all_buses == expected, f"Bus numbers {all_buses} are not sequential from 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
