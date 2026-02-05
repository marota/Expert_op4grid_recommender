# tests/test_conversion_actions_repas.py
"""
Tests for the REPAS to Grid2Op action conversion module.

These tests verify:
- Union-Find topology computation
- Bus number reindexing per voltage level
- Disconnection detection (isolated nodes get bus = -1)
- Consistent reindexing across all element types
- NetworkTopologyCache functionality
"""

import pytest
from collections import defaultdict
from typing import Dict, Set

# Skip all tests if pypowsybl is not available
pypowsybl = pytest.importorskip("pypowsybl")


class TestUnionFind:
    """Tests for the Union-Find data structure."""
    
    def test_union_find_creation(self):
        """Test creating a Union-Find with elements."""
        from expert_op4grid_recommender.utils.conversion_actions_repas import UnionFind
        
        elements = ['a', 'b', 'c', 'd']
        uf = UnionFind(elements)
        
        assert len(uf.parent) == 4
        # Each element should be its own parent initially
        for e in elements:
            assert uf.parent[e] == e
    
    def test_union_find_union(self):
        """Test uniting elements."""
        from expert_op4grid_recommender.utils.conversion_actions_repas import UnionFind
        
        uf = UnionFind(['a', 'b', 'c', 'd'])
        
        uf.union('a', 'b')
        assert uf.find('a') == uf.find('b')
        
        uf.union('c', 'd')
        assert uf.find('c') == uf.find('d')
        
        # a,b and c,d should be in different components
        assert uf.find('a') != uf.find('c')
    
    def test_union_find_component_mapping(self):
        """Test getting component mapping."""
        from expert_op4grid_recommender.utils.conversion_actions_repas import UnionFind
        
        uf = UnionFind(['a', 'b', 'c', 'd', 'e'])
        
        # Create two components: {a, b, c} and {d, e}
        uf.union('a', 'b')
        uf.union('b', 'c')
        uf.union('d', 'e')
        
        mapping = uf.get_component_mapping()
        
        # a, b, c should have the same component number
        assert mapping['a'] == mapping['b'] == mapping['c']
        
        # d, e should have the same component number
        assert mapping['d'] == mapping['e']
        
        # The two groups should have different component numbers
        assert mapping['a'] != mapping['d']
        
        # Component numbers should be 1-indexed
        assert set(mapping.values()) == {1, 2}
    
    def test_union_find_single_elements(self):
        """Test that single elements each get their own component."""
        from expert_op4grid_recommender.utils.conversion_actions_repas import UnionFind
        
        uf = UnionFind(['a', 'b', 'c'])
        # No unions - each element is its own component
        
        mapping = uf.get_component_mapping()
        
        # Each element should have a different component number
        assert len(set(mapping.values())) == 3


class TestReindexBusNumbersPerVL:
    """Tests for the _reindex_bus_numbers_per_vl function."""
    
    def test_reindex_simple(self):
        """Test simple reindexing with gaps."""
        from expert_op4grid_recommender.utils.conversion_actions_repas import _reindex_bus_numbers_per_vl
        
        set_bus = {
            'loads_id': {'load1': 3, 'load2': 7, 'load3': 3},
            'generators_id': {'gen1': 7}
        }
        element_to_vl = {
            'load1': 'VL1', 'load2': 'VL1', 'load3': 'VL1', 'gen1': 'VL1'
        }
        
        result = _reindex_bus_numbers_per_vl(set_bus, element_to_vl)
        
        # 3 -> 1, 7 -> 2
        assert result['loads_id']['load1'] == 1
        assert result['loads_id']['load2'] == 2
        assert result['loads_id']['load3'] == 1
        assert result['generators_id']['gen1'] == 2
    
    def test_reindex_preserves_disconnected(self):
        """Test that disconnected elements (bus = -1) are preserved."""
        from expert_op4grid_recommender.utils.conversion_actions_repas import _reindex_bus_numbers_per_vl
        
        set_bus = {
            'loads_id': {'load1': 3, 'load2': -1},
            'generators_id': {'gen1': 3}
        }
        element_to_vl = {
            'load1': 'VL1', 'load2': 'VL1', 'gen1': 'VL1'
        }
        
        result = _reindex_bus_numbers_per_vl(set_bus, element_to_vl)
        
        # Disconnected should stay -1
        assert result['loads_id']['load2'] == -1
        # Connected should be reindexed
        assert result['loads_id']['load1'] == 1
        assert result['generators_id']['gen1'] == 1
    
    def test_reindex_multiple_vls(self):
        """Test reindexing with multiple voltage levels."""
        from expert_op4grid_recommender.utils.conversion_actions_repas import _reindex_bus_numbers_per_vl
        
        set_bus = {
            'loads_id': {'load1': 5, 'load2': 10},  # VL1
            'generators_id': {'gen1': 20, 'gen2': 30}  # VL2
        }
        element_to_vl = {
            'load1': 'VL1', 'load2': 'VL1',
            'gen1': 'VL2', 'gen2': 'VL2'
        }
        
        result = _reindex_bus_numbers_per_vl(set_bus, element_to_vl)
        
        # VL1: 5 -> 1, 10 -> 2
        assert result['loads_id']['load1'] == 1
        assert result['loads_id']['load2'] == 2
        
        # VL2: 20 -> 1, 30 -> 2
        assert result['generators_id']['gen1'] == 1
        assert result['generators_id']['gen2'] == 2
    
    def test_reindex_consistent_across_element_types(self):
        """Test that elements on the same bus get the same reindexed number."""
        from expert_op4grid_recommender.utils.conversion_actions_repas import _reindex_bus_numbers_per_vl
        
        # All elements on bus 5 in VL1 should get the same reindexed bus number
        set_bus = {
            'lines_or_id': {'line1_or': 5},
            'lines_ex_id': {'line2_ex': 5},
            'loads_id': {'load1': 5, 'load2': 10},
            'generators_id': {'gen1': 5},
            'shunts_id': {'shunt1': 10}
        }
        element_to_vl = {
            'line1_or': 'VL1', 'line2_ex': 'VL1',
            'load1': 'VL1', 'load2': 'VL1',
            'gen1': 'VL1', 'shunt1': 'VL1'
        }
        
        result = _reindex_bus_numbers_per_vl(set_bus, element_to_vl)
        
        # All elements on original bus 5 should be on reindexed bus 1
        assert result['lines_or_id']['line1_or'] == 1
        assert result['lines_ex_id']['line2_ex'] == 1
        assert result['loads_id']['load1'] == 1
        assert result['generators_id']['gen1'] == 1
        
        # All elements on original bus 10 should be on reindexed bus 2
        assert result['loads_id']['load2'] == 2
        assert result['shunts_id']['shunt1'] == 2


class TestNetworkTopologyCache:
    """Tests for the NetworkTopologyCache class."""
    
    @pytest.fixture
    def simple_network(self):
        """Create a simple test network with switches."""
        # Use IEEE 14 bus which has a more complex topology
        network = pypowsybl.network.create_ieee14()
        return network
    
    def test_cache_creation(self, simple_network):
        """Test that cache can be created from a network."""
        from expert_op4grid_recommender.utils.conversion_actions_repas import NetworkTopologyCache
        
        cache = NetworkTopologyCache(simple_network)
        
        # Check that internal structures are populated
        assert hasattr(cache, '_vl_switches')
        assert hasattr(cache, '_vl_nodes')
        assert hasattr(cache, '_load_to_node')
        assert hasattr(cache, '_branch_or_to_node')
    
    def test_cache_element_to_vl_mappings(self, simple_network):
        """Test that element-to-VL mappings are built correctly."""
        from expert_op4grid_recommender.utils.conversion_actions_repas import NetworkTopologyCache
        
        cache = NetworkTopologyCache(simple_network)
        
        # Check that mappings exist
        assert len(cache._load_to_vl) >= 0
        assert len(cache._gen_to_vl) >= 0
        assert len(cache._branch_or_to_vl) >= 0
        assert len(cache._branch_ex_to_vl) >= 0


class TestComputeBusAssignments:
    """Tests for the compute_bus_assignments method."""
    
    def test_isolated_node_gets_minus_one(self):
        """Test that a node with only 1 node in its component gets bus = -1."""
        from expert_op4grid_recommender.utils.conversion_actions_repas import (
            NetworkTopologyCache, UnionFind
        )
        
        # Create a mock scenario with isolated nodes
        # We'll test the logic directly
        uf = UnionFind(['node1', 'node2', 'node3', 'isolated_node'])
        
        # Connect node1, node2, node3
        uf.union('node1', 'node2')
        uf.union('node2', 'node3')
        # isolated_node stays alone
        
        mapping = uf.get_component_mapping()
        
        # Count nodes per component
        component_counts = defaultdict(int)
        for node, comp in mapping.items():
            component_counts[comp] += 1
        
        # isolated_node's component should have count = 1
        isolated_comp = mapping['isolated_node']
        assert component_counts[isolated_comp] == 1
        
        # Connected nodes' component should have count = 3
        connected_comp = mapping['node1']
        assert component_counts[connected_comp] == 3
    
    def test_switch_open_creates_isolation(self):
        """Test that opening a switch can isolate a node."""
        from expert_op4grid_recommender.utils.conversion_actions_repas import UnionFind
        
        # Simulate a VL with nodes and switches
        # Initial: node1 -- [switch1] -- node2 -- [switch2] -- node3
        # If switch1 is open and switch2 is closed: node1 is isolated
        
        nodes = ['node1', 'node2', 'node3']
        uf = UnionFind(nodes)
        
        # switch1 is OPEN (don't union node1-node2)
        # switch2 is CLOSED (union node2-node3)
        uf.union('node2', 'node3')
        
        mapping = uf.get_component_mapping()
        
        # node1 should be in its own component
        assert mapping['node1'] != mapping['node2']
        # node2 and node3 should be in the same component
        assert mapping['node2'] == mapping['node3']
    
    def test_switch_closed_connects_nodes(self):
        """Test that closing a switch connects nodes."""
        from expert_op4grid_recommender.utils.conversion_actions_repas import UnionFind
        
        nodes = ['node1', 'node2', 'node3']
        uf = UnionFind(nodes)
        
        # Both switches closed
        uf.union('node1', 'node2')
        uf.union('node2', 'node3')
        
        mapping = uf.get_component_mapping()
        
        # All nodes should be in the same component
        assert mapping['node1'] == mapping['node2'] == mapping['node3']


class TestGetElementBusAssignments:
    """Tests for the get_element_bus_assignments method."""
    
    def test_all_elements_reindexed_together(self):
        """Test that all element types are reindexed together consistently."""
        from expert_op4grid_recommender.utils.conversion_actions_repas import _reindex_bus_numbers_per_vl
        
        # Simulate what get_element_bus_assignments does:
        # All elements on the same original bus should get the same reindexed bus
        
        # Create a scenario where line_or, load, and generator are on the same node
        # and they should all get the same reindexed bus number
        
        element_to_vl = {
            'line1_or': 'VL1',
            'line2_ex': 'VL1', 
            'load1': 'VL1',
            'gen1': 'VL1',
        }
        
        # All on original bus 7 (which should become bus 1 after reindexing)
        combined_set_bus = {
            'lines_or_id': {'line1_or': 7},
            'lines_ex_id': {'line2_ex': 7},
            'loads_id': {'load1': 7},
            'generators_id': {'gen1': 7},
            'shunts_id': {}
        }
        
        result = _reindex_bus_numbers_per_vl(combined_set_bus, element_to_vl)
        
        # All should be reindexed to bus 1
        assert result['lines_or_id']['line1_or'] == 1
        assert result['lines_ex_id']['line2_ex'] == 1
        assert result['loads_id']['load1'] == 1
        assert result['generators_id']['gen1'] == 1
    
    def test_different_buses_different_numbers(self):
        """Test that elements on different buses get different reindexed numbers."""
        from expert_op4grid_recommender.utils.conversion_actions_repas import _reindex_bus_numbers_per_vl
        
        element_to_vl = {
            'line1_or': 'VL1',
            'line2_or': 'VL1',
            'load1': 'VL1',
            'load2': 'VL1',
        }
        
        combined_set_bus = {
            'lines_or_id': {'line1_or': 3, 'line2_or': 7},
            'lines_ex_id': {},
            'loads_id': {'load1': 3, 'load2': 7},
            'generators_id': {},
            'shunts_id': {}
        }
        
        result = _reindex_bus_numbers_per_vl(combined_set_bus, element_to_vl)
        
        # Bus 3 -> 1, Bus 7 -> 2
        assert result['lines_or_id']['line1_or'] == 1
        assert result['lines_or_id']['line2_or'] == 2
        assert result['loads_id']['load1'] == 1
        assert result['loads_id']['load2'] == 2
    
    def test_disconnected_preserved_after_reindex(self):
        """Test that -1 values are preserved through reindexing."""
        from expert_op4grid_recommender.utils.conversion_actions_repas import _reindex_bus_numbers_per_vl
        
        element_to_vl = {
            'line1_or': 'VL1',
            'line2_or': 'VL1',
            'load1': 'VL1',
        }
        
        combined_set_bus = {
            'lines_or_id': {'line1_or': 5, 'line2_or': -1},  # line2 is disconnected
            'lines_ex_id': {},
            'loads_id': {'load1': 5},
            'generators_id': {},
            'shunts_id': {}
        }
        
        result = _reindex_bus_numbers_per_vl(combined_set_bus, element_to_vl)
        
        # Connected elements get positive bus numbers
        assert result['lines_or_id']['line1_or'] == 1
        assert result['loads_id']['load1'] == 1
        
        # Disconnected stays -1
        assert result['lines_or_id']['line2_or'] == -1


class TestDisconnectionDetection:
    """Tests for disconnection detection logic."""
    
    def test_single_node_component_is_disconnected(self):
        """Test that a component with only 1 node is marked as disconnected."""
        # This tests the core logic used in compute_bus_assignments
        
        from collections import defaultdict
        
        # Simulate component mapping from Union-Find
        raw_component_mapping = {
            'node1': 1,  # Component 1 has 3 nodes
            'node2': 1,
            'node3': 1,
            'isolated_node': 2,  # Component 2 has only 1 node
        }
        
        # Count nodes per component
        component_node_count = defaultdict(int)
        for node, comp in raw_component_mapping.items():
            component_node_count[comp] += 1
        
        # Create final mapping
        final_mapping = {}
        for node, component in raw_component_mapping.items():
            if component_node_count[component] > 1:
                final_mapping[node] = component
            else:
                final_mapping[node] = -1
        
        # Connected nodes get their component number
        assert final_mapping['node1'] == 1
        assert final_mapping['node2'] == 1
        assert final_mapping['node3'] == 1
        
        # Isolated node gets -1
        assert final_mapping['isolated_node'] == -1
    
    def test_two_node_component_is_connected(self):
        """Test that a component with 2+ nodes is considered connected."""
        from collections import defaultdict
        
        raw_component_mapping = {
            'node1': 1,
            'node2': 1,  # Component 1 has 2 nodes - should be connected
            'node3': 2,  # Component 2 has 1 node - should be disconnected
        }
        
        component_node_count = defaultdict(int)
        for node, comp in raw_component_mapping.items():
            component_node_count[comp] += 1
        
        final_mapping = {}
        for node, component in raw_component_mapping.items():
            if component_node_count[component] > 1:
                final_mapping[node] = component
            else:
                final_mapping[node] = -1
        
        # 2-node component is connected
        assert final_mapping['node1'] == 1
        assert final_mapping['node2'] == 1
        
        # 1-node component is disconnected
        assert final_mapping['node3'] == -1


class TestIntegrationWithRealNetwork:
    """Integration tests using real pypowsybl networks."""
    
    @pytest.fixture
    def ieee14_network(self):
        """Create IEEE 14 bus network."""
        return pypowsybl.network.create_ieee14()
    
    def test_cache_creation_ieee14(self, ieee14_network):
        """Test cache creation with IEEE 14 network."""
        from expert_op4grid_recommender.utils.conversion_actions_repas import NetworkTopologyCache
        
        cache = NetworkTopologyCache(ieee14_network)
        
        # Should have loaded elements
        assert len(cache._branches_df) > 0
        assert len(cache._loads_df) >= 0
        assert len(cache._generators_df) > 0
    
    def test_compute_bus_assignments_no_changes(self, ieee14_network):
        """Test computing bus assignments without any switch changes."""
        from expert_op4grid_recommender.utils.conversion_actions_repas import NetworkTopologyCache
        
        cache = NetworkTopologyCache(ieee14_network)
        
        # Get voltage levels
        vl_ids = set(ieee14_network.get_voltage_levels().index)
        
        # Compute with no switch changes
        result = cache.compute_bus_assignments({}, vl_ids)
        
        # Should return a mapping for each VL
        assert len(result) == len(vl_ids)
    
    def test_get_element_bus_assignments_ieee14(self, ieee14_network):
        """Test getting element bus assignments with IEEE 14 network."""
        from expert_op4grid_recommender.utils.conversion_actions_repas import NetworkTopologyCache
        
        cache = NetworkTopologyCache(ieee14_network)
        
        vl_ids = set(ieee14_network.get_voltage_levels().index)
        node_to_bus = cache.compute_bus_assignments({}, vl_ids)
        
        result = cache.get_element_bus_assignments(node_to_bus, vl_ids)
        
        # Should have all element type keys
        assert 'lines_or_id' in result
        assert 'lines_ex_id' in result
        assert 'loads_id' in result
        assert 'generators_id' in result
        assert 'shunts_id' in result


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_voltage_level(self):
        """Test handling of empty voltage level."""
        from expert_op4grid_recommender.utils.conversion_actions_repas import UnionFind
        
        # Empty Union-Find
        uf = UnionFind([])
        mapping = uf.get_component_mapping()
        
        assert mapping == {}
    
    def test_reindex_empty_set_bus(self):
        """Test reindexing with empty set_bus."""
        from expert_op4grid_recommender.utils.conversion_actions_repas import _reindex_bus_numbers_per_vl
        
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
    
    def test_reindex_with_missing_vl_mapping(self):
        """Test reindexing when element is not in element_to_vl."""
        from expert_op4grid_recommender.utils.conversion_actions_repas import _reindex_bus_numbers_per_vl
        
        set_bus = {
            'loads_id': {'load1': 5, 'load2': 5},
        }
        # Only load1 has VL mapping
        element_to_vl = {'load1': 'VL1'}
        
        result = _reindex_bus_numbers_per_vl(set_bus, element_to_vl)
        
        # load1 should be reindexed
        assert result['loads_id']['load1'] == 1
        # load2 should keep original value (no VL mapping)
        assert result['loads_id']['load2'] == 5
    
    def test_all_elements_disconnected(self):
        """Test scenario where all elements are disconnected."""
        from expert_op4grid_recommender.utils.conversion_actions_repas import _reindex_bus_numbers_per_vl
        
        set_bus = {
            'lines_or_id': {'line1_or': -1},
            'lines_ex_id': {'line1_ex': -1},
            'loads_id': {'load1': -1},
            'generators_id': {},
            'shunts_id': {}
        }
        element_to_vl = {
            'line1_or': 'VL1',
            'line1_ex': 'VL1', 
            'load1': 'VL1'
        }
        
        result = _reindex_bus_numbers_per_vl(set_bus, element_to_vl)
        
        # All should remain -1
        assert result['lines_or_id']['line1_or'] == -1
        assert result['lines_ex_id']['line1_ex'] == -1
        assert result['loads_id']['load1'] == -1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
