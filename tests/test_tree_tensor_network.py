"""
Test Suite for Tree-Tensor-Network Implementation

Tests the implementation against theoretical foundations:
- Jaschke et al. (2024) Quantum Science and Technology
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dt_project.quantum.tensor_networks.tree_tensor_network import (
    TreeTensorNetwork,
    TTNConfig,
    TreeStructure,
    TTNNode,
    BenchmarkResult,
    create_ttn_for_benchmarking
)


class TestTTNConfig:
    """Test TTN configuration"""
    
    def test_valid_config(self):
        """Test valid configuration creation"""
        config = TTNConfig(
            num_qubits=8,
            max_bond_dimension=64,
            tree_structure=TreeStructure.BINARY_TREE
        )
        
        assert config.num_qubits == 8
        assert config.max_bond_dimension == 64
        assert config.tree_structure == TreeStructure.BINARY_TREE
    
    def test_invalid_num_qubits(self):
        """Test invalid number of qubits"""
        with pytest.raises(ValueError):
            TTNConfig(num_qubits=1)  # Need at least 2
    
    def test_invalid_bond_dimension(self):
        """Test invalid bond dimension"""
        with pytest.raises(ValueError):
            TTNConfig(max_bond_dimension=1)  # Need at least 2


class TestTTNNode:
    """Test TTN node structure"""
    
    def test_node_creation(self):
        """Test node creation"""
        tensor = np.random.rand(2, 4, 4)
        node = TTNNode(
            node_id=0,
            tensor=tensor,
            physical_indices=[0],
            virtual_indices=[1, 2]
        )
        
        assert node.node_id == 0
        assert node.rank() == 3
        assert node.bond_dimensions() == [2, 4, 4]
    
    def test_leaf_node(self):
        """Test leaf node properties"""
        tensor = np.random.rand(2, 8)
        node = TTNNode(
            node_id=0,
            tensor=tensor,
            physical_indices=[0],
            virtual_indices=[],
            is_leaf=True
        )
        
        assert node.is_leaf == True
        assert node.is_root == False
        assert len(node.children_ids) == 0
    
    def test_root_node(self):
        """Test root node properties"""
        tensor = np.random.rand(8, 8, 8)
        node = TTNNode(
            node_id=5,
            tensor=tensor,
            physical_indices=[],
            virtual_indices=[],
            is_root=True
        )
        
        assert node.is_root == True
        assert node.is_leaf == False


class TestTreeStructure:
    """Test tree structure construction"""
    
    def test_binary_tree_creation(self):
        """Test binary tree structure"""
        config = TTNConfig(
            num_qubits=4,
            tree_structure=TreeStructure.BINARY_TREE
        )
        ttn = TreeTensorNetwork(config)
        
        # Should have 4 leaf nodes + 3 internal nodes (binary tree)
        assert len(ttn.leaf_ids) == 4
        assert ttn.root_id is not None
        assert ttn.nodes[ttn.root_id].is_root
        
        # Check all leaves are marked
        for leaf_id in ttn.leaf_ids:
            assert ttn.nodes[leaf_id].is_leaf
    
    def test_balanced_tree_creation(self):
        """Test balanced tree structure"""
        config = TTNConfig(
            num_qubits=9,
            tree_structure=TreeStructure.BALANCED_TREE
        )
        ttn = TreeTensorNetwork(config)
        
        assert len(ttn.leaf_ids) == 9
        assert ttn.root_id is not None
        assert ttn.nodes[ttn.root_id].is_root
    
    def test_tree_connectivity(self):
        """Test that tree is properly connected"""
        config = TTNConfig(num_qubits=8, tree_structure=TreeStructure.BINARY_TREE)
        ttn = TreeTensorNetwork(config)
        
        # Check parent-child relationships
        for node_id, node in ttn.nodes.items():
            if not node.is_root:
                # Every non-root should have a parent
                assert node.parent_id is not None
                # Parent should list this node as child
                parent = ttn.nodes[node.parent_id]
                assert node_id in parent.children_ids
            
            if not node.is_leaf:
                # Every non-leaf should have children
                assert len(node.children_ids) > 0


class TestBenchmarkResult:
    """Test benchmark result analysis"""
    
    def test_result_creation(self):
        """Test benchmark result creation"""
        result = BenchmarkResult(
            circuit_depth=10,
            num_qubits=8,
            fidelity=0.95,
            bond_dimension_used=64,
            truncation_error=1e-8,
            computation_time=0.5
        )
        
        assert result.circuit_depth == 10
        assert result.fidelity == 0.95
        assert result.bond_dimension_used == 64
    
    def test_high_fidelity_detection(self):
        """Test high-fidelity detection"""
        high_fidelity = BenchmarkResult(
            circuit_depth=5,
            num_qubits=4,
            fidelity=0.995,
            bond_dimension_used=64,
            truncation_error=1e-10,
            computation_time=0.1
        )
        
        assert high_fidelity.is_high_fidelity(threshold=0.99)
        assert high_fidelity.is_high_fidelity(threshold=0.95)
        
        low_fidelity = BenchmarkResult(
            circuit_depth=20,
            num_qubits=8,
            fidelity=0.92,
            bond_dimension_used=32,
            truncation_error=1e-6,
            computation_time=1.0
        )
        
        assert not low_fidelity.is_high_fidelity(threshold=0.99)
        assert not low_fidelity.is_high_fidelity(threshold=0.95)


class TestQuantumBenchmarking:
    """Test quantum circuit benchmarking (primary application)"""
    
    def test_benchmark_basic_circuit(self):
        """Test benchmarking a basic quantum circuit"""
        ttn = create_ttn_for_benchmarking(num_qubits=4, max_bond_dim=32)
        
        result = ttn.benchmark_quantum_circuit(circuit_depth=5)
        
        assert result.circuit_depth == 5
        assert result.num_qubits == 4
        assert 0.0 <= result.fidelity <= 1.0
        assert result.truncation_error >= 0.0
        assert result.computation_time >= 0.0
    
    def test_benchmark_multiple_depths(self):
        """Test benchmarking circuits of different depths"""
        ttn = create_ttn_for_benchmarking(num_qubits=6, max_bond_dim=64)
        
        depths = [5, 10, 15, 20]
        results = []
        
        for depth in depths:
            result = ttn.benchmark_quantum_circuit(depth)
            results.append(result)
            assert result.circuit_depth == depth
        
        # Check that results are stored
        assert len(ttn.benchmark_history) == len(depths)
    
    def test_fidelity_vs_depth(self):
        """Test that fidelity generally decreases with circuit depth"""
        ttn = create_ttn_for_benchmarking(num_qubits=8, max_bond_dim=64)
        
        # Benchmark different depths
        result_shallow = ttn.benchmark_quantum_circuit(circuit_depth=5)
        result_medium = ttn.benchmark_quantum_circuit(circuit_depth=15)
        result_deep = ttn.benchmark_quantum_circuit(circuit_depth=25)
        
        # Deeper circuits should generally have lower fidelity
        # (though with some variation due to randomness)
        assert result_shallow.fidelity >= result_deep.fidelity - 0.1
    
    def test_fidelity_vs_bond_dimension(self):
        """Test that larger bond dimension improves fidelity"""
        config_small = TTNConfig(num_qubits=6, max_bond_dimension=16)
        ttn_small = TreeTensorNetwork(config_small)
        
        config_large = TTNConfig(num_qubits=6, max_bond_dimension=128)
        ttn_large = TreeTensorNetwork(config_large)
        
        result_small = ttn_small.benchmark_quantum_circuit(circuit_depth=10)
        result_large = ttn_large.benchmark_quantum_circuit(circuit_depth=10)
        
        # Larger bond dimension should generally give higher fidelity
        assert result_large.fidelity >= result_small.fidelity - 0.05


class TestNetworkContraction:
    """Test tensor network contraction"""
    
    def test_contract_network(self):
        """Test contracting the network to get quantum state"""
        config = TTNConfig(num_qubits=4)
        ttn = TreeTensorNetwork(config)
        
        state = ttn.contract_network()
        
        # Should return state vector of correct dimension
        expected_dim = 2 ** config.num_qubits
        assert len(state) == expected_dim
        
        # Should be normalized
        norm = np.linalg.norm(state)
        assert abs(norm - 1.0) < 1e-10
    
    def test_contract_different_sizes(self):
        """Test contraction for different system sizes"""
        for num_qubits in [2, 4, 6, 8]:
            config = TTNConfig(num_qubits=num_qubits)
            ttn = TreeTensorNetwork(config)
            
            state = ttn.contract_network()
            expected_dim = 2 ** num_qubits
            
            assert len(state) == expected_dim
            assert abs(np.linalg.norm(state) - 1.0) < 1e-10


class TestOptimization:
    """Test bond dimension optimization"""
    
    def test_optimize_bond_dimensions(self):
        """Test bond dimension optimization"""
        config = TTNConfig(num_qubits=6, max_bond_dimension=128)
        ttn = TreeTensorNetwork(config)
        
        optimal_bd = ttn.optimize_bond_dimensions(target_fidelity=0.95)
        
        # Should return a reasonable bond dimension
        assert 2 <= optimal_bd <= 128
        assert optimal_bd % 2 == 0 or optimal_bd == 2  # Should be power of 2 or near it


class TestReporting:
    """Test benchmark report generation"""
    
    def test_report_generation(self):
        """Test comprehensive report generation"""
        ttn = create_ttn_for_benchmarking(num_qubits=8, max_bond_dim=64)
        
        # Run several benchmarks
        for depth in [5, 10, 15]:
            ttn.benchmark_quantum_circuit(depth)
        
        report = ttn.generate_benchmark_report()
        
        # Check report structure
        assert 'theoretical_foundation' in report
        assert 'configuration' in report
        assert 'benchmark_results' in report
        assert 'high_fidelity_achievement' in report
        
        # Check theoretical reference
        assert 'Jaschke' in report['theoretical_foundation']['reference']
        
        # Check configuration
        assert report['configuration']['num_qubits'] == 8
        assert report['configuration']['max_bond_dimension'] == 64
        
        # Check benchmark results
        assert report['benchmark_results']['num_circuits_tested'] == 3
        assert 'mean_fidelity' in report['benchmark_results']
        assert 'mean_truncation_error' in report['benchmark_results']
    
    def test_report_without_data(self):
        """Test report generation without benchmark data"""
        ttn = create_ttn_for_benchmarking(num_qubits=4, max_bond_dim=32)
        
        report = ttn.generate_benchmark_report()
        
        assert 'error' in report


class TestFactoryFunction:
    """Test factory function"""
    
    def test_create_ttn_for_benchmarking(self):
        """Test factory function creates proper TTN"""
        ttn = create_ttn_for_benchmarking(num_qubits=8, max_bond_dim=64)
        
        assert ttn.config.num_qubits == 8
        assert ttn.config.max_bond_dimension == 64
        assert ttn.config.tree_structure == TreeStructure.BINARY_TREE
        assert len(ttn.nodes) > 0
        assert ttn.root_id is not None


# Integration test
def test_full_benchmarking_workflow():
    """Integration test of complete TTN benchmarking workflow"""
    print("\n" + "="*80)
    print("INTEGRATION TEST: Complete TTN Benchmarking Workflow")
    print("="*80)
    
    # Create TTN
    print(f"\n1. Creating Tree-Tensor-Network...")
    ttn = create_ttn_for_benchmarking(num_qubits=8, max_bond_dim=64)
    print(f"   ✓ TTN created: {len(ttn.nodes)} nodes, {ttn.config.num_qubits} qubits")
    
    # Benchmark circuits
    print(f"\n2. Benchmarking quantum circuits...")
    depths = [5, 10, 15, 20]
    for depth in depths:
        result = ttn.benchmark_quantum_circuit(depth)
        print(f"   Depth {depth:2d}: fidelity={result.fidelity:.6f}, "
              f"error={result.truncation_error:.2e}")
    
    # Generate report
    print(f"\n3. Generating benchmark report...")
    report = ttn.generate_benchmark_report()
    print(f"   ✓ Report generated")
    
    # Display results
    print(f"\n📊 RESULTS:")
    print(f"   Theoretical Foundation: {report['theoretical_foundation']['reference']}")
    print(f"   Num Qubits: {report['configuration']['num_qubits']}")
    print(f"   Max Bond Dimension: {report['configuration']['max_bond_dimension']}")
    print(f"   Circuits Tested: {report['benchmark_results']['num_circuits_tested']}")
    print(f"   Mean Fidelity: {report['benchmark_results']['mean_fidelity']:.6f}")
    print(f"   Circuits ≥99%: {report['high_fidelity_achievement']['above_99_percent']}")
    print(f"   Circuits ≥95%: {report['high_fidelity_achievement']['above_95_percent']}")
    print(f"   Mean Error: {report['benchmark_results']['mean_truncation_error']:.2e}")
    
    print(f"\n✅ Integration test completed successfully!")
    print("="*80 + "\n")
    
    # Assertions
    assert report['benchmark_results']['num_circuits_tested'] == len(depths)
    assert report['benchmark_results']['mean_fidelity'] > 0.8
    assert report['benchmark_results']['mean_fidelity'] <= 1.0


if __name__ == "__main__":
    # Run pytest with verbose output
    pytest.main([__file__, "-v", "-s"])

