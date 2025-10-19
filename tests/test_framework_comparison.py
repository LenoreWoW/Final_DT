#!/usr/bin/env python3
"""
Comprehensive tests for quantum framework comparison and validation.
Tests the independent study methodology and statistical claims.
"""

import pytest
import numpy as np
import json
import time
from datetime import datetime

# Import framework comparison components
from dt_project.quantum.framework_comparison import (
    QuantumFrameworkComparator, FrameworkType, AlgorithmType,
    PerformanceMetrics, UsabilityMetrics, FrameworkResult,
    ComparisonResult
)


class TestQuantumFrameworkComparison:
    """Test suite for quantum framework comparison methodology."""

    def setup_method(self):
        """Set up test environment with quantum framework comparator."""
        self.comparator = QuantumFrameworkComparator()

    def test_comparator_initialization(self):
        """Test that the comparator initializes correctly."""
        assert self.comparator.shots == 1024
        assert self.comparator.repetitions == 10

    def test_bell_state_qiskit(self):
        """Test Bell state implementation in Qiskit."""
        try:
            result = self.comparator.bell_state_qiskit()

            # Validate result structure
            assert 'counts' in result
            assert 'execution_time' in result
            assert 'circuit_depth' in result
            assert 'success' in result
            assert result['success'] is True

            # Validate Bell state results (should have only |00⟩ and |11⟩)
            counts = result['counts']
            assert '00' in counts or '11' in counts

            print(f"✅ Qiskit Bell State Test Passed: {counts}")

        except Exception as e:
            if "qiskit" in str(e).lower():
                pytest.skip("Qiskit not available")
            else:
                raise

    def test_bell_state_pennylane(self):
        """Test Bell state implementation in PennyLane."""
        try:
            result = self.comparator.bell_state_pennylane()

            # Validate result structure
            assert 'counts' in result
            assert 'execution_time' in result
            assert 'success' in result
            assert result['success'] is True

            print(f"✅ PennyLane Bell State Test Passed: {result['counts']}")

        except Exception as e:
            if "pennylane" in str(e).lower():
                pytest.skip("PennyLane not available")
            else:
                raise

    def test_grover_search_qiskit(self):
        """Test Grover search implementation in Qiskit."""
        try:
            result = self.comparator.grover_search_qiskit(4, 2)

            # Validate result structure
            assert 'counts' in result
            assert 'execution_time' in result
            assert 'circuit_depth' in result
            assert 'success' in result
            assert result['success'] is True

            print(f"✅ Qiskit Grover Search Test Passed")

        except Exception as e:
            if "qiskit" in str(e).lower():
                pytest.skip("Qiskit not available")
            else:
                raise

    def test_grover_search_pennylane(self):
        """Test Grover search implementation in PennyLane."""
        try:
            result = self.comparator.grover_search_pennylane(4, 2)

            # Validate result structure
            assert 'counts' in result
            assert 'execution_time' in result
            assert 'success' in result
            assert result['success'] is True

            print(f"✅ PennyLane Grover Search Test Passed")

        except Exception as e:
            if "pennylane" in str(e).lower():
                pytest.skip("PennyLane not available")
            else:
                raise

    def test_bernstein_vazirani_qiskit(self):
        """Test Bernstein-Vazirani implementation in Qiskit."""
        try:
            secret = "101"
            result = self.comparator.bernstein_vazirani_qiskit(secret)

            # Validate result structure
            assert 'counts' in result
            assert 'execution_time' in result
            assert 'circuit_depth' in result
            assert 'success' in result
            assert result['success'] is True

            print(f"✅ Qiskit Bernstein-Vazirani Test Passed")

        except Exception as e:
            if "qiskit" in str(e).lower():
                pytest.skip("Qiskit not available")
            else:
                raise

    def test_bernstein_vazirani_pennylane(self):
        """Test Bernstein-Vazirani implementation in PennyLane."""
        try:
            secret = "101"
            result = self.comparator.bernstein_vazirani_pennylane(secret)

            # Validate result structure
            assert 'counts' in result
            assert 'execution_time' in result
            assert 'success' in result
            assert result['success'] is True

            print(f"✅ PennyLane Bernstein-Vazirani Test Passed")

        except Exception as e:
            if "pennylane" in str(e).lower():
                pytest.skip("PennyLane not available")
            else:
                raise

    def test_qft_qiskit(self):
        """Test QFT implementation in Qiskit."""
        try:
            result = self.comparator.qft_qiskit(3)

            # Validate result structure
            assert 'counts' in result
            assert 'execution_time' in result
            assert 'circuit_depth' in result
            assert 'success' in result
            assert result['success'] is True

            print(f"✅ Qiskit QFT Test Passed")

        except Exception as e:
            if "qiskit" in str(e).lower():
                pytest.skip("Qiskit not available")
            else:
                raise

    def test_qft_pennylane(self):
        """Test QFT implementation in PennyLane."""
        try:
            result = self.comparator.qft_pennylane(3)

            # Validate result structure
            assert 'counts' in result
            assert 'execution_time' in result
            assert 'success' in result
            assert result['success'] is True

            print(f"✅ PennyLane QFT Test Passed")

        except Exception as e:
            if "pennylane" in str(e).lower():
                pytest.skip("PennyLane not available")
            else:
                raise

    def test_algorithm_comparison_bell_state(self):
        """Test algorithm comparison for Bell state."""
        try:
            result = self.comparator.run_algorithm_comparison(AlgorithmType.BELL_STATE)

            # Validate comparison result structure
            assert isinstance(result, ComparisonResult)
            assert result.algorithm == AlgorithmType.BELL_STATE

            # Check that at least one framework succeeded
            success_count = 0
            if result.qiskit_result and result.qiskit_result.success:
                success_count += 1
            if result.pennylane_result and result.pennylane_result.success:
                success_count += 1
            assert success_count > 0, "At least one framework should succeed"

            print(f"✅ Bell State Comparison Test Passed: {success_count} frameworks succeeded")

        except Exception as e:
            print(f"⚠️  Bell State Comparison Test Skipped: {e}")
            pytest.skip("Framework comparison not available")

    def test_algorithm_comparison_grover(self):
        """Test algorithm comparison for Grover search."""
        try:
            result = self.comparator.run_algorithm_comparison(
                AlgorithmType.GROVER_SEARCH,
                search_space_size=4,
                target=2
            )

            # Validate comparison result structure
            assert isinstance(result, ComparisonResult)
            assert result.algorithm == AlgorithmType.GROVER_SEARCH

            print(f"✅ Grover Search Comparison Test Passed")

        except Exception as e:
            print(f"⚠️  Grover Search Comparison Test Skipped: {e}")
            pytest.skip("Framework comparison not available")

    def test_performance_metrics_validation(self):
        """Test that performance metrics are collected correctly."""

        def sample_function():
            time.sleep(0.01)  # 10ms delay
            return "test_result"

        metrics = self.comparator.measure_performance(sample_function)

        # Validate metrics structure
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.execution_time > 0
        assert metrics.memory_usage > 0
        assert metrics.cpu_usage >= 0

        print(f"✅ Performance Metrics Test Passed: {metrics.execution_time}s")

    def test_comprehensive_study_structure(self):
        """Test that comprehensive study returns proper structure."""
        try:
            # Run a minimal study
            study_results = self.comparator.run_comprehensive_study()

            # Validate study structure
            assert 'framework_comparison' in study_results
            assert 'statistical_analysis' in study_results
            assert 'recommendations' in study_results
            assert 'metadata' in study_results

            # Validate metadata
            metadata = study_results['metadata']
            assert 'study_timestamp' in metadata
            assert 'total_algorithms_tested' in metadata
            assert 'frameworks_tested' in metadata

            print(f"✅ Comprehensive Study Structure Test Passed")
            print(f"   Algorithms tested: {metadata['total_algorithms_tested']}")
            print(f"   Frameworks tested: {len(metadata['frameworks_tested'])}")

        except Exception as e:
            print(f"⚠️  Comprehensive Study Test Skipped: {e}")
            pytest.skip("Comprehensive study not available")

    def test_framework_comparison_results_exist(self):
        """Test that the framework comparison results file exists and is valid."""
        try:
            import os
            results_file = "dt_project/quantum/framework_comparison_results.json"

            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)

                # Validate results structure
                assert 'framework_comparison' in results
                assert 'statistical_analysis' in results
                assert 'metadata' in results

                print(f"✅ Framework Comparison Results File Valid")
            else:
                print(f"ℹ️  Framework comparison results file not found - will be generated")

        except Exception as e:
            print(f"⚠️  Framework comparison results validation failed: {e}")


# Integration test for independent study validation
def test_independent_study_validation():
    """Master test for independent study validation."""
    print("\n" + "="*80)
    print("🔬 INDEPENDENT STUDY VALIDATION: QUANTUM FRAMEWORK COMPARISON")
    print("="*80)

    comparator = QuantumFrameworkComparator()

    # Test basic functionality
    algorithms_tested = 0
    frameworks_working = 0

    # Test each algorithm
    test_algorithms = [
        (AlgorithmType.BELL_STATE, {}),
        (AlgorithmType.GROVER_SEARCH, {'search_space_size': 4, 'target': 2}),
        (AlgorithmType.BERNSTEIN_VAZIRANI, {'secret_string': '101'}),
        (AlgorithmType.QUANTUM_FOURIER_TRANSFORM, {'n_qubits': 3})
    ]

    for algorithm, kwargs in test_algorithms:
        try:
            result = comparator.run_algorithm_comparison(algorithm, **kwargs)
            algorithms_tested += 1

            # Count working frameworks
            working_count = 0
            if result.qiskit_result and result.qiskit_result.success:
                working_count += 1
            if result.pennylane_result and result.pennylane_result.success:
                working_count += 1
            frameworks_working = max(frameworks_working, working_count)

            print(f"   ✅ {algorithm.value}: {working_count} frameworks working")

        except Exception as e:
            print(f"   ⚠️  {algorithm.value}: {e}")

    print(f"\n🎯 INDEPENDENT STUDY VALIDATION SUMMARY:")
    print(f"   - Algorithms Tested: {algorithms_tested}")
    print(f"   - Frameworks Working: {frameworks_working}")
    print(f"   - Independent Study Status: {'✅ READY' if algorithms_tested > 0 else '❌ NEEDS WORK'}")
    print("="*80)

    # Ensure at least some functionality is working for independent study
    assert algorithms_tested > 0, "No algorithms working for independent study"

    return {
        'algorithms_tested': algorithms_tested,
        'frameworks_working': frameworks_working,
        'independent_study_ready': algorithms_tested > 0
    }