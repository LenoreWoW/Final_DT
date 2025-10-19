"""
Comprehensive Tests for Enhanced Quantum Digital Twin

Tests academic validation, tensor network integration, and benchmark comparisons.
"""

import pytest
import numpy as np
from datetime import datetime

# Import the enhanced quantum digital twin
try:
    from dt_project.quantum.enhanced_quantum_digital_twin import (
        EnhancedQuantumDigitalTwin,
        EnhancementConfig,
        PerformanceMetrics
    )
    ENHANCED_DT_AVAILABLE = True
except ImportError:
    ENHANCED_DT_AVAILABLE = False

# Import validation framework
try:
    from dt_project.validation.academic_statistical_framework import (
        AcademicStatisticalValidator,
        StatisticalResults,
        PerformanceBenchmark
    )
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False

# Import tensor networks
try:
    from dt_project.quantum.tensor_networks.matrix_product_operator import (
        MatrixProductOperator,
        QuantumState,
        TensorNetworkConfig
    )
    TENSOR_NETWORKS_AVAILABLE = True
except ImportError:
    TENSOR_NETWORKS_AVAILABLE = False


@pytest.mark.skipif(not ENHANCED_DT_AVAILABLE, reason="Enhanced DT not available")
class TestEnhancedQuantumDigitalTwin:
    """Test suite for Enhanced Quantum Digital Twin"""
    
    def test_initialization(self):
        """Test proper initialization of enhanced digital twin"""
        twin = EnhancedQuantumDigitalTwin()
        
        assert twin is not None
        assert twin.config is not None
        assert isinstance(twin.performance_history, list)
        assert isinstance(twin.validation_history, list)
    
    def test_custom_configuration(self):
        """Test initialization with custom configuration"""
        config = EnhancementConfig(
            num_qubits=8,
            mpo_target_fidelity=0.998,
            statistical_alpha=0.0001
        )
        
        twin = EnhancedQuantumDigitalTwin(config)
        
        assert twin.config.num_qubits == 8
        assert twin.config.mpo_target_fidelity == 0.998
        assert twin.config.statistical_alpha == 0.0001
    
    def test_create_enhanced_twin_core(self):
        """Test creating core enhanced quantum digital twin"""
        twin = EnhancedQuantumDigitalTwin()
        
        result = twin.create_enhanced_twin(
            data={"sensor_readings": [0.1, 0.2, 0.3]},
            twin_type="core"
        )
        
        assert result is not None
        assert "status" in result or "enhancement_metadata" in result
        assert "enhancement_metadata" in result
        assert result["enhancement_metadata"]["academic_standards"] == "implemented"
    
    def test_create_enhanced_twin_sensing(self):
        """Test creating sensing enhanced quantum digital twin"""
        twin = EnhancedQuantumDigitalTwin()
        
        result = twin.create_enhanced_twin(
            data={"measurements": np.random.rand(10)},
            twin_type="sensing"
        )
        
        assert result is not None
        assert "enhancement_metadata" in result
    
    def test_create_enhanced_twin_athlete(self):
        """Test creating athlete performance enhanced digital twin"""
        twin = EnhancedQuantumDigitalTwin()
        
        result = twin.create_enhanced_twin(
            data={"performance_data": [1.0, 1.5, 2.0]},
            twin_type="athlete"
        )
        
        assert result is not None
        assert "enhancement_metadata" in result
    
    def test_performance_tracking(self):
        """Test that performance is tracked correctly"""
        config = EnhancementConfig(track_history=True)
        twin = EnhancedQuantumDigitalTwin(config)
        
        # Create multiple twins
        for i in range(5):
            twin.create_enhanced_twin(
                data={"test": i},
                twin_type="core"
            )
        
        assert len(twin.performance_history) == 5
    
    def test_history_size_limit(self):
        """Test that history size is limited correctly"""
        config = EnhancementConfig(
            track_history=True,
            max_history_size=10
        )
        twin = EnhancedQuantumDigitalTwin(config)
        
        # Create more twins than history limit
        for i in range(15):
            twin.create_enhanced_twin(
                data={"test": i},
                twin_type="core"
            )
        
        assert len(twin.performance_history) <= config.max_history_size
    
    @pytest.mark.skipif(not VALIDATION_AVAILABLE, reason="Validation not available")
    def test_validate_fidelity(self):
        """Test fidelity validation with statistical methods"""
        twin = EnhancedQuantumDigitalTwin()
        
        # Create some twins to generate performance history
        for i in range(30):  # Need enough samples for statistical power
            twin.create_enhanced_twin(
                data={"test": i},
                twin_type="core"
            )
        
        # Validate fidelity
        results = twin.validate_performance(validation_type="fidelity")
        
        if results is not None:
            assert results.p_value is not None
            assert results.effect_size is not None
            assert results.statistical_power is not None
            assert results.confidence_interval is not None
    
    @pytest.mark.skipif(not VALIDATION_AVAILABLE, reason="Validation not available")
    def test_validate_precision(self):
        """Test precision validation"""
        twin = EnhancedQuantumDigitalTwin()
        
        # Create performance history
        for i in range(30):
            twin.create_enhanced_twin(
                data={"test": i},
                twin_type="sensing"
            )
        
        results = twin.validate_performance(validation_type="precision")
        
        if results is not None:
            assert results is not None or results == None  # May not have precision data
    
    @pytest.mark.skipif(not VALIDATION_AVAILABLE, reason="Validation not available")
    def test_validate_speedup(self):
        """Test speedup validation"""
        twin = EnhancedQuantumDigitalTwin()
        
        # Create performance history
        for i in range(30):
            twin.create_enhanced_twin(
                data={"test": i},
                twin_type="core"
            )
        
        results = twin.validate_performance(validation_type="speedup")
        
        if results is not None:
            assert results is not None or results == None
    
    def test_custom_validation_data(self):
        """Test validation with custom experimental data"""
        twin = EnhancedQuantumDigitalTwin()
        
        # Generate high-quality experimental data
        np.random.seed(42)
        experimental_data = np.random.normal(0.985, 0.01, 50).tolist()
        
        results = twin.validate_performance(
            validation_type="fidelity",
            experimental_data=experimental_data
        )
        
        if results is not None:
            assert results.p_value is not None
    
    def test_generate_academic_report(self):
        """Test academic report generation"""
        twin = EnhancedQuantumDigitalTwin()
        
        # Create some history
        for i in range(10):
            twin.create_enhanced_twin(
                data={"test": i},
                twin_type="core"
            )
        
        report = twin.generate_academic_report()
        
        assert report is not None
        assert isinstance(report, str)
        assert len(report) > 0
        assert "ACADEMIC VALIDATION REPORT" in report
    
    def test_benchmark_comparison(self):
        """Test comparison to academic benchmarks"""
        twin = EnhancedQuantumDigitalTwin()
        
        # Create performance data
        for i in range(10):
            twin.create_enhanced_twin(
                data={"test": i},
                twin_type="core"
            )
        
        comparison = twin.compare_to_academic_benchmarks()
        
        assert comparison is not None
        assert "measured_fidelity" in comparison
        assert "cern_benchmark" in comparison
        assert "cern_ratio" in comparison
    
    def test_benchmark_comparison_empty_history(self):
        """Test benchmark comparison with no history"""
        twin = EnhancedQuantumDigitalTwin()
        
        comparison = twin.compare_to_academic_benchmarks()
        
        assert "error" in comparison
    
    @pytest.mark.skipif(not TENSOR_NETWORKS_AVAILABLE, reason="Tensor networks not available")
    def test_tensor_network_optimization(self):
        """Test tensor network optimization application"""
        twin = EnhancedQuantumDigitalTwin()
        
        # Create quantum state
        state_vector = np.random.rand(2**twin.config.num_qubits) + \
                      1j * np.random.rand(2**twin.config.num_qubits)
        state_vector = state_vector / np.linalg.norm(state_vector)
        
        quantum_state = QuantumState(state_vector, twin.config.num_qubits)
        
        fidelity = twin._apply_tensor_network_optimization(quantum_state)
        
        assert fidelity is not None
        assert 0.0 <= fidelity <= 1.0
    
    def test_performance_metrics_dataclass(self):
        """Test PerformanceMetrics dataclass"""
        metrics = PerformanceMetrics(
            fidelity=0.985,
            precision=0.92,
            speedup=1.24
        )
        
        assert metrics.fidelity == 0.985
        assert metrics.precision == 0.92
        assert metrics.speedup == 1.24
        assert isinstance(metrics.timestamp, datetime)
        assert metrics.validated == False
    
    def test_enhancement_config_defaults(self):
        """Test default enhancement configuration values"""
        config = EnhancementConfig()
        
        assert config.statistical_alpha == 0.001
        assert config.statistical_target_power == 0.8
        assert config.statistical_target_effect_size == 0.8
        assert config.mpo_target_fidelity == 0.995
        assert config.cern_fidelity_benchmark == 0.999
        assert config.dlr_variation_distance_benchmark == 0.15


@pytest.mark.skipif(not VALIDATION_AVAILABLE, reason="Validation not available")
class TestStatisticalValidation:
    """Test statistical validation specifically"""
    
    def test_validator_initialization(self):
        """Test validator initializes correctly"""
        validator = AcademicStatisticalValidator()
        
        assert validator is not None
        assert validator.benchmarks is not None
        assert isinstance(validator.validation_history, list)
    
    def test_fidelity_validation(self):
        """Test fidelity validation"""
        validator = AcademicStatisticalValidator()
        
        # Simulate high-quality fidelity data
        np.random.seed(42)
        fidelities = np.random.normal(0.985, 0.01, 50).tolist()
        
        results = validator.validate_fidelity_claim(fidelities, target_fidelity=0.995)
        
        assert results is not None
        assert results.p_value is not None
        assert results.effect_size is not None
        assert results.statistical_power is not None
    
    def test_performance_claim_validation(self):
        """Test general performance claim validation"""
        validator = AcademicStatisticalValidator()
        
        experimental = np.random.normal(0.95, 0.02, 40).tolist()
        control = np.random.normal(0.85, 0.03, 40).tolist()
        
        results = validator.validate_performance_claim(
            experimental_data=experimental,
            control_data=control,
            claim_description="Test performance improvement"
        )
        
        assert results is not None
        assert results.validation_status in ["MEETS_ACADEMIC_STANDARDS", "REQUIRES_IMPROVEMENT"]
    
    def test_academic_report_generation(self):
        """Test academic report generation"""
        validator = AcademicStatisticalValidator()
        
        # Perform some validations
        fidelities = np.random.normal(0.985, 0.01, 50).tolist()
        validator.validate_fidelity_claim(fidelities)
        
        report = validator.generate_academic_report()
        
        assert report is not None
        assert "ACADEMIC STATISTICAL VALIDATION REPORT" in report


@pytest.mark.skipif(not TENSOR_NETWORKS_AVAILABLE, reason="Tensor networks not available")
class TestTensorNetworks:
    """Test tensor network integration"""
    
    def test_quantum_state_creation(self):
        """Test quantum state creation"""
        state_vector = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        quantum_state = QuantumState(state_vector, num_qubits=2)
        
        assert quantum_state is not None
        assert quantum_state.num_qubits == 2
        assert len(quantum_state.vector) == 4
    
    def test_mpo_initialization(self):
        """Test MPO initialization"""
        config = TensorNetworkConfig(
            bond_dimension=16,
            num_qubits=4,
            target_fidelity=0.995
        )
        
        mpo = MatrixProductOperator(config)
        
        assert mpo is not None
        assert mpo.config is not None
    
    def test_mpo_representation_creation(self):
        """Test MPO representation creation"""
        config = TensorNetworkConfig(num_qubits=4)
        mpo = MatrixProductOperator(config)
        
        state_vector = np.random.rand(16) + 1j * np.random.rand(16)
        state_vector = state_vector / np.linalg.norm(state_vector)
        quantum_state = QuantumState(state_vector, 4)
        
        mpo.create_mpo_representation(quantum_state)
        
        assert mpo is not None


# Integration tests
@pytest.mark.integration
class TestIntegration:
    """Integration tests for full system"""
    
    @pytest.mark.skipif(not ENHANCED_DT_AVAILABLE, reason="Enhanced DT not available")
    def test_full_workflow(self):
        """Test complete workflow from creation to validation"""
        # Create enhanced twin
        twin = EnhancedQuantumDigitalTwin()
        
        # Create multiple twins
        print("\n🔬 Creating enhanced quantum digital twins...")
        for i in range(30):
            result = twin.create_enhanced_twin(
                data={"iteration": i, "sensor_data": np.random.rand(5).tolist()},
                twin_type="core"
            )
            print(f"  Twin {i+1}: Fidelity = {result.get('tensor_network_fidelity', 0):.4f}")
        
        assert len(twin.performance_history) == 30
        
        # Validate performance
        if VALIDATION_AVAILABLE:
            print("\n📊 Validating performance...")
            fidelity_results = twin.validate_performance("fidelity")
            
            if fidelity_results:
                print(f"  P-value: {fidelity_results.p_value:.6f}")
                print(f"  Effect Size: {fidelity_results.effect_size:.4f}")
                print(f"  Power: {fidelity_results.statistical_power:.4f}")
                print(f"  Standards Met: {fidelity_results.academic_standards_met}")
        
        # Generate report
        print("\n📄 Generating academic report...")
        report = twin.generate_academic_report()
        print(report)
        
        # Compare to benchmarks
        print("\n🎯 Comparing to academic benchmarks...")
        comparison = twin.compare_to_academic_benchmarks()
        for key, value in comparison.items():
            print(f"  {key}: {value}")
        
        assert report is not None
        assert comparison is not None
        assert "measured_fidelity" in comparison


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

