"""
Advanced Quantum Benchmarking and Performance Analysis
Comprehensive benchmarking framework for quantum hardware and algorithms
"""

import numpy as np
import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
import concurrent.futures
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

# Import quantum libraries
try:
    from qiskit import QuantumCircuit, execute, Aer
    from qiskit.quantum_info import random_clifford, Clifford
    from qiskit.ignis.verification.randomized_benchmarking import randomized_benchmarking_seq
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests"""
    test_name: str
    num_runs: int = 10
    timeout_seconds: int = 300
    shots_per_run: int = 1024
    qubit_ranges: List[int] = field(default_factory=lambda: [2, 3, 4, 5])
    depth_ranges: List[int] = field(default_factory=lambda: [10, 20, 50, 100])
    enable_classical_comparison: bool = True
    enable_error_analysis: bool = True
    enable_scalability_test: bool = True
    enable_fidelity_benchmark: bool = True
    warm_up_runs: int = 3

@dataclass
class BenchmarkResult:
    """Single benchmark test result"""
    test_name: str
    backend_name: str
    timestamp: datetime
    execution_time: float
    success: bool
    result_data: Dict[str, Any]
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    throughput_ops_per_second: float
    latency_mean_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    success_rate: float
    error_rate: float
    resource_utilization: Dict[str, float]
    scalability_score: float
    fidelity_score: float

class QuantumBenchmarkSuite:
    """Comprehensive quantum benchmarking framework"""
    
    def __init__(self):
        self.results_storage: List[BenchmarkResult] = []
        self.performance_history: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.benchmark_configs: Dict[str, BenchmarkConfig] = {}
        
        # Performance tracking
        self.execution_times: Dict[str, List[float]] = defaultdict(list)
        self.success_rates: Dict[str, List[float]] = defaultdict(list)
        self.throughput_history: Dict[str, List[float]] = defaultdict(list)
        
        # Initialize benchmark suite
        self._initialize_benchmark_configs()
        
        # Results directory
        self.results_dir = Path("benchmark_results/quantum_benchmarks")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _initialize_benchmark_configs(self):
        """Initialize standard benchmark configurations"""
        
        # Algorithm performance benchmarks
        self.benchmark_configs["algorithm_performance"] = BenchmarkConfig(
            test_name="Algorithm Performance",
            num_runs=20,
            shots_per_run=2048,
            qubit_ranges=[2, 3, 4, 5, 6],
            depth_ranges=[5, 10, 20, 50],
            enable_classical_comparison=True
        )
        
        # Hardware characterization benchmarks
        self.benchmark_configs["hardware_characterization"] = BenchmarkConfig(
            test_name="Hardware Characterization",
            num_runs=50,
            shots_per_run=8192,
            enable_fidelity_benchmark=True,
            enable_error_analysis=True
        )
        
        # Scalability benchmarks
        self.benchmark_configs["scalability"] = BenchmarkConfig(
            test_name="Scalability Analysis",
            num_runs=15,
            qubit_ranges=[2, 4, 6, 8, 10, 12],
            depth_ranges=[10, 25, 50, 100, 200],
            enable_scalability_test=True
        )
        
        # Error rate benchmarks
        self.benchmark_configs["error_characterization"] = BenchmarkConfig(
            test_name="Error Characterization",
            num_runs=30,
            shots_per_run=4096,
            enable_error_analysis=True,
            enable_fidelity_benchmark=True
        )
        
        # Coherence benchmarks
        self.benchmark_configs["coherence_analysis"] = BenchmarkConfig(
            test_name="Coherence Analysis",
            num_runs=25,
            shots_per_run=2048,
            depth_ranges=[1, 5, 10, 20, 50, 100, 200, 500]
        )
    
    async def run_comprehensive_benchmark(self, 
                                        backend_names: List[str],
                                        benchmark_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run comprehensive benchmark suite across multiple backends"""
        
        logger.info(f"Starting comprehensive benchmark for backends: {backend_names}")
        
        if benchmark_types is None:
            benchmark_types = list(self.benchmark_configs.keys())
        
        start_time = datetime.now()
        comprehensive_results = {
            'start_time': start_time.isoformat(),
            'backend_results': {},
            'comparative_analysis': {},
            'summary_metrics': {},
            'recommendations': []
        }
        
        try:
            # Run benchmarks for each backend
            for backend_name in backend_names:
                logger.info(f"Benchmarking backend: {backend_name}")
                
                backend_results = {}
                for benchmark_type in benchmark_types:
                    config = self.benchmark_configs[benchmark_type]
                    
                    logger.info(f"Running {benchmark_type} benchmark")
                    benchmark_results = await self._run_benchmark_type(
                        backend_name, benchmark_type, config
                    )
                    backend_results[benchmark_type] = benchmark_results
                
                comprehensive_results['backend_results'][backend_name] = backend_results
            
            # Perform comparative analysis
            if len(backend_names) > 1:
                comparative_analysis = await self._perform_comparative_analysis(
                    comprehensive_results['backend_results']
                )
                comprehensive_results['comparative_analysis'] = comparative_analysis
            
            # Generate summary metrics
            summary_metrics = self._generate_summary_metrics(
                comprehensive_results['backend_results']
            )
            comprehensive_results['summary_metrics'] = summary_metrics
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                comprehensive_results['backend_results'],
                comprehensive_results.get('comparative_analysis', {})
            )
            comprehensive_results['recommendations'] = recommendations
            
            # Save results
            end_time = datetime.now()
            comprehensive_results['end_time'] = end_time.isoformat()
            comprehensive_results['total_duration'] = (end_time - start_time).total_seconds()
            
            await self._save_comprehensive_results(comprehensive_results)
            
            logger.info(f"Comprehensive benchmark completed in {comprehensive_results['total_duration']:.2f}s")
            
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"Comprehensive benchmark failed: {str(e)}")
            comprehensive_results['error'] = str(e)
            comprehensive_results['end_time'] = datetime.now().isoformat()
            return comprehensive_results
    
    async def _run_benchmark_type(self, 
                                backend_name: str, 
                                benchmark_type: str, 
                                config: BenchmarkConfig) -> Dict[str, Any]:
        """Run specific benchmark type for a backend"""
        
        results = {
            'config': asdict(config),
            'individual_results': [],
            'aggregated_metrics': {},
            'performance_analysis': {}
        }
        
        try:
            # Warm-up runs
            logger.info(f"Running {config.warm_up_runs} warm-up iterations")
            for _ in range(config.warm_up_runs):
                await self._run_single_benchmark(backend_name, benchmark_type, config)
            
            # Actual benchmark runs
            logger.info(f"Running {config.num_runs} benchmark iterations")
            benchmark_results = []
            
            for run_idx in range(config.num_runs):
                try:
                    result = await self._run_single_benchmark(backend_name, benchmark_type, config)
                    benchmark_results.append(result)
                    results['individual_results'].append(result.to_dict())
                    
                    if run_idx % 5 == 0:
                        logger.info(f"Completed {run_idx + 1}/{config.num_runs} runs")
                        
                except Exception as e:
                    logger.error(f"Benchmark run {run_idx + 1} failed: {str(e)}")
                    failed_result = BenchmarkResult(
                        test_name=config.test_name,
                        backend_name=backend_name,
                        timestamp=datetime.now(),
                        execution_time=0.0,
                        success=False,
                        result_data={},
                        error_message=str(e)
                    )
                    results['individual_results'].append(failed_result.to_dict())
            
            # Analyze results
            if benchmark_results:
                aggregated_metrics = self._aggregate_benchmark_results(benchmark_results)
                results['aggregated_metrics'] = aggregated_metrics
                
                performance_analysis = await self._analyze_performance(
                    benchmark_results, benchmark_type
                )
                results['performance_analysis'] = performance_analysis
            
            return results
            
        except Exception as e:
            logger.error(f"Benchmark type {benchmark_type} failed: {str(e)}")
            results['error'] = str(e)
            return results
    
    async def _run_single_benchmark(self, 
                                  backend_name: str, 
                                  benchmark_type: str, 
                                  config: BenchmarkConfig) -> BenchmarkResult:
        """Run single benchmark iteration"""
        
        start_time = datetime.now()
        
        try:
            # Select benchmark function based on type
            benchmark_function = self._get_benchmark_function(benchmark_type)
            
            # Execute benchmark with timeout
            async with asyncio.timeout(config.timeout_seconds):
                result_data = await benchmark_function(backend_name, config)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return BenchmarkResult(
                test_name=config.test_name,
                backend_name=backend_name,
                timestamp=start_time,
                execution_time=execution_time,
                success=True,
                result_data=result_data,
                metadata={
                    'shots': config.shots_per_run,
                    'benchmark_type': benchmark_type
                }
            )
            
        except asyncio.TimeoutError:
            execution_time = (datetime.now() - start_time).total_seconds()
            return BenchmarkResult(
                test_name=config.test_name,
                backend_name=backend_name,
                timestamp=start_time,
                execution_time=execution_time,
                success=False,
                result_data={},
                error_message=f"Benchmark timed out after {config.timeout_seconds}s"
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return BenchmarkResult(
                test_name=config.test_name,
                backend_name=backend_name,
                timestamp=start_time,
                execution_time=execution_time,
                success=False,
                result_data={},
                error_message=str(e)
            )
    
    def _get_benchmark_function(self, benchmark_type: str) -> Callable:
        """Get benchmark function for specific type"""
        
        benchmark_functions = {
            'algorithm_performance': self._benchmark_algorithm_performance,
            'hardware_characterization': self._benchmark_hardware_characterization,
            'scalability': self._benchmark_scalability,
            'error_characterization': self._benchmark_error_characterization,
            'coherence_analysis': self._benchmark_coherence_analysis
        }
        
        return benchmark_functions.get(benchmark_type, self._benchmark_default)
    
    async def _benchmark_algorithm_performance(self, backend_name: str, config: BenchmarkConfig) -> Dict[str, Any]:
        """Benchmark quantum algorithm performance"""
        
        results = {
            'algorithm_results': {},
            'execution_times': {},
            'success_rates': {},
            'fidelity_scores': {}
        }
        
        # Test different quantum algorithms
        algorithms = ['qaoa', 'vqe', 'grover', 'qft']
        
        for algorithm in algorithms:
            algorithm_results = []
            
            for num_qubits in config.qubit_ranges:
                if num_qubits > 10:  # Practical limit for simulation
                    continue
                
                # Generate test circuit for algorithm
                circuit = self._generate_algorithm_circuit(algorithm, num_qubits)
                
                # Execute and measure performance
                start_time = time.time()
                execution_result = await self._execute_circuit(circuit, backend_name, config.shots_per_run)
                execution_time = time.time() - start_time
                
                algorithm_results.append({
                    'qubits': num_qubits,
                    'execution_time': execution_time,
                    'result': execution_result,
                    'success': execution_result.get('success', True)
                })
            
            results['algorithm_results'][algorithm] = algorithm_results
            results['execution_times'][algorithm] = [r['execution_time'] for r in algorithm_results]
            results['success_rates'][algorithm] = sum(r['success'] for r in algorithm_results) / len(algorithm_results)
        
        return results
    
    async def _benchmark_hardware_characterization(self, backend_name: str, config: BenchmarkConfig) -> Dict[str, Any]:
        """Benchmark hardware characteristics"""
        
        results = {
            'gate_fidelities': {},
            'readout_errors': {},
            'coherence_times': {},
            'crosstalk_analysis': {}
        }
        
        # Single-qubit gate fidelity
        for num_qubits in config.qubit_ranges[:3]:  # Limit for detailed analysis
            gate_fidelities = {}
            
            # Test different single-qubit gates
            for gate in ['x', 'y', 'z', 'h']:
                circuit = self._generate_gate_fidelity_circuit(gate, num_qubits)
                fidelity = await self._measure_gate_fidelity(circuit, backend_name, config.shots_per_run)
                gate_fidelities[gate] = fidelity
            
            results['gate_fidelities'][f'{num_qubits}_qubits'] = gate_fidelities
        
        # Two-qubit gate fidelity (CNOT)
        if len(config.qubit_ranges) >= 2:
            cnot_fidelity = await self._measure_cnot_fidelity(backend_name, config.shots_per_run)
            results['gate_fidelities']['cnot'] = cnot_fidelity
        
        # Readout error characterization
        readout_errors = await self._measure_readout_errors(backend_name, config.shots_per_run)
        results['readout_errors'] = readout_errors
        
        return results
    
    async def _benchmark_scalability(self, backend_name: str, config: BenchmarkConfig) -> Dict[str, Any]:
        """Benchmark quantum system scalability"""
        
        results = {
            'qubit_scaling': {},
            'depth_scaling': {},
            'complexity_analysis': {}
        }
        
        # Qubit scaling analysis
        execution_times = []
        for num_qubits in config.qubit_ranges:
            if num_qubits > 12:  # Practical simulation limit
                break
            
            circuit = self._generate_scalability_circuit(num_qubits, depth=20)
            
            start_time = time.time()
            await self._execute_circuit(circuit, backend_name, config.shots_per_run)
            execution_time = time.time() - start_time
            
            execution_times.append({
                'qubits': num_qubits,
                'execution_time': execution_time,
                'circuit_depth': 20
            })
        
        results['qubit_scaling'] = execution_times
        
        # Depth scaling analysis (fixed qubits)
        if config.qubit_ranges:
            fixed_qubits = min(config.qubit_ranges[-1], 6)
            depth_times = []
            
            for depth in config.depth_ranges:
                if depth > 500:  # Practical limit
                    break
                
                circuit = self._generate_scalability_circuit(fixed_qubits, depth)
                
                start_time = time.time()
                await self._execute_circuit(circuit, backend_name, config.shots_per_run)
                execution_time = time.time() - start_time
                
                depth_times.append({
                    'depth': depth,
                    'execution_time': execution_time,
                    'qubits': fixed_qubits
                })
            
            results['depth_scaling'] = depth_times
        
        return results
    
    async def _benchmark_error_characterization(self, backend_name: str, config: BenchmarkConfig) -> Dict[str, Any]:
        """Benchmark quantum error characteristics"""
        
        results = {
            'randomized_benchmarking': {},
            'process_tomography': {},
            'state_tomography': {}
        }
        
        # Simplified randomized benchmarking
        if QISKIT_AVAILABLE:
            rb_results = await self._perform_randomized_benchmarking(backend_name, config.shots_per_run)
            results['randomized_benchmarking'] = rb_results
        
        # Basic error rate measurement
        error_rates = await self._measure_basic_error_rates(backend_name, config.shots_per_run)
        results['error_rates'] = error_rates
        
        return results
    
    async def _benchmark_coherence_analysis(self, backend_name: str, config: BenchmarkConfig) -> Dict[str, Any]:
        """Benchmark quantum coherence characteristics"""
        
        results = {
            'T1_measurements': {},
            'T2_measurements': {},
            'coherence_scaling': {}
        }
        
        # Simulate T1 and T2 measurements
        for depth in config.depth_ranges:
            if depth > 1000:
                break
            
            # T1-like measurement (amplitude damping)
            t1_result = await self._simulate_t1_measurement(backend_name, depth, config.shots_per_run)
            results['T1_measurements'][f'depth_{depth}'] = t1_result
            
            # T2-like measurement (dephasing)
            t2_result = await self._simulate_t2_measurement(backend_name, depth, config.shots_per_run)
            results['T2_measurements'][f'depth_{depth}'] = t2_result
        
        return results
    
    async def _benchmark_default(self, backend_name: str, config: BenchmarkConfig) -> Dict[str, Any]:
        """Default benchmark for unknown types"""
        
        circuit = self._generate_random_circuit(4, 20)
        result = await self._execute_circuit(circuit, backend_name, config.shots_per_run)
        
        return {
            'default_benchmark': True,
            'result': result
        }
    
    def _aggregate_benchmark_results(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Aggregate multiple benchmark results into summary statistics"""
        
        if not results:
            return {}
        
        successful_results = [r for r in results if r.success]
        execution_times = [r.execution_time for r in successful_results]
        
        if not execution_times:
            return {'success_rate': 0.0}
        
        aggregated = {
            'success_rate': len(successful_results) / len(results),
            'total_runs': len(results),
            'successful_runs': len(successful_results),
            'execution_time_stats': {
                'mean': statistics.mean(execution_times),
                'median': statistics.median(execution_times),
                'std_dev': statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0,
                'min': min(execution_times),
                'max': max(execution_times),
                'p95': np.percentile(execution_times, 95),
                'p99': np.percentile(execution_times, 99)
            },
            'throughput_ops_per_second': len(successful_results) / sum(execution_times) if sum(execution_times) > 0 else 0.0
        }
        
        return aggregated
    
    async def _analyze_performance(self, results: List[BenchmarkResult], benchmark_type: str) -> Dict[str, Any]:
        """Analyze performance characteristics from benchmark results"""
        
        analysis = {
            'performance_grade': 'Unknown',
            'bottlenecks': [],
            'optimization_suggestions': [],
            'trend_analysis': {}
        }
        
        try:
            successful_results = [r for r in results if r.success]
            if not successful_results:
                analysis['performance_grade'] = 'Failed'
                return analysis
            
            # Calculate performance metrics
            execution_times = [r.execution_time for r in successful_results]
            mean_time = statistics.mean(execution_times)
            success_rate = len(successful_results) / len(results)
            
            # Grade performance
            if success_rate >= 0.95 and mean_time < 1.0:
                analysis['performance_grade'] = 'Excellent'
            elif success_rate >= 0.85 and mean_time < 5.0:
                analysis['performance_grade'] = 'Good'
            elif success_rate >= 0.7 and mean_time < 10.0:
                analysis['performance_grade'] = 'Fair'
            else:
                analysis['performance_grade'] = 'Poor'
            
            # Identify bottlenecks
            if mean_time > 10.0:
                analysis['bottlenecks'].append('High execution time')
            if success_rate < 0.8:
                analysis['bottlenecks'].append('Low success rate')
            
            # Generate suggestions
            if mean_time > 5.0:
                analysis['optimization_suggestions'].append('Consider circuit optimization')
            if success_rate < 0.9:
                analysis['optimization_suggestions'].append('Implement error mitigation')
            
            return analysis
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {str(e)}")
            analysis['error'] = str(e)
            return analysis
    
    async def _perform_comparative_analysis(self, backend_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comparative analysis across multiple backends"""
        
        analysis = {
            'performance_rankings': {},
            'relative_performance': {},
            'best_use_cases': {},
            'comparative_metrics': {}
        }
        
        try:
            # Extract performance metrics for comparison
            backend_metrics = {}
            
            for backend_name, results in backend_results.items():
                metrics = {}
                
                for benchmark_type, benchmark_data in results.items():
                    if 'aggregated_metrics' in benchmark_data:
                        agg_metrics = benchmark_data['aggregated_metrics']
                        metrics[benchmark_type] = {
                            'success_rate': agg_metrics.get('success_rate', 0.0),
                            'mean_execution_time': agg_metrics.get('execution_time_stats', {}).get('mean', float('inf')),
                            'throughput': agg_metrics.get('throughput_ops_per_second', 0.0)
                        }
                
                backend_metrics[backend_name] = metrics
            
            # Rank backends by overall performance
            overall_scores = {}
            for backend_name, metrics in backend_metrics.items():
                score = 0.0
                count = 0
                
                for benchmark_type, benchmark_metrics in metrics.items():
                    # Weighted score: success_rate * throughput / execution_time
                    success_rate = benchmark_metrics['success_rate']
                    throughput = benchmark_metrics['throughput']
                    exec_time = benchmark_metrics['mean_execution_time']
                    
                    if exec_time > 0 and exec_time != float('inf'):
                        benchmark_score = (success_rate * throughput) / exec_time
                        score += benchmark_score
                        count += 1
                
                overall_scores[backend_name] = score / count if count > 0 else 0.0
            
            # Rank by overall score
            ranked_backends = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
            analysis['performance_rankings'] = {
                f'rank_{i+1}': {'backend': backend, 'score': score}
                for i, (backend, score) in enumerate(ranked_backends)
            }
            
            # Best use cases analysis
            for benchmark_type in ['algorithm_performance', 'scalability', 'error_characterization']:
                best_backend = None
                best_score = -1
                
                for backend_name, metrics in backend_metrics.items():
                    if benchmark_type in metrics:
                        bench_metrics = metrics[benchmark_type]
                        score = bench_metrics['success_rate'] * bench_metrics['throughput']
                        
                        if score > best_score:
                            best_score = score
                            best_backend = backend_name
                
                if best_backend:
                    analysis['best_use_cases'][benchmark_type] = best_backend
            
            return analysis
            
        except Exception as e:
            logger.error(f"Comparative analysis failed: {str(e)}")
            analysis['error'] = str(e)
            return analysis
    
    def _generate_summary_metrics(self, backend_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary metrics across all benchmarks"""
        
        summary = {
            'total_backends': len(backend_results),
            'total_benchmarks': 0,
            'overall_success_rate': 0.0,
            'average_execution_time': 0.0,
            'performance_distribution': {},
            'reliability_scores': {}
        }
        
        try:
            all_success_rates = []
            all_execution_times = []
            total_benchmarks = 0
            
            for backend_name, results in backend_results.items():
                backend_success_rates = []
                backend_execution_times = []
                
                for benchmark_type, benchmark_data in results.items():
                    total_benchmarks += 1
                    
                    if 'aggregated_metrics' in benchmark_data:
                        metrics = benchmark_data['aggregated_metrics']
                        success_rate = metrics.get('success_rate', 0.0)
                        exec_time_stats = metrics.get('execution_time_stats', {})
                        mean_time = exec_time_stats.get('mean', 0.0)
                        
                        all_success_rates.append(success_rate)
                        backend_success_rates.append(success_rate)
                        
                        if mean_time > 0:
                            all_execution_times.append(mean_time)
                            backend_execution_times.append(mean_time)
                
                # Calculate backend reliability score
                if backend_success_rates:
                    reliability_score = statistics.mean(backend_success_rates)
                    summary['reliability_scores'][backend_name] = reliability_score
            
            # Overall statistics
            summary['total_benchmarks'] = total_benchmarks
            
            if all_success_rates:
                summary['overall_success_rate'] = statistics.mean(all_success_rates)
            
            if all_execution_times:
                summary['average_execution_time'] = statistics.mean(all_execution_times)
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary metrics generation failed: {str(e)}")
            summary['error'] = str(e)
            return summary
    
    def _generate_recommendations(self, 
                                backend_results: Dict[str, Any], 
                                comparative_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on benchmark results"""
        
        recommendations = []
        
        try:
            # Analyze overall performance
            for backend_name, results in backend_results.items():
                backend_recommendations = []
                
                for benchmark_type, benchmark_data in results.items():
                    if 'aggregated_metrics' in benchmark_data:
                        metrics = benchmark_data['aggregated_metrics']
                        success_rate = metrics.get('success_rate', 0.0)
                        
                        if success_rate < 0.8:
                            backend_recommendations.append(
                                f"Improve {benchmark_type} reliability for {backend_name} (current: {success_rate:.1%})"
                            )
                
                if backend_recommendations:
                    recommendations.extend(backend_recommendations[:2])  # Limit recommendations
            
            # General recommendations
            if 'performance_rankings' in comparative_analysis:
                rankings = comparative_analysis['performance_rankings']
                if rankings:
                    top_backend = rankings.get('rank_1', {}).get('backend')
                    if top_backend:
                        recommendations.append(f"Consider {top_backend} for production workloads")
            
            # Add standard recommendations
            recommendations.extend([
                "Implement error mitigation strategies for improved accuracy",
                "Use circuit optimization to reduce execution time",
                "Monitor hardware performance regularly for degradation",
                "Consider hybrid classical-quantum approaches for complex problems"
            ])
            
            return recommendations[:5]  # Limit to top 5 recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {str(e)}")
            return ["Unable to generate recommendations due to analysis error"]
    
    # Mock implementations for circuit generation and execution
    def _generate_algorithm_circuit(self, algorithm: str, num_qubits: int) -> str:
        """Generate test circuit for specific algorithm"""
        return f"{algorithm}_circuit_{num_qubits}q"
    
    def _generate_gate_fidelity_circuit(self, gate: str, num_qubits: int) -> str:
        """Generate circuit for gate fidelity measurement"""
        return f"{gate}_fidelity_circuit_{num_qubits}q"
    
    def _generate_scalability_circuit(self, num_qubits: int, depth: int) -> str:
        """Generate circuit for scalability testing"""
        return f"scalability_circuit_{num_qubits}q_{depth}d"
    
    def _generate_random_circuit(self, num_qubits: int, depth: int) -> str:
        """Generate random circuit for testing"""
        return f"random_circuit_{num_qubits}q_{depth}d"
    
    async def _execute_circuit(self, circuit: str, backend_name: str, shots: int) -> Dict[str, Any]:
        """Mock circuit execution"""
        # Simulate execution time based on circuit complexity
        await asyncio.sleep(0.1 + np.random.exponential(0.2))
        
        success_prob = 0.9 if 'ibm' in backend_name.lower() else 0.8
        success = np.random.random() < success_prob
        
        return {
            'success': success,
            'counts': {'00': shots//2, '11': shots//2} if success else {'error': shots},
            'execution_time': 0.1 + np.random.exponential(0.2)
        }
    
    async def _measure_gate_fidelity(self, circuit: str, backend_name: str, shots: int) -> float:
        """Mock gate fidelity measurement"""
        base_fidelity = 0.95 if 'ibm' in backend_name.lower() else 0.92
        return base_fidelity + np.random.normal(0, 0.02)
    
    async def _measure_cnot_fidelity(self, backend_name: str, shots: int) -> float:
        """Mock CNOT fidelity measurement"""
        base_fidelity = 0.90 if 'ibm' in backend_name.lower() else 0.87
        return base_fidelity + np.random.normal(0, 0.03)
    
    async def _measure_readout_errors(self, backend_name: str, shots: int) -> Dict[str, float]:
        """Mock readout error measurement"""
        base_error = 0.02 if 'ibm' in backend_name.lower() else 0.03
        return {
            'qubit_0': base_error + np.random.normal(0, 0.005),
            'qubit_1': base_error + np.random.normal(0, 0.005)
        }
    
    async def _perform_randomized_benchmarking(self, backend_name: str, shots: int) -> Dict[str, Any]:
        """Mock randomized benchmarking"""
        return {
            'gate_fidelity': 0.95 + np.random.normal(0, 0.02),
            'decay_rate': 0.01 + np.random.normal(0, 0.005),
            'confidence_interval': [0.94, 0.97]
        }
    
    async def _measure_basic_error_rates(self, backend_name: str, shots: int) -> Dict[str, float]:
        """Mock basic error rate measurement"""
        return {
            'single_qubit_error': 0.001 + np.random.normal(0, 0.0002),
            'two_qubit_error': 0.01 + np.random.normal(0, 0.002),
            'readout_error': 0.02 + np.random.normal(0, 0.005)
        }
    
    async def _simulate_t1_measurement(self, backend_name: str, depth: int, shots: int) -> Dict[str, Any]:
        """Mock T1 coherence measurement"""
        t1_time = 100e-6 + np.random.normal(0, 10e-6)  # ~100 microseconds
        decay = np.exp(-depth * 1e-6 / t1_time)  # Assuming 1us per depth unit
        
        return {
            'T1_estimate': t1_time,
            'decay_factor': decay,
            'measurement_depth': depth
        }
    
    async def _simulate_t2_measurement(self, backend_name: str, depth: int, shots: int) -> Dict[str, Any]:
        """Mock T2 coherence measurement"""
        t2_time = 50e-6 + np.random.normal(0, 5e-6)  # ~50 microseconds
        decay = np.exp(-depth * 1e-6 / t2_time)
        
        return {
            'T2_estimate': t2_time,
            'decay_factor': decay,
            'measurement_depth': depth
        }
    
    async def _save_comprehensive_results(self, results: Dict[str, Any]):
        """Save comprehensive benchmark results to file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_benchmark_{timestamp}.json"
        filepath = self.results_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Comprehensive benchmark results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save benchmark results: {str(e)}")

# Global benchmark instance
quantum_benchmark = QuantumBenchmarkSuite()

async def run_comprehensive_benchmark(backend_names: List[str], 
                                    benchmark_types: Optional[List[str]] = None) -> Dict[str, Any]:
    """Convenient function to run comprehensive benchmarks"""
    return await quantum_benchmark.run_comprehensive_benchmark(backend_names, benchmark_types)

def get_benchmark_report() -> Dict[str, Any]:
    """Get benchmark performance report"""
    return {
        'total_results': len(quantum_benchmark.results_storage),
        'performance_history': dict(quantum_benchmark.performance_history),
        'available_configs': list(quantum_benchmark.benchmark_configs.keys())
    }