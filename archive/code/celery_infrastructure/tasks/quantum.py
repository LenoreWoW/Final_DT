"""
Quantum computing related Celery tasks.
Handles circuit execution, optimization, and backend management.
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import structlog

from dt_project.celery_app import celery_app, quantum_task
from dt_project.quantum.async_quantum_backend import AsyncQuantumProcessor
from dt_project.quantum.quantum_optimization import create_quantum_optimizer
from dt_project.monitoring.metrics import metrics

logger = structlog.get_logger(__name__)

# Global quantum processor instance
_quantum_processor = None

def get_quantum_processor():
    """Get or create quantum processor instance."""
    global _quantum_processor
    if _quantum_processor is None:
        _quantum_processor = AsyncQuantumProcessor()
    return _quantum_processor

@quantum_task
def execute_quantum_circuit(self, circuit_data: Dict[str, Any], backend_preference: str = 'simulator', shots: int = 1024):
    """
    Execute a quantum circuit asynchronously.
    
    Args:
        circuit_data: Circuit specification with n_qubits, gates, etc.
        backend_preference: Preferred backend to use
        shots: Number of measurement shots
    
    Returns:
        Dict containing execution results and metadata
    """
    task_id = self.request.id
    logger.info("Starting quantum circuit execution", task_id=task_id, backend=backend_preference)
    
    try:
        # Update task state to processing
        self.update_state(
            state='PROCESSING',
            meta={'message': 'Initializing quantum processor', 'progress': 10}
        )
        
        processor = get_quantum_processor()
        
        # Create event loop for async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Submit quantum job
            self.update_state(
                state='PROCESSING',
                meta={'message': 'Submitting job to quantum backend', 'progress': 30}
            )
            
            start_time = time.time()
            job_result = loop.run_until_complete(
                processor.submit_job(
                    circuit_data=circuit_data,
                    backend_preference=backend_preference,
                    shots=shots
                )
            )
            
            # Wait for result
            self.update_state(
                state='PROCESSING',
                meta={'message': 'Waiting for quantum execution results', 'progress': 60}
            )
            
            result = loop.run_until_complete(
                processor.get_result(job_result['job_id'])
            )
            
            execution_time = time.time() - start_time
            
            # Process and format results
            self.update_state(
                state='PROCESSING',
                meta={'message': 'Processing results', 'progress': 90}
            )
            
            # Record metrics
            if metrics:
                metrics.record_quantum_circuit(
                    backend=backend_preference,
                    n_qubits=circuit_data['n_qubits'],
                    depth=len(circuit_data.get('gates', []))
                )
            
            formatted_result = {
                'counts': result.counts if hasattr(result, 'counts') else result,
                'execution_time': execution_time,
                'backend_used': backend_preference,
                'shots': shots,
                'n_qubits': circuit_data['n_qubits'],
                'circuit_depth': len(circuit_data.get('gates', [])),
                'job_id': job_result['job_id'],
                'completed_at': datetime.utcnow().isoformat(),
                'task_id': task_id
            }
            
            logger.info("Quantum circuit execution completed", 
                       task_id=task_id, 
                       execution_time=execution_time,
                       backend=backend_preference)
            
            return {
                'status': 'success',
                'data': formatted_result
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error("Quantum circuit execution failed", 
                    task_id=task_id, 
                    error=str(e),
                    backend=backend_preference)
        
        # Record failed metrics
        if metrics:
            metrics.record_failed_quantum_job(backend=backend_preference)
        
        raise self.retry(
            exc=e,
            countdown=min(60 * (self.request.retries + 1), 300),
            max_retries=2
        )

@quantum_task
def run_optimization_algorithm(self, algorithm: str, problem_data: Dict[str, Any]):
    """
    Run quantum optimization algorithm asynchronously.
    
    Args:
        algorithm: Algorithm type ('qaoa', 'vqe', 'hybrid')
        problem_data: Problem specification and parameters
    
    Returns:
        Dict containing optimization results
    """
    task_id = self.request.id
    logger.info("Starting quantum optimization", 
                task_id=task_id, 
                algorithm=algorithm,
                problem_type=problem_data.get('problem_type'))
    
    try:
        self.update_state(
            state='PROCESSING',
            meta={'message': f'Initializing {algorithm.upper()} optimizer', 'progress': 10}
        )
        
        # Create optimizer
        n_qubits = problem_data.get('n_qubits', 4)
        optimizer = create_quantum_optimizer(algorithm, n_qubits)
        
        self.update_state(
            state='PROCESSING',
            meta={'message': 'Running optimization', 'progress': 30}
        )
        
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            start_time = time.time()
            
            # Run optimization based on algorithm and problem type
            if algorithm.lower() == 'qaoa' and problem_data.get('problem_type') == 'maxcut':
                edges = problem_data['parameters']['edges']
                result = loop.run_until_complete(optimizer.optimize_maxcut(edges))
                
            elif algorithm.lower() == 'vqe':
                molecule_data = problem_data.get('parameters', {})
                result = loop.run_until_complete(
                    optimizer.optimize_molecular_hamiltonian(molecule_data)
                )
                
            elif algorithm.lower() == 'hybrid':
                if problem_data.get('problem_type') == 'portfolio':
                    assets = problem_data['parameters']['assets']
                    risk_tolerance = problem_data['parameters'].get('risk_tolerance', 0.5)
                    result = loop.run_until_complete(
                        optimizer.optimize_portfolio(assets, risk_tolerance)
                    )
                else:
                    raise ValueError(f"Unsupported hybrid problem type: {problem_data.get('problem_type')}")
                    
            else:
                raise ValueError(f"Unsupported algorithm/problem combination")
            
            execution_time = time.time() - start_time
            
            self.update_state(
                state='PROCESSING',
                meta={'message': 'Formatting results', 'progress': 90}
            )
            
            # Record metrics
            if metrics:
                metrics.record_optimization_result(
                    algorithm=algorithm,
                    iterations=result.n_function_evaluations,
                    advantage=result.quantum_advantage
                )
            
            # Format results for JSON serialization
            formatted_result = {
                'algorithm': algorithm,
                'problem_type': problem_data.get('problem_type'),
                'optimal_value': float(result.optimal_value),
                'optimal_parameters': result.optimal_parameters.tolist(),
                'convergence_history': result.convergence_history,
                'quantum_advantage': float(result.quantum_advantage),
                'execution_time': execution_time,
                'total_execution_time': execution_time,
                'n_function_evaluations': result.n_function_evaluations,
                'circuit_depth': result.circuit_depth,
                'metadata': result.metadata,
                'completed_at': datetime.utcnow().isoformat(),
                'task_id': task_id
            }
            
            logger.info("Quantum optimization completed", 
                       task_id=task_id,
                       algorithm=algorithm,
                       execution_time=execution_time,
                       optimal_value=result.optimal_value)
            
            return {
                'status': 'success',
                'data': formatted_result
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error("Quantum optimization failed", 
                    task_id=task_id,
                    algorithm=algorithm,
                    error=str(e))
        
        raise self.retry(
            exc=e,
            countdown=min(120 * (self.request.retries + 1), 600),
            max_retries=2
        )

@quantum_task
def check_backend_status(self):
    """
    Check status of all quantum backends.
    
    Returns:
        Dict containing backend status information
    """
    task_id = self.request.id
    logger.info("Checking quantum backend status", task_id=task_id)
    
    try:
        processor = get_quantum_processor()
        backend_status = {}
        
        for name, backend in processor.backends.items():
            try:
                backend_info = backend.get_backend_info()
                backend_status[name] = {
                    'name': name,
                    'type': backend_info.get('type', 'unknown'),
                    'available': backend.is_available(),
                    'queue_length': backend_info.get('queue_length', 0),
                    'max_qubits': backend_info.get('max_qubits', 0),
                    'last_checked': datetime.utcnow().isoformat()
                }
            except Exception as e:
                backend_status[name] = {
                    'name': name,
                    'available': False,
                    'error': str(e),
                    'last_checked': datetime.utcnow().isoformat()
                }
        
        # Record metrics
        if metrics:
            available_backends = sum(1 for status in backend_status.values() if status.get('available', False))
            metrics.quantum_backends_available.set(available_backends)
        
        result = {
            'backends': backend_status,
            'total_backends': len(backend_status),
            'available_backends': len([b for b in backend_status.values() if b.get('available', False)]),
            'checked_at': datetime.utcnow().isoformat(),
            'task_id': task_id
        }
        
        logger.info("Backend status check completed", 
                   task_id=task_id,
                   total_backends=result['total_backends'],
                   available_backends=result['available_backends'])
        
        return {
            'status': 'success',
            'data': result
        }
        
    except Exception as e:
        logger.error("Backend status check failed", task_id=task_id, error=str(e))
        raise

@quantum_task
def process_quantum_job_queue(self, max_jobs: int = 10):
    """
    Process queued quantum jobs in batch.
    
    Args:
        max_jobs: Maximum number of jobs to process in this batch
    
    Returns:
        Dict containing processing results
    """
    task_id = self.request.id
    logger.info("Processing quantum job queue", task_id=task_id, max_jobs=max_jobs)
    
    try:
        processor = get_quantum_processor()
        processed_jobs = []
        failed_jobs = []
        
        # This would integrate with a job queue system
        # For now, return structure for future implementation
        
        result = {
            'processed_jobs': processed_jobs,
            'failed_jobs': failed_jobs,
            'total_processed': len(processed_jobs),
            'total_failed': len(failed_jobs),
            'processed_at': datetime.utcnow().isoformat(),
            'task_id': task_id
        }
        
        logger.info("Quantum job queue processed", 
                   task_id=task_id,
                   processed=len(processed_jobs),
                   failed=len(failed_jobs))
        
        return {
            'status': 'success',
            'data': result
        }
        
    except Exception as e:
        logger.error("Quantum job queue processing failed", task_id=task_id, error=str(e))
        raise

@quantum_task
def benchmark_quantum_algorithms(self, algorithms: List[str], problem_sizes: List[int]):
    """
    Benchmark quantum algorithms across different problem sizes.
    
    Args:
        algorithms: List of algorithms to benchmark
        problem_sizes: List of problem sizes (qubit counts) to test
    
    Returns:
        Dict containing benchmark results
    """
    task_id = self.request.id
    logger.info("Starting quantum algorithm benchmarking", 
                task_id=task_id,
                algorithms=algorithms,
                problem_sizes=problem_sizes)
    
    try:
        benchmark_results = {}
        total_tests = len(algorithms) * len(problem_sizes)
        completed_tests = 0
        
        for algorithm in algorithms:
            benchmark_results[algorithm] = {}
            
            for n_qubits in problem_sizes:
                self.update_state(
                    state='PROCESSING',
                    meta={
                        'message': f'Benchmarking {algorithm} with {n_qubits} qubits',
                        'progress': int((completed_tests / total_tests) * 100)
                    }
                )
                
                try:
                    # Create test problem based on algorithm
                    if algorithm.lower() == 'qaoa':
                        # Create random MaxCut problem
                        edges = [(i, (i + 1) % n_qubits, 1.0) for i in range(n_qubits)]
                        problem_data = {
                            'problem_type': 'maxcut',
                            'parameters': {'edges': edges},
                            'n_qubits': n_qubits,
                            'max_iterations': 50
                        }
                    elif algorithm.lower() == 'vqe':
                        # Create simple Hamiltonian
                        problem_data = {
                            'problem_type': 'hamiltonian',
                            'parameters': {'name': f'test_{n_qubits}q'},
                            'n_qubits': n_qubits,
                            'max_iterations': 50
                        }
                    else:
                        continue
                    
                    # Run benchmark
                    start_time = time.time()
                    
                    optimizer = create_quantum_optimizer(algorithm, n_qubits)
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        if algorithm.lower() == 'qaoa':
                            result = loop.run_until_complete(
                                optimizer.optimize_maxcut(problem_data['parameters']['edges'])
                            )
                        else:
                            # For VQE, create mock Hamiltonian
                            from unittest.mock import Mock
                            hamiltonian = Mock()
                            hamiltonian.to_list = Mock(return_value=[('Z' + 'I' * (n_qubits-1), 1.0)])
                            result = loop.run_until_complete(
                                optimizer.find_ground_state(hamiltonian)
                            )
                    finally:
                        loop.close()
                    
                    execution_time = time.time() - start_time
                    
                    benchmark_results[algorithm][n_qubits] = {
                        'execution_time': execution_time,
                        'optimal_value': float(result.optimal_value),
                        'n_function_evaluations': result.n_function_evaluations,
                        'quantum_advantage': float(result.quantum_advantage),
                        'circuit_depth': result.circuit_depth,
                        'convergence_iterations': len(result.convergence_history)
                    }
                    
                except Exception as e:
                    logger.error(f"Benchmark failed for {algorithm} with {n_qubits} qubits", error=str(e))
                    benchmark_results[algorithm][n_qubits] = {
                        'error': str(e),
                        'failed': True
                    }
                
                completed_tests += 1
        
        result = {
            'benchmark_results': benchmark_results,
            'algorithms_tested': algorithms,
            'problem_sizes': problem_sizes,
            'total_tests': total_tests,
            'completed_tests': completed_tests,
            'benchmark_date': datetime.utcnow().isoformat(),
            'task_id': task_id
        }
        
        logger.info("Quantum algorithm benchmarking completed", 
                   task_id=task_id,
                   algorithms_tested=len(algorithms),
                   sizes_tested=len(problem_sizes))
        
        return {
            'status': 'success',
            'data': result
        }
        
    except Exception as e:
        logger.error("Quantum algorithm benchmarking failed", task_id=task_id, error=str(e))
        raise