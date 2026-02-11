"""
API Routes for Quantum Circuit Designer
Advanced API endpoints for circuit execution and optimization
"""

from flask import Blueprint, request, jsonify
from flask import current_app as app
import json
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

# Import quantum optimization modules
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../quantum'))
    from hardware_optimization import optimize_circuit, apply_error_mitigation
    from quantum_benchmarking import run_comprehensive_benchmark
    QUANTUM_MODULES_AVAILABLE = True
except ImportError:
    QUANTUM_MODULES_AVAILABLE = False

# Create API blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/quantum/execute-circuit', methods=['POST'])
def execute_circuit():
    """Execute quantum circuit with hardware optimization"""
    try:
        data = request.get_json()
        
        if not data or 'circuit' not in data:
            return jsonify({'error': 'Circuit data required'}), 400
        
        circuit_data = data['circuit']
        shots = data.get('shots', 1024)
        backend = data.get('backend', 'qasm_simulator')
        optimize = data.get('optimize', True)
        
        # Validate circuit data
        if not circuit_data.get('gates'):
            return jsonify({'error': 'Circuit must contain gates'}), 400
        
        start_time = time.time()
        
        # Execute circuit (mock implementation for now)
        result = execute_quantum_circuit_mock(circuit_data, shots, backend, optimize)
        
        execution_time = time.time() - start_time
        
        response = {
            'success': True,
            'result': result,
            'execution_time': execution_time,
            'shots': shots,
            'backend': backend,
            'optimized': optimize,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"Circuit execution error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@api_bp.route('/quantum/optimize-circuit', methods=['POST'])
def optimize_quantum_circuit():
    """Optimize quantum circuit for specific hardware backend"""
    try:
        data = request.get_json()
        
        if not data or 'circuit' not in data:
            return jsonify({'error': 'Circuit data required'}), 400
        
        circuit_data = data['circuit']
        optimization_level = data.get('optimization_level', 3)
        backend_name = data.get('backend', 'ibmq_qasm_simulator')
        
        start_time = time.time()
        
        # Mock optimization for now
        optimization_result = optimize_circuit_mock(circuit_data, optimization_level, backend_name)
        
        optimization_time = time.time() - start_time
        
        response = {
            'success': True,
            'original_circuit': circuit_data,
            'optimized_circuit': optimization_result['optimized_circuit'],
            'original_gates': optimization_result['original_gates'],
            'optimized_gates': optimization_result['optimized_gates'],
            'original_depth': optimization_result['original_depth'],
            'optimized_depth': optimization_result['optimized_depth'],
            'gate_reduction': optimization_result['gate_reduction'],
            'depth_reduction': optimization_result['depth_reduction'],
            'optimization_level': optimization_level,
            'backend': backend_name,
            'optimization_time': optimization_time,
            'techniques_applied': optimization_result['techniques'],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"Circuit optimization error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@api_bp.route('/quantum/validate-circuit', methods=['POST'])
def validate_circuit():
    """Validate quantum circuit syntax and structure"""
    try:
        data = request.get_json()
        
        if not data or 'circuit' not in data:
            return jsonify({'error': 'Circuit data required'}), 400
        
        circuit_data = data['circuit']
        
        # Perform validation
        validation_result = validate_quantum_circuit(circuit_data)
        
        response = {
            'valid': validation_result['is_valid'],
            'errors': validation_result['errors'],
            'warnings': validation_result['warnings'],
            'suggestions': validation_result['suggestions'],
            'circuit_info': {
                'qubits': validation_result['qubit_count'],
                'gates': validation_result['gate_count'],
                'depth': validation_result['circuit_depth'],
                'two_qubit_gates': validation_result['two_qubit_count'],
                'measurement_gates': validation_result['measurement_count']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"Circuit validation error: {str(e)}")
        return jsonify({
            'valid': False,
            'errors': [str(e)],
            'timestamp': datetime.now().isoformat()
        }), 500

@api_bp.route('/quantum/benchmark-circuit', methods=['POST'])
def benchmark_circuit():
    """Benchmark quantum circuit performance"""
    try:
        data = request.get_json()
        
        if not data or 'circuit' not in data:
            return jsonify({'error': 'Circuit data required'}), 400
        
        circuit_data = data['circuit']
        backends = data.get('backends', ['qasm_simulator'])
        shots = data.get('shots', 1024)
        runs = data.get('runs', 5)
        
        start_time = time.time()
        
        # Mock benchmarking
        benchmark_results = benchmark_circuit_mock(circuit_data, backends, shots, runs)
        
        benchmark_time = time.time() - start_time
        
        response = {
            'success': True,
            'circuit_info': {
                'qubits': circuit_data.get('qubits', 4),
                'gates': len(circuit_data.get('gates', [])),
                'depth': calculate_circuit_depth(circuit_data)
            },
            'benchmark_results': benchmark_results,
            'benchmark_time': benchmark_time,
            'configuration': {
                'backends': backends,
                'shots': shots,
                'runs': runs
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"Circuit benchmarking error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@api_bp.route('/quantum/transpile-circuit', methods=['POST'])
def transpile_circuit():
    """Transpile quantum circuit for specific hardware backend"""
    try:
        data = request.get_json()
        
        if not data or 'circuit' not in data:
            return jsonify({'error': 'Circuit data required'}), 400
        
        circuit_data = data['circuit']
        target_backend = data.get('backend', 'ibmq_qasm_simulator')
        optimization_level = data.get('optimization_level', 1)
        
        start_time = time.time()
        
        # Mock transpilation
        transpilation_result = transpile_circuit_mock(circuit_data, target_backend, optimization_level)
        
        transpilation_time = time.time() - start_time
        
        response = {
            'success': True,
            'original_circuit': circuit_data,
            'transpiled_circuit': transpilation_result['transpiled_circuit'],
            'coupling_map': transpilation_result['coupling_map'],
            'basis_gates': transpilation_result['basis_gates'],
            'initial_layout': transpilation_result['initial_layout'],
            'final_layout': transpilation_result['final_layout'],
            'gate_mapping': transpilation_result['gate_mapping'],
            'transpilation_time': transpilation_time,
            'backend': target_backend,
            'optimization_level': optimization_level,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"Circuit transpilation error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@api_bp.route('/quantum/generate-circuit', methods=['POST'])
def generate_circuit():
    """Generate quantum circuit from high-level algorithm specification"""
    try:
        data = request.get_json()
        
        if not data or 'algorithm' not in data:
            return jsonify({'error': 'Algorithm specification required'}), 400
        
        algorithm = data['algorithm']
        parameters = data.get('parameters', {})
        qubits = data.get('qubits', 4)
        
        # Generate circuit based on algorithm
        generated_circuit = generate_algorithm_circuit(algorithm, parameters, qubits)
        
        response = {
            'success': True,
            'algorithm': algorithm,
            'parameters': parameters,
            'generated_circuit': generated_circuit,
            'circuit_info': {
                'qubits': generated_circuit['qubits'],
                'gates': len(generated_circuit['gates']),
                'depth': calculate_circuit_depth(generated_circuit)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"Circuit generation error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@api_bp.route('/quantum/available-backends', methods=['GET'])
def get_available_backends():
    """Get list of available quantum backends"""
    try:
        # Mock backend information
        backends = [
            {
                'name': 'qasm_simulator',
                'provider': 'Qiskit Aer',
                'type': 'simulator',
                'qubits': 32,
                'status': 'online',
                'description': 'High-performance quantum circuit simulator'
            },
            {
                'name': 'statevector_simulator', 
                'provider': 'Qiskit Aer',
                'type': 'simulator',
                'qubits': 20,
                'status': 'online',
                'description': 'Exact statevector quantum simulator'
            },
            {
                'name': 'ibmq_qasm_simulator',
                'provider': 'IBM Quantum',
                'type': 'simulator',
                'qubits': 32,
                'status': 'online',
                'description': 'IBM Quantum cloud simulator'
            },
            {
                'name': 'ibmq_cairo',
                'provider': 'IBM Quantum',
                'type': 'hardware',
                'qubits': 27,
                'status': 'online',
                'description': 'IBM Quantum 27-qubit hardware backend',
                'queue_length': 15,
                'estimated_wait': '2 hours'
            },
            {
                'name': 'Aspen-M-3',
                'provider': 'Rigetti',
                'type': 'hardware', 
                'qubits': 80,
                'status': 'online',
                'description': 'Rigetti 80-qubit superconducting processor',
                'queue_length': 8,
                'estimated_wait': '45 minutes'
            },
            {
                'name': 'ionq_harmony',
                'provider': 'IonQ',
                'type': 'hardware',
                'qubits': 11,
                'status': 'online',
                'description': 'IonQ trapped-ion quantum computer',
                'queue_length': 3,
                'estimated_wait': '15 minutes'
            }
        ]
        
        response = {
            'success': True,
            'backends': backends,
            'total_backends': len(backends),
            'simulators': len([b for b in backends if b['type'] == 'simulator']),
            'hardware_devices': len([b for b in backends if b['type'] == 'hardware']),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"Backend listing error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# Helper functions for mock implementations

def execute_quantum_circuit_mock(circuit_data: Dict[str, Any], shots: int, backend: str, optimize: bool) -> Dict[str, Any]:
    """Mock quantum circuit execution"""
    
    num_qubits = circuit_data.get('qubits', 4)
    num_states = 2 ** num_qubits
    
    # Generate mock measurement results
    counts = {}
    remaining_shots = shots
    
    # Create realistic distribution
    for i in range(min(8, num_states)):  # Limit to 8 most probable states
        state = format(i, f'0{num_qubits}b')
        prob = max(0.1, 1.0 - (i * 0.15))  # Decreasing probability
        count = int(remaining_shots * prob / (8 - i))
        if count > 0:
            counts[state] = count
            remaining_shots -= count
        
        if remaining_shots <= 0:
            break
    
    # Add any remaining shots to most probable state
    if remaining_shots > 0 and counts:
        first_state = list(counts.keys())[0]
        counts[first_state] += remaining_shots
    
    # Calculate success metrics
    total_counts = sum(counts.values())
    success_rate = total_counts / shots if shots > 0 else 0
    
    return {
        'counts': counts,
        'shots': shots,
        'success_count': total_counts,
        'success_rate': success_rate,
        'backend': backend,
        'optimized': optimize,
        'quantum_time': 0.1 + (len(circuit_data.get('gates', [])) * 0.01),
        'classical_overhead': 0.05
    }

def optimize_circuit_mock(circuit_data: Dict[str, Any], optimization_level: int, backend_name: str) -> Dict[str, Any]:
    """Mock quantum circuit optimization"""
    
    original_gates = len(circuit_data.get('gates', []))
    original_depth = calculate_circuit_depth(circuit_data)
    
    # Mock optimization improvements based on level
    gate_reduction_factor = 0.1 + (optimization_level * 0.05)
    depth_reduction_factor = 0.15 + (optimization_level * 0.1)
    
    optimized_gates = max(1, int(original_gates * (1 - gate_reduction_factor)))
    optimized_depth = max(1, int(original_depth * (1 - depth_reduction_factor)))
    
    # Create optimized circuit (simplified)
    optimized_circuit = circuit_data.copy()
    optimized_circuit['gates'] = circuit_data.get('gates', [])[:optimized_gates]
    
    # List techniques based on optimization level
    techniques = ['Gate cancellation']
    if optimization_level >= 2:
        techniques.extend(['SABRE routing', 'Basis gate decomposition'])
    if optimization_level >= 3:
        techniques.extend(['Commutative gate optimization', 'Template optimization'])
    
    return {
        'optimized_circuit': optimized_circuit,
        'original_gates': original_gates,
        'optimized_gates': optimized_gates,
        'original_depth': original_depth,
        'optimized_depth': optimized_depth,
        'gate_reduction': (original_gates - optimized_gates) / original_gates if original_gates > 0 else 0,
        'depth_reduction': (original_depth - optimized_depth) / original_depth if original_depth > 0 else 0,
        'techniques': techniques
    }

def validate_quantum_circuit(circuit_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate quantum circuit structure and syntax"""
    
    errors = []
    warnings = []
    suggestions = []
    
    # Check required fields
    if 'qubits' not in circuit_data:
        errors.append("Circuit must specify number of qubits")
    
    if 'gates' not in circuit_data:
        errors.append("Circuit must contain gates array")
    
    qubits = circuit_data.get('qubits', 0)
    gates = circuit_data.get('gates', [])
    
    # Validate qubits
    if qubits <= 0:
        errors.append("Number of qubits must be positive")
    elif qubits > 20:
        warnings.append("Large number of qubits may result in slow simulation")
    
    # Validate gates
    gate_count = len(gates)
    two_qubit_count = 0
    measurement_count = 0
    
    valid_gate_types = {'H', 'X', 'Y', 'Z', 'S', 'T', 'RX', 'RY', 'RZ', 'CNOT', 'CZ', 'SWAP', 'MEASURE'}
    
    for i, gate in enumerate(gates):
        if not isinstance(gate, dict):
            errors.append(f"Gate {i} must be a dictionary")
            continue
            
        gate_type = gate.get('type')
        if not gate_type:
            errors.append(f"Gate {i} missing type field")
            continue
            
        if gate_type not in valid_gate_types:
            errors.append(f"Gate {i} has invalid type: {gate_type}")
        
        # Check qubit indices
        qubit = gate.get('qubit')
        if qubit is None:
            errors.append(f"Gate {i} missing qubit field")
        elif qubit < 0 or qubit >= qubits:
            errors.append(f"Gate {i} qubit index {qubit} out of range")
        
        # Check two-qubit gates
        if gate_type in {'CNOT', 'CZ', 'SWAP'}:
            two_qubit_count += 1
            control_qubit = gate.get('control_qubit')
            target_qubit = gate.get('target_qubit')
            
            if control_qubit is None or target_qubit is None:
                errors.append(f"Two-qubit gate {i} missing control_qubit or target_qubit")
            elif control_qubit == target_qubit:
                errors.append(f"Two-qubit gate {i} cannot have same control and target qubit")
        
        # Count measurements
        if gate_type == 'MEASURE':
            measurement_count += 1
    
    # Generate suggestions
    if gate_count == 0:
        suggestions.append("Add some gates to create a meaningful quantum circuit")
    elif gate_count > 100:
        suggestions.append("Consider circuit optimization to reduce gate count")
    
    if two_qubit_count == 0 and qubits > 1:
        suggestions.append("Add entangling gates to explore quantum advantages")
    
    if measurement_count == 0:
        suggestions.append("Add measurement gates to observe quantum results")
    
    circuit_depth = calculate_circuit_depth(circuit_data)
    if circuit_depth > 50:
        warnings.append("Deep circuits may be affected by quantum noise")
    
    return {
        'is_valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'suggestions': suggestions,
        'qubit_count': qubits,
        'gate_count': gate_count,
        'circuit_depth': circuit_depth,
        'two_qubit_count': two_qubit_count,
        'measurement_count': measurement_count
    }

def benchmark_circuit_mock(circuit_data: Dict[str, Any], backends: List[str], shots: int, runs: int) -> Dict[str, Any]:
    """Mock circuit benchmarking across backends"""
    
    results = {}
    
    for backend in backends:
        backend_results = []
        
        for run in range(runs):
            # Mock execution time based on backend type
            if 'simulator' in backend:
                execution_time = 0.1 + (len(circuit_data.get('gates', [])) * 0.01)
            else:
                execution_time = 5.0 + (len(circuit_data.get('gates', [])) * 0.5)  # Hardware takes longer
            
            # Add some variance
            execution_time *= (0.8 + 0.4 * (run / runs))
            
            backend_results.append({
                'run': run + 1,
                'execution_time': round(execution_time, 3),
                'success': True,
                'shots': shots,
                'fidelity_estimate': 0.95 - (run * 0.01)  # Mock fidelity degradation
            })
        
        # Calculate statistics
        execution_times = [r['execution_time'] for r in backend_results]
        
        results[backend] = {
            'runs': backend_results,
            'statistics': {
                'mean_execution_time': sum(execution_times) / len(execution_times),
                'min_execution_time': min(execution_times),
                'max_execution_time': max(execution_times),
                'success_rate': sum(1 for r in backend_results if r['success']) / len(backend_results),
                'total_runs': runs
            }
        }
    
    return results

def transpile_circuit_mock(circuit_data: Dict[str, Any], target_backend: str, optimization_level: int) -> Dict[str, Any]:
    """Mock circuit transpilation"""
    
    # Mock coupling map based on backend
    coupling_maps = {
        'ibmq_cairo': [(0, 1), (1, 2), (1, 4), (2, 3), (3, 5)],
        'ibmq_montreal': [(0, 1), (1, 2), (2, 3), (3, 4)],
        'Aspen-M-3': [(0, 1), (1, 2), (2, 3), (4, 5)],
        'ionq_harmony': 'all-to-all'
    }
    
    basis_gates_map = {
        'ibmq_cairo': ['cx', 'id', 'rz', 'sx', 'x'],
        'ibmq_montreal': ['cx', 'id', 'rz', 'sx', 'x'], 
        'Aspen-M-3': ['cz', 'id', 'rx', 'ry', 'rz'],
        'ionq_harmony': ['gpi', 'gpi2', 'ms']
    }
    
    # Create transpiled version (simplified)
    transpiled_circuit = circuit_data.copy()
    
    return {
        'transpiled_circuit': transpiled_circuit,
        'coupling_map': coupling_maps.get(target_backend, []),
        'basis_gates': basis_gates_map.get(target_backend, ['h', 'x', 'cx']),
        'initial_layout': list(range(circuit_data.get('qubits', 4))),
        'final_layout': list(range(circuit_data.get('qubits', 4))),
        'gate_mapping': {'H': 'ry(π/2) + rz(π)', 'CNOT': 'cx'}
    }

def generate_algorithm_circuit(algorithm: str, parameters: Dict[str, Any], qubits: int) -> Dict[str, Any]:
    """Generate circuit for specific quantum algorithm"""
    
    circuit_templates = {
        'bell_state': {
            'qubits': 2,
            'gates': [
                {'type': 'H', 'qubit': 0, 'column': 0},
                {'type': 'CNOT', 'control_qubit': 0, 'target_qubit': 1, 'qubit': 0, 'column': 1},
                {'type': 'MEASURE', 'qubit': 0, 'column': 2},
                {'type': 'MEASURE', 'qubit': 1, 'column': 2}
            ]
        },
        'ghz_state': {
            'qubits': qubits,
            'gates': [{'type': 'H', 'qubit': 0, 'column': 0}] + 
                    [{'type': 'CNOT', 'control_qubit': 0, 'target_qubit': i, 'qubit': 0, 'column': 1} for i in range(1, qubits)] +
                    [{'type': 'MEASURE', 'qubit': i, 'column': 2} for i in range(qubits)]
        },
        'qaoa': {
            'qubits': qubits,
            'gates': [{'type': 'H', 'qubit': i, 'column': 0} for i in range(qubits)] +
                    [{'type': 'RZ', 'qubit': i, 'column': 1, 'parameters': {'angle': parameters.get('gamma', 0.5)}} for i in range(qubits)] +
                    [{'type': 'RX', 'qubit': i, 'column': 2, 'parameters': {'angle': parameters.get('beta', 0.3)}} for i in range(qubits)] +
                    [{'type': 'MEASURE', 'qubit': i, 'column': 3} for i in range(qubits)]
        },
        'vqe': {
            'qubits': min(qubits, 4),
            'gates': [{'type': 'RY', 'qubit': 0, 'column': 0, 'parameters': {'angle': parameters.get('theta1', 0.5)}},
                     {'type': 'RY', 'qubit': 1, 'column': 0, 'parameters': {'angle': parameters.get('theta2', 0.3)}},
                     {'type': 'CNOT', 'control_qubit': 0, 'target_qubit': 1, 'qubit': 0, 'column': 1}] +
                    [{'type': 'MEASURE', 'qubit': i, 'column': 2} for i in range(min(qubits, 4))]
        },
        'qft': {
            'qubits': qubits,
            'gates': []
        }
    }
    
    if algorithm in circuit_templates:
        return circuit_templates[algorithm]
    else:
        # Generate random circuit for unknown algorithms
        gates = []
        for i in range(qubits * 2):
            gates.append({
                'type': ['H', 'X', 'Y', 'Z'][i % 4],
                'qubit': i % qubits,
                'column': i // qubits
            })
        
        return {
            'qubits': qubits,
            'gates': gates
        }

def calculate_circuit_depth(circuit_data: Dict[str, Any]) -> int:
    """Calculate quantum circuit depth"""
    
    gates = circuit_data.get('gates', [])
    if not gates:
        return 0
    
    # Simple depth calculation based on columns
    max_column = 0
    for gate in gates:
        column = gate.get('column', 0)
        max_column = max(max_column, column)
    
    return max_column + 1

def create_api_routes():
    """Factory function to create API routes"""
    return api_bp