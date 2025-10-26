"""
Quantum computing routes for Quantum Trail API.
"""

import asyncio
import time
import json
from datetime import datetime
from flask import Blueprint, request, jsonify, g
import structlog

from dt_project.monitoring.metrics import metrics

logger = structlog.get_logger(__name__)

def create_quantum_routes(quantum_app):
    """Create quantum routes blueprint."""
    bp = Blueprint('quantum', __name__)
    
    @bp.route('/execute', methods=['POST'])
    def execute_quantum_circuit():
        """Execute a quantum circuit."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            # Validate required fields
            n_qubits = data.get('n_qubits')
            gates = data.get('gates', [])
            shots = data.get('shots', 1024)
            backend = data.get('backend', 'simulator')
            
            if not n_qubits or n_qubits < 1:
                return jsonify({'error': 'Invalid n_qubits'}), 400
            
            if not isinstance(gates, list):
                return jsonify({'error': 'Gates must be a list'}), 400
            
            # Validate shots
            if not isinstance(shots, int) or shots < 1 or shots > 100000:
                return jsonify({'error': 'Invalid shots (must be 1-100000)'}), 400
            
            # Create circuit data
            circuit_data = {
                'n_qubits': n_qubits,
                'gates': gates,
                'type': data.get('circuit_type', 'custom')
            }
            
            start_time = time.time()
            
            # Submit quantum job
            if quantum_app.quantum_processor:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    job_result = loop.run_until_complete(
                        quantum_app.quantum_processor.submit_job(
                            circuit_data=circuit_data,
                            backend_preference=backend,
                            shots=shots
                        )
                    )
                    
                    result = loop.run_until_complete(
                        quantum_app.quantum_processor.get_result(job_result['job_id'])
                    )
                finally:
                    loop.close()
            else:
                # Mock result if no quantum processor
                result = _mock_quantum_execution(circuit_data, shots)
            
            execution_time = time.time() - start_time
            
            # Record metrics
            if metrics:
                metrics.record_quantum_circuit(
                    backend=backend,
                    n_qubits=n_qubits,
                    depth=len(gates)
                )
            
            return jsonify({
                'status': 'success',
                'data': {
                    'counts': result.counts if hasattr(result, 'counts') else result,
                    'execution_time': execution_time,
                    'backend_used': backend,
                    'shots': shots,
                    'n_qubits': n_qubits,
                    'circuit_depth': len(gates)
                }
            }), 200
            
        except Exception as e:
            logger.error(f"Quantum execution error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @bp.route('/optimize', methods=['POST'])
    def run_quantum_optimization():
        """Run quantum optimization algorithm."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            algorithm = data.get('algorithm', 'qaoa').lower()
            problem_type = data.get('problem_type')
            parameters = data.get('parameters', {})
            n_qubits = data.get('n_qubits', 4)
            max_iterations = data.get('max_iterations', 100)
            
            if not problem_type:
                return jsonify({'error': 'Missing problem_type'}), 400
            
            # Validate parameters based on problem type
            if problem_type == 'maxcut':
                edges = parameters.get('edges', [])
                if not edges:
                    return jsonify({'error': 'MaxCut requires edges parameter'}), 400
            elif problem_type == 'portfolio':
                assets = parameters.get('assets', [])
                if not assets:
                    return jsonify({'error': 'Portfolio optimization requires assets'}), 400
            
            problem_data = {
                'problem_type': problem_type,
                'parameters': parameters,
                'n_qubits': n_qubits,
                'max_iterations': max_iterations
            }
            
            start_time = time.time()
            
            # Run optimization
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    quantum_app.run_quantum_optimization(algorithm, problem_data)
                )
            finally:
                loop.close()
            
            execution_time = time.time() - start_time
            result['total_execution_time'] = execution_time
            
            return jsonify({
                'status': 'success',
                'data': result
            }), 200
            
        except Exception as e:
            logger.error(f"Quantum optimization error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @bp.route('/backends', methods=['GET'])
    def get_quantum_backends():
        """Get available quantum backends."""
        try:
            if quantum_app.quantum_processor:
                backends = []
                for name, backend in quantum_app.quantum_processor.backends.items():
                    backend_info = backend.get_backend_info()
                    backends.append({
                        'name': name,
                        'type': backend_info.get('type', 'unknown'),
                        'available': backend.is_available(),
                        'queue_length': backend_info.get('queue_length', 0),
                        'max_qubits': backend_info.get('max_qubits', 0)
                    })
            else:
                backends = [
                    {
                        'name': 'mock_simulator',
                        'type': 'simulator',
                        'available': True,
                        'queue_length': 0,
                        'max_qubits': 20
                    }
                ]
            
            return jsonify({
                'status': 'success',
                'data': {
                    'backends': backends,
                    'total_backends': len(backends),
                    'available_backends': len([b for b in backends if b['available']])
                }
            }), 200
            
        except Exception as e:
            logger.error(f"Backends query error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @bp.route('/circuit/examples', methods=['GET'])
    def get_circuit_examples():
        """Get example quantum circuits."""
        examples = {
            'bell_state': {
                'description': 'Create a Bell state (maximally entangled 2-qubit state)',
                'n_qubits': 2,
                'gates': [
                    {'type': 'h', 'qubits': [0]},
                    {'type': 'cx', 'qubits': [0, 1]}
                ],
                'expected_result': 'Equal probability of |00⟩ and |11⟩'
            },
            'ghz_state': {
                'description': 'Create a GHZ state (3-qubit entangled state)',
                'n_qubits': 3,
                'gates': [
                    {'type': 'h', 'qubits': [0]},
                    {'type': 'cx', 'qubits': [0, 1]},
                    {'type': 'cx', 'qubits': [1, 2]}
                ],
                'expected_result': 'Equal probability of |000⟩ and |111⟩'
            },
            'superposition': {
                'description': 'Create superposition on all qubits',
                'n_qubits': 4,
                'gates': [
                    {'type': 'h', 'qubits': [0]},
                    {'type': 'h', 'qubits': [1]},
                    {'type': 'h', 'qubits': [2]},
                    {'type': 'h', 'qubits': [3]}
                ],
                'expected_result': 'Equal probability of all 16 basis states'
            },
            'quantum_fourier_transform': {
                'description': 'Simple 2-qubit QFT',
                'n_qubits': 2,
                'gates': [
                    {'type': 'h', 'qubits': [0]},
                    {'type': 'rz', 'qubits': [1], 'params': [1.5708]},  # π/2
                    {'type': 'cx', 'qubits': [0, 1]},
                    {'type': 'h', 'qubits': [1]},
                    {'type': 'swap', 'qubits': [0, 1]}
                ],
                'expected_result': 'Quantum Fourier Transform of input state'
            }
        }
        
        return jsonify({
            'status': 'success',
            'data': {
                'examples': examples,
                'supported_gates': [
                    'h', 'x', 'y', 'z',
                    'rx', 'ry', 'rz',
                    'cx', 'cnot', 'cz',
                    'swap'
                ]
            }
        }), 200
    
    @bp.route('/measure', methods=['POST'])
    def measure_quantum_state():
        """Measure a quantum state."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            entity_id = data.get('entity_id')
            measurement_type = data.get('measurement_type', 'computational_basis')
            collapse_state = data.get('collapse_state', False)
            
            if not entity_id:
                return jsonify({'error': 'Missing entity_id'}), 400
            
            if entity_id not in quantum_app.active_twins:
                return jsonify({'error': f'Quantum twin {entity_id} not found'}), 404
            
            twin = quantum_app.active_twins[entity_id]
            
            # Perform measurement
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    twin.state_manager.measure_quantum_state(
                        entity_id=entity_id,
                        collapse_state=collapse_state
                    )
                )
            finally:
                loop.close()
            
            return jsonify({
                'status': 'success',
                'data': {
                    'entity_id': entity_id,
                    'measurement_type': measurement_type,
                    'result': result,
                    'measured_at': datetime.utcnow().isoformat()
                }
            }), 200
            
        except Exception as e:
            logger.error(f"Measurement error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @bp.route('/state/<entity_id>', methods=['GET'])
    def get_quantum_state(entity_id):
        """Get quantum state information."""
        try:
            if entity_id not in quantum_app.active_twins:
                return jsonify({'error': f'Quantum twin {entity_id} not found'}), 404
            
            twin = quantum_app.active_twins[entity_id]
            state = twin.state_manager.get_quantum_state(entity_id)
            classical_info = twin.state_manager.get_classical_correlates(entity_id)
            
            if not state:
                return jsonify({'error': f'No quantum state for {entity_id}'}), 404
            
            return jsonify({
                'status': 'success',
                'data': {
                    'entity_id': entity_id,
                    'fidelity': state.fidelity,
                    'n_qubits': int(len(state.state_vector).bit_length() - 1),
                    'entangled_entities': state.entangled_entities,
                    'classical_correlates': classical_info,
                    'measurement_count': len(state.measurement_history),
                    'timestamp': state.timestamp.isoformat()
                }
            }), 200
            
        except Exception as e:
            logger.error(f"State query error: {e}")
            return jsonify({'error': str(e)}), 500
    
    return bp

def _mock_quantum_execution(circuit_data, shots):
    """Mock quantum execution for testing."""
    import numpy as np
    
    n_qubits = circuit_data['n_qubits']
    gates = circuit_data['gates']
    
    # Simple mock based on circuit type
    if any(gate['type'] == 'h' for gate in gates) and any(gate['type'] == 'cx' for gate in gates):
        # Bell state-like circuit
        counts = {
            '0' * n_qubits: shots // 2,
            '1' * n_qubits: shots - shots // 2
        }
    elif any(gate['type'] == 'h' for gate in gates):
        # Superposition
        n_states = min(2 ** n_qubits, 16)  # Limit for efficiency
        counts = {}
        base_count = shots // n_states
        remainder = shots % n_states
        
        for i in range(n_states):
            state = format(i, f'0{n_qubits}b')
            counts[state] = base_count + (1 if i < remainder else 0)
    else:
        # Default to all |0⟩
        counts = {'0' * n_qubits: shots}
    
    return counts