"""
Quantum Research Laboratory Routes
Interactive quantum experiment platform for users
"""

from flask import Blueprint, render_template, jsonify, request, redirect, url_for, flash
from flask import current_app as app
import json
import time
import numpy as np
from datetime import datetime

# Create quantum lab blueprint
quantum_lab_bp = Blueprint('quantum_lab', __name__)

@quantum_lab_bp.route('/quantum-lab')
def quantum_lab():
    """Interactive quantum research laboratory"""
    try:
        return render_template('quantum_lab.html')
    except Exception as e:
        app.logger.error(f"Quantum lab error: {e}")
        return jsonify({'error': 'Quantum lab temporarily unavailable'}), 500

@quantum_lab_bp.route('/api/quantum-lab/run-experiment', methods=['POST'])
def run_quantum_experiment():
    """Execute a quantum experiment with given parameters"""
    try:
        data = request.get_json()
        algorithm = data.get('algorithm', 'qaoa')
        parameters = data.get('parameters', {})
        qasm_code = data.get('qasm_code', '')
        
        # Simulate quantum experiment execution
        execution_time = 1000 + np.random.random() * 2000  # ms
        
        # Generate realistic quantum results based on algorithm
        results = generate_experiment_results(algorithm, parameters)
        
        return jsonify({
            'success': True,
            'experiment_id': f"exp_{int(time.time())}",
            'algorithm': algorithm,
            'execution_time': execution_time,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Experiment execution error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@quantum_lab_bp.route('/api/quantum-lab/validate-qasm', methods=['POST'])
def validate_qasm():
    """Validate QASM code syntax and structure"""
    try:
        data = request.get_json()
        qasm_code = data.get('qasm_code', '')
        
        # Basic QASM validation
        validation_result = validate_qasm_syntax(qasm_code)
        
        return jsonify({
            'valid': validation_result['valid'],
            'errors': validation_result.get('errors', []),
            'warnings': validation_result.get('warnings', []),
            'gates_count': validation_result.get('gates_count', 0),
            'qubits_count': validation_result.get('qubits_count', 0)
        })
        
    except Exception as e:
        app.logger.error(f"QASM validation error: {e}")
        return jsonify({'error': str(e), 'valid': False}), 500

@quantum_lab_bp.route('/api/quantum-lab/upload-data', methods=['POST'])
def upload_data():
    """Handle data file uploads for quantum experiments"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Process the uploaded file
        file_content = file.read()
        file_type = file.filename.split('.')[-1].lower()
        
        processed_data = process_uploaded_data(file_content, file_type)
        
        return jsonify({
            'success': True,
            'filename': file.filename,
            'file_type': file_type,
            'data_preview': processed_data.get('preview', []),
            'total_rows': processed_data.get('total_rows', 0),
            'columns': processed_data.get('columns', [])
        })
        
    except Exception as e:
        app.logger.error(f"Data upload error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@quantum_lab_bp.route('/api/quantum-lab/get-templates')
def get_circuit_templates():
    """Get available quantum circuit templates"""
    try:
        templates = {
            'bell_state': {
                'name': 'Bell State Creation',
                'description': 'Creates a maximally entangled two-qubit state',
                'qasm': '''OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;''',
                'expected_results': {'00': 0.5, '11': 0.5}
            },
            'grover_2qubit': {
                'name': 'Grover Search (2-qubit)',
                'description': 'Quantum search algorithm for 2 qubits',
                'qasm': '''OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
h q[1];
z q[1];
cz q[0], q[1];
h q[0];
h q[1];
z q[0];
z q[1];
cz q[0], q[1];
h q[0];
h q[1];
measure q -> c;''',
                'expected_results': {'01': 0.8, '00': 0.067, '10': 0.067, '11': 0.067}
            },
            'superposition': {
                'name': 'Superposition Demo',
                'description': 'Demonstrates quantum superposition',
                'qasm': '''OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
h q[1];
measure q -> c;''',
                'expected_results': {'00': 0.25, '01': 0.25, '10': 0.25, '11': 0.25}
            },
            'quantum_teleportation': {
                'name': 'Quantum Teleportation',
                'description': 'Demonstrates quantum teleportation protocol',
                'qasm': '''OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[1];
cx q[1], q[2];
cx q[0], q[1];
h q[0];
measure q[0] -> c[0];
measure q[1] -> c[1];
if(c[1]==1) x q[2];
if(c[0]==1) z q[2];
measure q[2] -> c[2];''',
                'expected_results': {'000': 0.5, '100': 0.5}
            }
        }
        
        return jsonify({'templates': templates})
        
    except Exception as e:
        app.logger.error(f"Template retrieval error: {e}")
        return jsonify({'error': str(e), 'templates': {}}), 500

@quantum_lab_bp.route('/api/quantum-lab/export-results/<experiment_id>')
def export_experiment_results(experiment_id):
    """Export experiment results in various formats"""
    try:
        # In a real implementation, you'd retrieve results from a database
        # For now, we'll return a mock result
        
        results = {
            'experiment_id': experiment_id,
            'algorithm': 'qaoa',
            'parameters': {'qubits': 4, 'depth': 2},
            'execution_time': 1247.5,
            'quantum_advantage': 12.3,
            'fidelity': 95.7,
            'success_rate': 94.2,
            'measurement_data': {
                '0000': 245,
                '0001': 189,
                '0010': 156,
                '0011': 203,
                '0100': 87,
                '0101': 134,
                '0110': 78,
                '0111': 92,
                '1000': 156,
                '1001': 234,
                '1010': 198,
                '1011': 145,
                '1100': 67,
                '1101': 89,
                '1110': 123,
                '1111': 178
            },
            'timestamp': datetime.now().isoformat(),
            'platform': 'Quantum Digital Twin Platform',
            'version': '1.0'
        }
        
        return jsonify(results)
        
    except Exception as e:
        app.logger.error(f"Results export error: {e}")
        return jsonify({'error': str(e)}), 500

def generate_experiment_results(algorithm, parameters):
    """Generate realistic quantum experiment results"""
    
    if algorithm == 'qaoa':
        # QAOA results
        qubits = int(parameters.get('qaoa-qubits', 4))
        states = [format(i, f'0{qubits}b') for i in range(2**qubits)]
        
        # Generate weighted random results (QAOA tends to favor certain states)
        weights = np.random.exponential(1, len(states))
        weights = weights / np.sum(weights) * 1000
        
        results = {state: int(weight) for state, weight in zip(states, weights)}
        
        return {
            'measurement_counts': results,
            'quantum_advantage': 8.5 + np.random.random() * 8,
            'fidelity': 92 + np.random.random() * 6,
            'success_rate': 89 + np.random.random() * 9,
            'convergence_data': [0.8, 0.85, 0.91, 0.94, 0.96],
            'energy_expectation': -1.2 - np.random.random() * 0.5
        }
    
    elif algorithm == 'vqe':
        # VQE results
        return {
            'ground_state_energy': -1.857 - np.random.random() * 0.1,
            'optimization_steps': list(range(1, 51)),
            'energy_history': [-1.2 - i*0.01 - np.random.random()*0.05 for i in range(50)],
            'quantum_advantage': 6.2 + np.random.random() * 5,
            'fidelity': 94 + np.random.random() * 4,
            'success_rate': 91 + np.random.random() * 7
        }
    
    elif algorithm == 'grover':
        # Grover's algorithm results
        qubits = int(parameters.get('grover-qubits', 3))
        target = int(parameters.get('grover-target', 3))
        
        results = {}
        total_shots = 1000
        
        # Grover amplifies the target state
        for i in range(2**qubits):
            if i == target:
                results[format(i, f'0{qubits}b')] = int(total_shots * 0.85)  # High probability for target
            else:
                results[format(i, f'0{qubits}b')] = int(total_shots * 0.15 / (2**qubits - 1))
        
        return {
            'measurement_counts': results,
            'search_probability': 0.85 + np.random.random() * 0.1,
            'iterations': int(parameters.get('grover-iterations', 2)),
            'quantum_advantage': np.sqrt(2**qubits),  # Theoretical advantage
            'fidelity': 88 + np.random.random() * 8,
            'success_rate': 85 + np.random.random() * 12
        }
    
    elif algorithm == 'qml':
        # Quantum ML results
        return {
            'training_accuracy': 0.78 + np.random.random() * 0.15,
            'validation_accuracy': 0.72 + np.random.random() * 0.12,
            'loss_history': [1.2 - i*0.02 + np.random.random()*0.1 for i in range(20)],
            'feature_importance': np.random.random(4).tolist(),
            'quantum_advantage': 3.2 + np.random.random() * 4,
            'fidelity': 89 + np.random.random() * 8,
            'success_rate': 87 + np.random.random() * 10
        }
    
    else:
        # Default results
        return {
            'measurement_counts': {'00': 250, '01': 250, '10': 250, '11': 250},
            'quantum_advantage': 1.0,
            'fidelity': 95.0,
            'success_rate': 90.0
        }

def validate_qasm_syntax(qasm_code):
    """Basic QASM syntax validation"""
    
    lines = qasm_code.strip().split('\n')
    errors = []
    warnings = []
    gates_count = 0
    qubits_count = 0
    
    # Check for required headers
    if not qasm_code.startswith('OPENQASM'):
        errors.append("Missing OPENQASM version declaration")
    
    if 'include "qelib1.inc"' not in qasm_code:
        warnings.append("Standard gate library not included")
    
    # Count qubits and gates
    for line in lines:
        line = line.strip()
        if line.startswith('qreg'):
            # Extract qubit count from qreg q[n];
            try:
                qubits_count = int(line.split('[')[1].split(']')[0])
            except:
                errors.append(f"Invalid qubit register declaration: {line}")
        
        # Count gate operations
        gate_keywords = ['h', 'x', 'y', 'z', 'cx', 'cy', 'cz', 'rx', 'ry', 'rz', 't', 's', 'measure']
        for gate in gate_keywords:
            if line.startswith(gate + ' '):
                gates_count += 1
    
    # Basic validation checks
    if qubits_count == 0:
        errors.append("No qubit registers defined")
    
    if gates_count == 0:
        warnings.append("No quantum gates found")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'gates_count': gates_count,
        'qubits_count': qubits_count
    }

def process_uploaded_data(file_content, file_type):
    """Process uploaded data files"""
    
    try:
        if file_type == 'json':
            data = json.loads(file_content.decode('utf-8'))
            return {
                'preview': list(data.items())[:10] if isinstance(data, dict) else data[:10],
                'total_rows': len(data) if isinstance(data, (list, dict)) else 1,
                'columns': list(data.keys()) if isinstance(data, dict) else ['value']
            }
        
        elif file_type == 'csv':
            lines = file_content.decode('utf-8').strip().split('\n')
            headers = lines[0].split(',')
            rows = [line.split(',') for line in lines[1:6]]  # First 5 rows as preview
            
            return {
                'preview': [dict(zip(headers, row)) for row in rows],
                'total_rows': len(lines) - 1,
                'columns': headers
            }
        
        elif file_type in ['qasm', 'txt']:
            content = file_content.decode('utf-8')
            return {
                'preview': content[:500],  # First 500 characters
                'total_rows': len(content.split('\n')),
                'columns': ['code']
            }
        
        else:
            return {
                'preview': ['Binary file content'],
                'total_rows': 1,
                'columns': ['data']
            }
    
    except Exception as e:
        return {
            'preview': [f'Error processing file: {str(e)}'],
            'total_rows': 0,
            'columns': ['error']
        }