"""
Simulation routes for Quantum Trail API.
"""

import asyncio
import time
from datetime import datetime
from flask import Blueprint, request, jsonify, g
import structlog

from dt_project.quantum.quantum_digital_twin import QuantumTwinType
from dt_project.models import SimulationRun

logger = structlog.get_logger(__name__)

def create_simulation_routes(quantum_app):
    """Create simulation routes blueprint."""
    bp = Blueprint('simulation', __name__)
    
    @bp.route('/status', methods=['GET'])
    def get_system_status():
        """Get system status."""
        try:
            active_twins = quantum_app.list_active_twins()
            return jsonify({
                'status': 'operational',
                'active_simulations': len(active_twins),
                'active_twins': active_twins,
                'quantum_processor_available': quantum_app.quantum_processor is not None,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
        except Exception as e:
            logger.error(f"Status error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @bp.route('/create', methods=['POST'])
    def create_simulation():
        """Create new simulation."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            simulation_type = data.get('simulation_type')
            parameters = data.get('parameters', {})
            
            if not simulation_type:
                return jsonify({'error': 'Missing simulation_type'}), 400
            
            simulation_id = f"sim_{int(time.time() * 1000)}"
            
            # Create simulation record
            simulation = SimulationRun(
                simulation_id=simulation_id,
                simulation_type=simulation_type,
                status='pending',
                input_params=parameters,
                user_id=request.headers.get('User-ID'),
                ip_address=request.remote_addr
            )
            
            g.db_session.add(simulation)
            g.db_session.commit()
            
            # Handle different simulation types
            if simulation_type == 'athlete_performance':
                result = _create_athlete_simulation(quantum_app, simulation_id, parameters)
            elif simulation_type == 'military_mission':
                result = _create_military_simulation(quantum_app, simulation_id, parameters)
            else:
                return jsonify({'error': f'Unsupported simulation type: {simulation_type}'}), 400
            
            simulation.status = 'created'
            simulation.results = result
            g.db_session.commit()
            
            return jsonify({
                'status': 'success',
                'data': {
                    'simulation_id': simulation_id,
                    'simulation_type': simulation_type,
                    'status': 'created',
                    'result': result
                }
            }), 201
            
        except Exception as e:
            logger.error(f"Create simulation error: {e}")
            if hasattr(g, 'db_session'):
                g.db_session.rollback()
            return jsonify({'error': str(e)}), 500
    
    return bp

def _create_athlete_simulation(quantum_app, simulation_id, parameters):
    """Create athlete simulation."""
    entity_id = f"athlete_{simulation_id}"
    
    initial_state = {
        'fitness_level': parameters.get('fitness_level', 0.8),
        'endurance_level': parameters.get('endurance_level', 0.7),
        'fatigue_level': parameters.get('fatigue_level', 0.2),
        'motivation_level': parameters.get('motivation_level', 0.8)
    }
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            quantum_app.create_quantum_twin(
                entity_id=entity_id,
                twin_type=QuantumTwinType.ATHLETE,
                initial_state=initial_state,
                n_qubits=4
            )
        )
    finally:
        loop.close()
    
    parameters['entity_id'] = entity_id
    
    return {
        'entity_id': entity_id,
        'twin_type': 'athlete',
        'initial_state': initial_state,
        'quantum_enhanced': True
    }

def _create_military_simulation(quantum_app, simulation_id, parameters):
    """Create military simulation."""
    entity_id = f"unit_{simulation_id}"
    
    initial_state = {
        'readiness_level': parameters.get('readiness_level', 0.8),
        'morale': parameters.get('morale', 0.75),
        'equipment_condition': parameters.get('equipment_condition', 0.9),
        'tactical_advantage': parameters.get('tactical_advantage', 0.6)
    }
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            quantum_app.create_quantum_twin(
                entity_id=entity_id,
                twin_type=QuantumTwinType.MILITARY_UNIT,
                initial_state=initial_state,
                n_qubits=4
            )
        )
    finally:
        loop.close()
    
    parameters['entity_id'] = entity_id
    
    return {
        'entity_id': entity_id,
        'twin_type': 'military_unit',
        'initial_state': initial_state,
        'quantum_enhanced': True
    }