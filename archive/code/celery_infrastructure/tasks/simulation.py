"""
Simulation related Celery tasks.
Handles quantum digital twin operations and simulation management.
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import structlog
import numpy as np

from dt_project.celery_app import celery_app, simulation_task
from dt_project.quantum.quantum_digital_twin import QuantumDigitalTwin, QuantumTwinType
from dt_project.models import SimulationRun
from dt_project.monitoring.metrics import metrics

logger = structlog.get_logger(__name__)

# Global storage for active twins
_active_twins = {}

@simulation_task
def create_quantum_twin(self, entity_id: str, twin_type: str, initial_state: Dict[str, Any], n_qubits: int = 4):
    """
    Create a new quantum digital twin asynchronously.
    
    Args:
        entity_id: Unique identifier for the twin
        twin_type: Type of twin ('athlete', 'military_unit', etc.)
        initial_state: Initial state parameters
        n_qubits: Number of qubits for quantum representation
    
    Returns:
        Dict containing twin creation results
    """
    task_id = self.request.id
    logger.info("Creating quantum digital twin", 
                task_id=task_id,
                entity_id=entity_id,
                twin_type=twin_type)
    
    try:
        self.update_state(
            state='PROCESSING',
            meta={'message': 'Initializing quantum digital twin', 'progress': 20}
        )
        
        # Convert string to enum
        twin_type_enum = QuantumTwinType(twin_type.upper())
        
        # Create quantum twin
        twin = QuantumDigitalTwin(entity_id, twin_type_enum)
        
        self.update_state(
            state='PROCESSING',
            meta={'message': 'Setting up quantum state', 'progress': 50}
        )
        
        # Initialize with async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(twin.initialize(initial_state, n_qubits))
        finally:
            loop.close()
        
        # Store twin globally
        _active_twins[entity_id] = twin
        
        self.update_state(
            state='PROCESSING',
            meta={'message': 'Recording twin metadata', 'progress': 80}
        )
        
        # Record metrics
        if metrics:
            metrics.record_quantum_circuit(
                backend='simulator',
                n_qubits=n_qubits,
                depth=1
            )
        
        result = {
            'entity_id': entity_id,
            'twin_type': twin_type,
            'n_qubits': n_qubits,
            'initial_state': initial_state,
            'created_at': twin.created_at.isoformat(),
            'quantum_enhanced': True,
            'task_id': task_id
        }
        
        logger.info("Quantum digital twin created successfully", 
                   task_id=task_id,
                   entity_id=entity_id,
                   twin_type=twin_type)
        
        return {
            'status': 'success',
            'data': result
        }
        
    except Exception as e:
        logger.error("Quantum twin creation failed", 
                    task_id=task_id,
                    entity_id=entity_id,
                    error=str(e))
        
        raise self.retry(
            exc=e,
            countdown=min(60 * (self.request.retries + 1), 300),
            max_retries=3
        )

@simulation_task
def update_twin_from_sensors(self, entity_id: str, sensor_data: Dict[str, Any]):
    """
    Update quantum twin with real sensor data.
    
    Args:
        entity_id: Twin identifier
        sensor_data: New sensor readings and measurements
    
    Returns:
        Dict containing update results
    """
    task_id = self.request.id
    logger.info("Updating quantum twin from sensors", 
                task_id=task_id,
                entity_id=entity_id)
    
    try:
        if entity_id not in _active_twins:
            raise ValueError(f"Quantum twin {entity_id} not found")
        
        twin = _active_twins[entity_id]
        
        self.update_state(
            state='PROCESSING',
            meta={'message': 'Processing sensor data', 'progress': 30}
        )
        
        # Process sensor data
        processed_data = _process_sensor_data(sensor_data, twin.twin_type)
        
        self.update_state(
            state='PROCESSING',
            meta={'message': 'Updating quantum state', 'progress': 60}
        )
        
        # Update twin state
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            start_time = time.time()
            result = loop.run_until_complete(twin.update_from_sensors(processed_data))
            execution_time = time.time() - start_time
        finally:
            loop.close()
        
        # Get updated state information
        classical_info = twin.state_manager.get_classical_correlates(entity_id)
        quantum_state = twin.state_manager.get_quantum_state(entity_id)
        
        update_result = {
            'entity_id': entity_id,
            'update_count': twin.update_count,
            'current_fidelity': result.get('current_fidelity'),
            'classical_correlates': classical_info,
            'quantum_fidelity': quantum_state.fidelity if quantum_state else None,
            'sensor_data_processed': len(processed_data),
            'execution_time': execution_time,
            'updated_at': datetime.utcnow().isoformat(),
            'task_id': task_id
        }
        
        # Record metrics
        if metrics:
            metrics.twin_updates_total.inc()
        
        logger.info("Quantum twin updated successfully", 
                   task_id=task_id,
                   entity_id=entity_id,
                   fidelity=result.get('current_fidelity'))
        
        return {
            'status': 'success',
            'data': update_result
        }
        
    except Exception as e:
        logger.error("Twin sensor update failed", 
                    task_id=task_id,
                    entity_id=entity_id,
                    error=str(e))
        
        raise self.retry(
            exc=e,
            countdown=min(30 * (self.request.retries + 1), 180),
            max_retries=3
        )

@simulation_task
def run_simulation_step(self, entity_id: str, step_parameters: Dict[str, Any]):
    """
    Run a single simulation step for a quantum twin.
    
    Args:
        entity_id: Twin identifier
        step_parameters: Parameters for this simulation step
    
    Returns:
        Dict containing simulation step results
    """
    task_id = self.request.id
    logger.info("Running simulation step", 
                task_id=task_id,
                entity_id=entity_id)
    
    try:
        if entity_id not in _active_twins:
            raise ValueError(f"Quantum twin {entity_id} not found")
        
        twin = _active_twins[entity_id]
        
        self.update_state(
            state='PROCESSING',
            meta={'message': 'Preparing simulation step', 'progress': 25}
        )
        
        # Extract step parameters
        time_delta = step_parameters.get('time_delta', 1.0)
        external_forces = step_parameters.get('external_forces', {})
        environmental_factors = step_parameters.get('environmental_factors', {})
        
        self.update_state(
            state='PROCESSING',
            meta={'message': 'Computing quantum evolution', 'progress': 60}
        )
        
        # Run simulation step
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            start_time = time.time()
            
            # Simulate time evolution
            evolution_result = loop.run_until_complete(
                _simulate_time_evolution(twin, time_delta, external_forces, environmental_factors)
            )
            
            execution_time = time.time() - start_time
        finally:
            loop.close()
        
        # Collect results
        quantum_state = twin.state_manager.get_quantum_state(entity_id)
        classical_info = twin.state_manager.get_classical_correlates(entity_id)
        
        step_result = {
            'entity_id': entity_id,
            'step_number': twin.update_count,
            'time_delta': time_delta,
            'quantum_fidelity': quantum_state.fidelity if quantum_state else None,
            'classical_state': classical_info,
            'evolution_result': evolution_result,
            'execution_time': execution_time,
            'step_completed_at': datetime.utcnow().isoformat(),
            'task_id': task_id
        }
        
        logger.info("Simulation step completed", 
                   task_id=task_id,
                   entity_id=entity_id,
                   step_number=twin.update_count)
        
        return {
            'status': 'success',
            'data': step_result
        }
        
    except Exception as e:
        logger.error("Simulation step failed", 
                    task_id=task_id,
                    entity_id=entity_id,
                    error=str(e))
        
        raise self.retry(
            exc=e,
            countdown=min(45 * (self.request.retries + 1), 270),
            max_retries=2
        )

@simulation_task
def run_batch_simulation(self, simulation_config: Dict[str, Any]):
    """
    Run a batch simulation with multiple steps and entities.
    
    Args:
        simulation_config: Complete simulation configuration
    
    Returns:
        Dict containing batch simulation results
    """
    task_id = self.request.id
    simulation_id = simulation_config.get('simulation_id', f'batch_{task_id[:8]}')
    
    logger.info("Starting batch simulation", 
                task_id=task_id,
                simulation_id=simulation_id)
    
    try:
        entities = simulation_config.get('entities', [])
        n_steps = simulation_config.get('n_steps', 10)
        step_duration = simulation_config.get('step_duration', 1.0)
        
        self.update_state(
            state='PROCESSING',
            meta={'message': 'Initializing batch simulation', 'progress': 10}
        )
        
        batch_results = {
            'simulation_id': simulation_id,
            'entities': [],
            'steps_completed': 0,
            'total_steps': n_steps * len(entities),
            'start_time': datetime.utcnow().isoformat()
        }
        
        for entity_config in entities:
            entity_id = entity_config['entity_id']
            entity_results = []
            
            # Create entity twin if needed
            if entity_id not in _active_twins:
                twin_type = entity_config.get('twin_type', 'ATHLETE')
                initial_state = entity_config.get('initial_state', {})
                n_qubits = entity_config.get('n_qubits', 4)
                
                # Create twin synchronously within this task
                twin = QuantumDigitalTwin(entity_id, QuantumTwinType(twin_type.upper()))
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(twin.initialize(initial_state, n_qubits))
                finally:
                    loop.close()
                
                _active_twins[entity_id] = twin
            
            # Run simulation steps
            for step in range(n_steps):
                progress = int(((len(batch_results['entities']) * n_steps + step) / batch_results['total_steps']) * 90)
                self.update_state(
                    state='PROCESSING',
                    meta={'message': f'Step {step+1}/{n_steps} for {entity_id}', 'progress': progress}
                )
                
                step_params = {
                    'time_delta': step_duration,
                    'external_forces': entity_config.get('external_forces', {}),
                    'environmental_factors': entity_config.get('environmental_factors', {})
                }
                
                # Run simulation step directly
                try:
                    twin = _active_twins[entity_id]
                    
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        evolution_result = loop.run_until_complete(
                            _simulate_time_evolution(twin, step_duration, 
                                                   step_params.get('external_forces', {}),
                                                   step_params.get('environmental_factors', {}))
                        )
                    finally:
                        loop.close()
                    
                    quantum_state = twin.state_manager.get_quantum_state(entity_id)
                    classical_info = twin.state_manager.get_classical_correlates(entity_id)
                    
                    step_result = {
                        'step': step,
                        'quantum_fidelity': quantum_state.fidelity if quantum_state else None,
                        'classical_state': classical_info,
                        'evolution_result': evolution_result
                    }
                    
                    entity_results.append(step_result)
                    batch_results['steps_completed'] += 1
                    
                except Exception as e:
                    logger.warning(f"Step {step} failed for entity {entity_id}", error=str(e))
                    entity_results.append({'step': step, 'error': str(e)})
            
            batch_results['entities'].append({
                'entity_id': entity_id,
                'results': entity_results,
                'total_steps': len(entity_results)
            })
        
        batch_results['end_time'] = datetime.utcnow().isoformat()
        batch_results['total_duration'] = (
            datetime.fromisoformat(batch_results['end_time'].replace('Z', '+00:00')) - 
            datetime.fromisoformat(batch_results['start_time'].replace('Z', '+00:00'))
        ).total_seconds()
        
        logger.info("Batch simulation completed", 
                   task_id=task_id,
                   simulation_id=simulation_id,
                   entities_processed=len(entities),
                   steps_completed=batch_results['steps_completed'])
        
        return {
            'status': 'success',
            'data': batch_results
        }
        
    except Exception as e:
        logger.error("Batch simulation failed", 
                    task_id=task_id,
                    simulation_id=simulation_id,
                    error=str(e))
        
        raise self.retry(
            exc=e,
            countdown=min(120 * (self.request.retries + 1), 600),
            max_retries=1
        )

@simulation_task
def process_queued_simulations(self):
    """
    Process pending simulations from the database queue.
    
    Returns:
        Dict containing processing results
    """
    task_id = self.request.id
    logger.info("Processing queued simulations", task_id=task_id)
    
    try:
        # This would integrate with database to find pending simulations
        # For now, return structure for future implementation
        processed_simulations = []
        failed_simulations = []
        
        result = {
            'processed_simulations': processed_simulations,
            'failed_simulations': failed_simulations,
            'total_processed': len(processed_simulations),
            'total_failed': len(failed_simulations),
            'processed_at': datetime.utcnow().isoformat(),
            'task_id': task_id
        }
        
        logger.info("Simulation queue processed", 
                   task_id=task_id,
                   processed=len(processed_simulations),
                   failed=len(failed_simulations))
        
        return {
            'status': 'success',
            'data': result
        }
        
    except Exception as e:
        logger.error("Simulation queue processing failed", task_id=task_id, error=str(e))
        raise

# Helper functions

def _process_sensor_data(sensor_data: Dict[str, Any], twin_type: QuantumTwinType) -> Dict[str, Any]:
    """Process raw sensor data based on twin type."""
    processed_data = {}
    
    if twin_type == QuantumTwinType.ATHLETE:
        # Process athlete-specific sensors
        processed_data['heart_rate'] = sensor_data.get('heart_rate', 70)
        processed_data['speed'] = sensor_data.get('speed', 0)
        processed_data['acceleration'] = sensor_data.get('acceleration', [0, 0, 0])
        processed_data['gps_position'] = sensor_data.get('gps_position', [0, 0])
        processed_data['body_temperature'] = sensor_data.get('body_temperature', 37.0)
        processed_data['hydration_level'] = sensor_data.get('hydration_level', 0.8)
        
    elif twin_type == QuantumTwinType.MILITARY_UNIT:
        # Process military unit sensors
        processed_data['position'] = sensor_data.get('position', [0, 0])
        processed_data['equipment_status'] = sensor_data.get('equipment_status', {})
        processed_data['communication_strength'] = sensor_data.get('communication_strength', 1.0)
        processed_data['environmental_conditions'] = sensor_data.get('environmental_conditions', {})
        processed_data['threat_level'] = sensor_data.get('threat_level', 0)
        
    else:
        # Generic processing
        processed_data = sensor_data
    
    # Add timestamp
    processed_data['timestamp'] = datetime.utcnow().isoformat()
    
    return processed_data

async def _simulate_time_evolution(twin: QuantumDigitalTwin, time_delta: float, 
                                 external_forces: Dict[str, Any], 
                                 environmental_factors: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate quantum time evolution for a twin."""
    
    # Get current state
    entity_id = twin.entity_id
    quantum_state = twin.state_manager.get_quantum_state(entity_id)
    
    if not quantum_state:
        return {'error': 'No quantum state found'}
    
    # Apply time evolution based on twin type and external factors
    if twin.twin_type == QuantumTwinType.ATHLETE:
        # Simulate athlete dynamics
        fatigue_factor = external_forces.get('exercise_intensity', 0.5)
        recovery_factor = environmental_factors.get('rest_quality', 0.8)
        
        # Update states based on physical dynamics
        state_updates = {
            'fatigue_level': min(1.0, quantum_state.classical_correlates.get('fatigue_level', 0.2) + 
                                time_delta * fatigue_factor * 0.1),
            'fitness_level': max(0.0, quantum_state.classical_correlates.get('fitness_level', 0.8) + 
                               time_delta * (recovery_factor - fatigue_factor) * 0.05)
        }
        
    elif twin.twin_type == QuantumTwinType.MILITARY_UNIT:
        # Simulate military unit dynamics
        threat_level = external_forces.get('threat_level', 0.0)
        support_level = environmental_factors.get('support_availability', 1.0)
        
        state_updates = {
            'readiness_level': max(0.0, quantum_state.classical_correlates.get('readiness_level', 0.8) - 
                                 time_delta * threat_level * 0.1 + time_delta * support_level * 0.05),
            'morale': max(0.0, quantum_state.classical_correlates.get('morale', 0.75) - 
                        time_delta * threat_level * 0.08)
        }
        
    else:
        # Generic evolution
        state_updates = {
            'general_state': quantum_state.classical_correlates.get('general_state', 0.5) + 
                           np.random.normal(0, 0.1) * time_delta
        }
    
    # Apply quantum decoherence
    decoherence_rate = environmental_factors.get('decoherence_rate', 0.01)
    fidelity_decay = np.exp(-decoherence_rate * time_delta)
    
    # Update twin state
    await twin.update_from_sensors(state_updates)
    
    return {
        'time_evolved': time_delta,
        'state_updates': state_updates,
        'fidelity_decay': fidelity_decay,
        'evolution_successful': True
    }