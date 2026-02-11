"""
WebSocket handler for real-time quantum state updates and notifications.
"""

import asyncio
import json
import logging
from typing import Dict, Set, Any, Optional
from datetime import datetime
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
from flask import request
import structlog

logger = structlog.get_logger(__name__)

class QuantumWebSocketHandler:
    """Manages WebSocket connections for real-time quantum updates."""
    
    def __init__(self, socketio: SocketIO):
        self.socketio = socketio
        self.connected_clients: Dict[str, Set[str]] = {}  # room -> client_ids
        self.client_subscriptions: Dict[str, Set[str]] = {}  # client_id -> subscriptions
        self.quantum_state_cache: Dict[str, Any] = {}
        self.active_simulations: Dict[str, Any] = {}
        
        # Register event handlers
        self._register_handlers()
        
    def _register_handlers(self):
        """Register Socket.IO event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            client_id = request.sid
            logger.info(f"Client connected: {client_id}")
            
            # Send initial connection confirmation
            emit('connected', {
                'client_id': client_id,
                'timestamp': datetime.utcnow().isoformat(),
                'available_rooms': self._get_available_rooms()
            })
            
            # Add to default room
            self._add_client_to_room(client_id, 'default')
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            client_id = request.sid
            logger.info(f"Client disconnected: {client_id}")
            
            # Remove from all rooms and subscriptions
            self._remove_client(client_id)
        
        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            """Handle subscription to specific events or entities."""
            client_id = request.sid
            subscription_type = data.get('type')
            entity_id = data.get('entity_id')
            
            if not subscription_type:
                emit('error', {'message': 'Subscription type required'})
                return
            
            # Add subscription
            if client_id not in self.client_subscriptions:
                self.client_subscriptions[client_id] = set()
            
            subscription_key = f"{subscription_type}:{entity_id}" if entity_id else subscription_type
            self.client_subscriptions[client_id].add(subscription_key)
            
            # Join room for this subscription
            join_room(subscription_key)
            
            logger.info(f"Client {client_id} subscribed to {subscription_key}")
            
            emit('subscribed', {
                'subscription': subscription_key,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Send initial state if available
            self._send_initial_state(client_id, subscription_type, entity_id)
        
        @self.socketio.on('unsubscribe')
        def handle_unsubscribe(data):
            """Handle unsubscription from events."""
            client_id = request.sid
            subscription_type = data.get('type')
            entity_id = data.get('entity_id')
            
            subscription_key = f"{subscription_type}:{entity_id}" if entity_id else subscription_type
            
            if client_id in self.client_subscriptions:
                self.client_subscriptions[client_id].discard(subscription_key)
            
            leave_room(subscription_key)
            
            emit('unsubscribed', {
                'subscription': subscription_key,
                'timestamp': datetime.utcnow().isoformat()
            })
        
        @self.socketio.on('quantum_measurement_request')
        def handle_measurement_request(data):
            """Handle request for quantum measurement."""
            client_id = request.sid
            entity_id = data.get('entity_id')
            measurement_type = data.get('measurement_type', 'computational_basis')
            
            logger.info(f"Measurement request from {client_id} for {entity_id}")
            
            # Simulate async measurement
            self.socketio.start_background_task(
                self._perform_quantum_measurement,
                client_id,
                entity_id,
                measurement_type
            )
        
        @self.socketio.on('simulation_control')
        def handle_simulation_control(data):
            """Handle simulation control commands (start, stop, pause)."""
            client_id = request.sid
            command = data.get('command')
            simulation_id = data.get('simulation_id')
            
            if command == 'start':
                self._start_simulation(client_id, simulation_id, data.get('params', {}))
            elif command == 'stop':
                self._stop_simulation(simulation_id)
            elif command == 'pause':
                self._pause_simulation(simulation_id)
            elif command == 'resume':
                self._resume_simulation(simulation_id)
            else:
                emit('error', {'message': f'Unknown command: {command}'})
        
        @self.socketio.on('ping')
        def handle_ping():
            """Handle ping for connection keepalive."""
            emit('pong', {'timestamp': datetime.utcnow().isoformat()})
    
    def _add_client_to_room(self, client_id: str, room: str):
        """Add client to a room."""
        if room not in self.connected_clients:
            self.connected_clients[room] = set()
        self.connected_clients[room].add(client_id)
        join_room(room)
    
    def _remove_client(self, client_id: str):
        """Remove client from all rooms and subscriptions."""
        # Remove from rooms
        for room, clients in list(self.connected_clients.items()):
            if client_id in clients:
                clients.discard(client_id)
                if not clients:
                    del self.connected_clients[room]
        
        # Remove subscriptions
        if client_id in self.client_subscriptions:
            del self.client_subscriptions[client_id]
    
    def _get_available_rooms(self) -> list:
        """Get list of available rooms/topics."""
        return [
            'quantum_states',
            'simulation_updates',
            'measurement_results',
            'optimization_progress',
            'system_metrics'
        ]
    
    def _send_initial_state(self, client_id: str, subscription_type: str, entity_id: Optional[str]):
        """Send initial state to newly subscribed client."""
        if subscription_type == 'quantum_states' and entity_id:
            if entity_id in self.quantum_state_cache:
                self.socketio.emit('quantum_state_update', 
                                 self.quantum_state_cache[entity_id],
                                 room=client_id)
        elif subscription_type == 'simulation_updates' and entity_id:
            if entity_id in self.active_simulations:
                self.socketio.emit('simulation_status',
                                 self.active_simulations[entity_id],
                                 room=client_id)
    
    async def _perform_quantum_measurement(self, client_id: str, entity_id: str, measurement_type: str):
        """Perform quantum measurement and send result."""
        try:
            # Simulate measurement delay
            await asyncio.sleep(0.5)
            
            # Mock measurement result
            result = {
                'entity_id': entity_id,
                'measurement_type': measurement_type,
                'result': {
                    'outcome': '01',  # Mock outcome
                    'probability': 0.25,
                    'basis_state': '|01⟩',
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
            # Send to requesting client
            self.socketio.emit('measurement_result', result, room=client_id)
            
            # Broadcast to subscribers
            room = f"measurement_results:{entity_id}"
            self.socketio.emit('measurement_result', result, room=room)
            
        except Exception as e:
            logger.error(f"Measurement failed: {e}")
            self.socketio.emit('error', 
                             {'message': f'Measurement failed: {str(e)}'},
                             room=client_id)
    
    def _start_simulation(self, client_id: str, simulation_id: str, params: Dict[str, Any]):
        """Start a new simulation."""
        if simulation_id in self.active_simulations:
            self.socketio.emit('error',
                             {'message': f'Simulation {simulation_id} already running'},
                             room=client_id)
            return
        
        # Create simulation entry
        self.active_simulations[simulation_id] = {
            'id': simulation_id,
            'status': 'running',
            'started_at': datetime.utcnow().isoformat(),
            'params': params,
            'progress': 0,
            'client_id': client_id
        }
        
        # Start background task for simulation
        self.socketio.start_background_task(
            self._run_simulation,
            simulation_id
        )
        
        # Send confirmation
        self.socketio.emit('simulation_started',
                         {'simulation_id': simulation_id},
                         room=client_id)
    
    async def _run_simulation(self, simulation_id: str):
        """Run simulation and send progress updates."""
        try:
            for progress in range(0, 101, 10):
                if simulation_id not in self.active_simulations:
                    break  # Simulation stopped
                
                if self.active_simulations[simulation_id]['status'] == 'paused':
                    await asyncio.sleep(0.1)
                    continue
                
                # Update progress
                self.active_simulations[simulation_id]['progress'] = progress
                
                # Send progress update
                update = {
                    'simulation_id': simulation_id,
                    'progress': progress,
                    'status': 'running',
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Broadcast to simulation room
                room = f"simulation_updates:{simulation_id}"
                self.socketio.emit('simulation_progress', update, room=room)
                
                # Also send to initiating client
                client_id = self.active_simulations[simulation_id].get('client_id')
                if client_id:
                    self.socketio.emit('simulation_progress', update, room=client_id)
                
                await asyncio.sleep(1)  # Simulate work
            
            # Mark as completed
            if simulation_id in self.active_simulations:
                self.active_simulations[simulation_id]['status'] = 'completed'
                self.active_simulations[simulation_id]['progress'] = 100
                
                # Send completion
                completion = {
                    'simulation_id': simulation_id,
                    'status': 'completed',
                    'results': {
                        'success': True,
                        'data': {'mock': 'results'}
                    },
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                room = f"simulation_updates:{simulation_id}"
                self.socketio.emit('simulation_completed', completion, room=room)
                
                # Clean up after delay
                await asyncio.sleep(60)
                if simulation_id in self.active_simulations:
                    del self.active_simulations[simulation_id]
                    
        except Exception as e:
            logger.error(f"Simulation {simulation_id} failed: {e}")
            if simulation_id in self.active_simulations:
                self.active_simulations[simulation_id]['status'] = 'error'
                self.active_simulations[simulation_id]['error'] = str(e)
    
    def _stop_simulation(self, simulation_id: str):
        """Stop a running simulation."""
        if simulation_id in self.active_simulations:
            self.active_simulations[simulation_id]['status'] = 'stopped'
            self.socketio.emit('simulation_stopped',
                             {'simulation_id': simulation_id},
                             room=f"simulation_updates:{simulation_id}")
    
    def _pause_simulation(self, simulation_id: str):
        """Pause a running simulation."""
        if simulation_id in self.active_simulations:
            self.active_simulations[simulation_id]['status'] = 'paused'
            self.socketio.emit('simulation_paused',
                             {'simulation_id': simulation_id},
                             room=f"simulation_updates:{simulation_id}")
    
    def _resume_simulation(self, simulation_id: str):
        """Resume a paused simulation."""
        if simulation_id in self.active_simulations:
            self.active_simulations[simulation_id]['status'] = 'running'
            self.socketio.emit('simulation_resumed',
                             {'simulation_id': simulation_id},
                             room=f"simulation_updates:{simulation_id}")
    
    def broadcast_quantum_update(self, entity_id: str, state_data: Dict[str, Any]):
        """Broadcast quantum state update to subscribed clients."""
        # Cache the state
        self.quantum_state_cache[entity_id] = state_data
        
        # Broadcast to subscribers
        room = f"quantum_states:{entity_id}"
        self.socketio.emit('quantum_state_update', state_data, room=room)
        
        # Also broadcast to general quantum_states room
        self.socketio.emit('quantum_state_update', state_data, room='quantum_states')
    
    def broadcast_metric(self, metric_type: str, metric_data: Dict[str, Any]):
        """Broadcast system metric to subscribed clients."""
        self.socketio.emit('metric_update', {
            'type': metric_type,
            'data': metric_data,
            'timestamp': datetime.utcnow().isoformat()
        }, room='system_metrics')

def create_websocket_handler(app):
    """Create and configure WebSocket handler for Flask app."""
    # Configure Socket.IO
    socketio = SocketIO(
        app,
        cors_allowed_origins=app.config.get('WEBSOCKET_CORS_ORIGINS', '*'),
        async_mode='threading',
        logger=True,
        engineio_logger=False
    )
    
    # Create handler
    handler = QuantumWebSocketHandler(socketio)
    
    # Store reference in app
    app.quantum_ws = handler
    app.socketio = socketio
    
    logger.info("WebSocket handler initialized")
    
    return socketio, handler