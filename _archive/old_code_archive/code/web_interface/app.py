"""
Main Flask application for Quantum Trail.
This is the central application that integrates all components.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, g
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import configuration and utilities
from dt_project.config.secure_config import get_config, validate_input, sanitize_input
from dt_project.web_interface.secure_app import create_secure_app
from dt_project.web_interface.websocket_handler import create_websocket_handler
from dt_project.web_interface.graphql_schema import create_graphql_view
from dt_project.monitoring.metrics import setup_metrics_endpoint, metrics, get_system_health
from dt_project.models import Base

# Import quantum components
from dt_project.quantum.async_quantum_backend import AsyncQuantumProcessor
from dt_project.quantum.quantum_digital_twin import QuantumDigitalTwin, QuantumTwinType
from dt_project.quantum.quantum_optimization import create_quantum_optimizer

# Import routes
from dt_project.web_interface.routes.simulation_routes import create_simulation_routes
from dt_project.web_interface.routes.quantum_routes import create_quantum_routes

# Configure logging
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

class QuantumTrailApplication:
    """Main Quantum Trail application class."""
    
    def __init__(self, config=None):
        """Initialize the Quantum Trail application."""
        self.config = config or get_config()
        self.app = None
        self.db = None
        self.db_session = None
        self.quantum_processor = None
        self.socketio = None
        self.ws_handler = None
        self.active_twins = {}  # Store active quantum twins
        
        # Initialize application
        self._create_app()
        self._setup_database()
        self._setup_quantum_processing()
        self._setup_websockets()
        self._setup_graphql()
        self._setup_metrics()
        self._setup_routes()
        self._setup_error_handlers()
        
        logger.info("Quantum Trail application initialized successfully")
    
    def _create_app(self):
        """Create and configure Flask application."""
        self.app = create_secure_app()
        
        # Additional configuration
        self.app.config.update({
            'SQLALCHEMY_DATABASE_URI': self.config.get('DATABASE.URL'),
            'SQLALCHEMY_TRACK_MODIFICATIONS': False,
            'SQLALCHEMY_ENGINE_OPTIONS': {
                'pool_size': self.config.get('DATABASE.POOL_SIZE'),
                'max_overflow': self.config.get('DATABASE.MAX_OVERFLOW'),
                'pool_timeout': self.config.get('DATABASE.POOL_TIMEOUT'),
                'pool_pre_ping': True,
                'pool_recycle': 3600,
            }
        })
        
        # Store references for access by routes
        self.app.quantum_app = self
        
        logger.info("Flask application created with secure configuration")
    
    def _setup_database(self):
        """Set up database connection and models."""
        try:
            # Create SQLAlchemy instance
            self.db = SQLAlchemy(self.app)
            
            # Create engine and session factory
            engine = create_engine(
                self.config.get('DATABASE.URL'),
                **self.app.config['SQLALCHEMY_ENGINE_OPTIONS']
            )
            
            # Create tables
            Base.metadata.create_all(engine)
            
            # Create session factory
            SessionFactory = sessionmaker(bind=engine)
            self.db_session = SessionFactory()
            
            # Add database utilities to app context
            @self.app.before_request
            def before_request():
                g.db_session = self.db_session
                g.start_time = datetime.utcnow()
            
            @self.app.teardown_appcontext
            def close_db_session(error):
                if hasattr(g, 'db_session'):
                    if error:
                        g.db_session.rollback()
                    else:
                        try:
                            g.db_session.commit()
                        except Exception as e:
                            logger.error(f"Database commit failed: {e}")
                            g.db_session.rollback()
                    g.db_session.close()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def _setup_quantum_processing(self):
        """Set up quantum processing components."""
        try:
            # Initialize quantum processor
            self.quantum_processor = AsyncQuantumProcessor()
            
            # Store in app context for access by routes
            self.app.quantum_processor = self.quantum_processor
            
            logger.info(f"Quantum processor initialized with backends: {list(self.quantum_processor.backends.keys())}")
            
        except Exception as e:
            logger.error(f"Quantum processing initialization failed: {e}")
            # Continue without quantum processing for development
            self.quantum_processor = None
    
    def _setup_websockets(self):
        """Set up WebSocket support for real-time updates."""
        try:
            if self.config.get('ENABLE_WEBSOCKET', True):
                self.socketio, self.ws_handler = create_websocket_handler(self.app)
                logger.info("WebSocket support enabled")
            else:
                logger.info("WebSocket support disabled")
                
        except Exception as e:
            logger.error(f"WebSocket setup failed: {e}")
            self.socketio = None
            self.ws_handler = None
    
    def _setup_graphql(self):
        """Set up GraphQL API endpoint."""
        try:
            if self.config.get('ENABLE_GRAPHQL', True):
                create_graphql_view(self.app, self.db_session, self.quantum_processor)
                logger.info("GraphQL endpoint configured at /graphql")
            else:
                logger.info("GraphQL disabled")
                
        except Exception as e:
            logger.error(f"GraphQL setup failed: {e}")
    
    def _setup_metrics(self):
        """Set up Prometheus metrics collection."""
        try:
            if self.config.get('ENABLE_METRICS', True):
                setup_metrics_endpoint(self.app)
                logger.info("Metrics collection enabled at /metrics")
            else:
                logger.info("Metrics collection disabled")
                
        except Exception as e:
            logger.error(f"Metrics setup failed: {e}")
    
    def _setup_routes(self):
        """Set up API routes."""
        try:
            # Import route blueprints
            from dt_project.web_interface.routes.main_routes import create_main_routes
            from dt_project.web_interface.routes.admin_routes import create_admin_routes
            from dt_project.web_interface.routes.docs_routes import create_docs_routes
            
            # Register main routes (includes home page, dashboard, GraphQL playground)
            main_bp = create_main_routes()
            self.app.register_blueprint(main_bp)
            
            # Register admin routes
            admin_bp = create_admin_routes()
            self.app.register_blueprint(admin_bp)
            
            # Register documentation routes
            docs_bp = create_docs_routes()
            self.app.register_blueprint(docs_bp)
            
            # Health check endpoint
            @self.app.route('/health')
            def health_check():
                """Health check endpoint for monitoring."""
                health_data = get_system_health()
                
                # Add application-specific health checks
                health_data.update({
                    'quantum_processor_available': self.quantum_processor is not None,
                    'database_connected': self.db_session is not None,
                    'websocket_enabled': self.socketio is not None,
                    'active_quantum_twins': len(self.active_twins),
                })
                
                status_code = 200 if health_data['status'] == 'healthy' else 503
                return jsonify(health_data), status_code
            
            # Create simulation routes
            simulation_bp = create_simulation_routes(self)
            self.app.register_blueprint(simulation_bp, url_prefix='/api/simulation')
            
            # Create quantum routes  
            quantum_bp = create_quantum_routes(self)
            self.app.register_blueprint(quantum_bp, url_prefix='/api/quantum')
            
            # Create quantum lab routes
            try:
                from dt_project.web_interface.routes.quantum_lab_routes import quantum_lab_bp
                self.app.register_blueprint(quantum_lab_bp)
                logger.info("Quantum lab routes registered")
            except ImportError as e:
                logger.warning(f"Quantum lab routes not available: {e}")
            
            logger.info("All routes configured successfully")
            
        except Exception as e:
            logger.error(f"Routes setup failed: {e}")
            raise
    
    def _setup_error_handlers(self):
        """Set up global error handlers."""
        
        @self.app.errorhandler(404)
        def not_found(error):
            """Handle 404 errors."""
            return jsonify({
                'error': 'Not Found',
                'message': 'The requested resource was not found',
                'status_code': 404
            }), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            """Handle 500 errors."""
            logger.error(f"Internal server error: {error}")
            
            if hasattr(g, 'db_session'):
                g.db_session.rollback()
            
            return jsonify({
                'error': 'Internal Server Error',
                'message': 'An internal server error occurred',
                'status_code': 500
            }), 500
        
        @self.app.errorhandler(429)
        def ratelimit_handler(e):
            """Handle rate limiting errors."""
            return jsonify({
                'error': 'Rate Limit Exceeded',
                'message': 'Request rate limit exceeded. Please try again later.',
                'status_code': 429,
                'retry_after': getattr(e, 'retry_after', None)
            }), 429
        
        @self.app.errorhandler(ValidationError)
        def validation_error(error):
            """Handle validation errors."""
            return jsonify({
                'error': 'Validation Error',
                'message': str(error),
                'status_code': 400
            }), 400
    
    async def create_quantum_twin(self, entity_id: str, twin_type: QuantumTwinType, 
                                initial_state: dict, n_qubits: int = 4) -> QuantumDigitalTwin:
        """Create a new quantum digital twin."""
        try:
            # Create quantum twin
            twin = QuantumDigitalTwin(entity_id, twin_type)
            await twin.initialize(initial_state, n_qubits)
            
            # Store in active twins
            self.active_twins[entity_id] = twin
            
            # Broadcast creation via WebSocket
            if self.ws_handler:
                self.ws_handler.broadcast_quantum_update(
                    entity_id,
                    {
                        'event': 'twin_created',
                        'entity_id': entity_id,
                        'twin_type': twin_type.value,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                )
            
            # Record metrics
            if metrics:
                metrics.record_quantum_circuit(
                    backend='simulator',
                    n_qubits=n_qubits,
                    depth=1
                )
            
            logger.info(f"Created quantum twin {entity_id} of type {twin_type.value}")
            return twin
            
        except Exception as e:
            logger.error(f"Failed to create quantum twin {entity_id}: {e}")
            raise
    
    async def update_quantum_twin(self, entity_id: str, sensor_data: dict) -> dict:
        """Update quantum twin with sensor data."""
        try:
            if entity_id not in self.active_twins:
                raise ValueError(f"Quantum twin {entity_id} not found")
            
            twin = self.active_twins[entity_id]
            result = await twin.update_from_sensors(sensor_data)
            
            # Broadcast update via WebSocket
            if self.ws_handler:
                classical_info = twin.state_manager.get_classical_correlates(entity_id)
                self.ws_handler.broadcast_quantum_update(
                    entity_id,
                    {
                        'event': 'state_updated',
                        'entity_id': entity_id,
                        'classical_correlates': classical_info,
                        'fidelity': result.get('current_fidelity'),
                        'timestamp': datetime.utcnow().isoformat()
                    }
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to update quantum twin {entity_id}: {e}")
            raise
    
    def get_quantum_twin(self, entity_id: str) -> QuantumDigitalTwin:
        """Get active quantum twin."""
        if entity_id not in self.active_twins:
            raise ValueError(f"Quantum twin {entity_id} not found")
        return self.active_twins[entity_id]
    
    def list_active_twins(self) -> list:
        """List all active quantum twins."""
        return [
            {
                'entity_id': entity_id,
                'twin_type': twin.twin_type.value,
                'created_at': twin.created_at.isoformat(),
                'update_count': twin.update_count
            }
            for entity_id, twin in self.active_twins.items()
        ]
    
    async def run_quantum_optimization(self, algorithm: str, problem_data: dict) -> dict:
        """Run quantum optimization algorithm."""
        try:
            # Create optimizer
            n_qubits = problem_data.get('n_qubits', 4)
            optimizer = create_quantum_optimizer(algorithm, n_qubits)
            
            # Track with metrics
            with metrics.track_quantum_job('simulator', 'optimization'):
                if algorithm.lower() == 'qaoa' and problem_data.get('problem_type') == 'maxcut':
                    edges = problem_data['parameters']['edges']
                    result = await optimizer.optimize_maxcut(edges)
                    
                elif algorithm.lower() == 'vqe':
                    # For now, use molecular hamiltonian
                    molecule_data = problem_data.get('parameters', {})
                    result = await optimizer.optimize_molecular_hamiltonian(molecule_data)
                    
                elif algorithm.lower() == 'hybrid':
                    if problem_data.get('problem_type') == 'portfolio':
                        assets = problem_data['parameters']['assets']
                        risk_tolerance = problem_data['parameters'].get('risk_tolerance', 0.5)
                        result = await optimizer.optimize_portfolio(assets, risk_tolerance)
                    else:
                        raise ValueError(f"Unsupported hybrid problem type: {problem_data.get('problem_type')}")
                        
                else:
                    raise ValueError(f"Unsupported algorithm/problem combination")
            
            # Record metrics
            if metrics:
                metrics.record_optimization_result(
                    algorithm=algorithm,
                    iterations=result.n_function_evaluations,
                    advantage=result.quantum_advantage
                )
            
            # Convert to JSON-serializable format
            return {
                'optimal_value': float(result.optimal_value),
                'optimal_parameters': result.optimal_parameters.tolist(),
                'convergence_history': result.convergence_history,
                'quantum_advantage': float(result.quantum_advantage),
                'execution_time': float(result.execution_time),
                'metadata': result.metadata
            }
            
        except Exception as e:
            logger.error(f"Quantum optimization failed: {e}")
            raise
    
    def run(self, host='127.0.0.1', port=5000, debug=False, **kwargs):
        """Run the application."""
        try:
            logger.info(f"Starting Quantum Trail application on {host}:{port}")
            logger.info(f"Debug mode: {debug}")
            logger.info(f"Active backends: {list(self.quantum_processor.backends.keys()) if self.quantum_processor else []}")
            
            if self.socketio:
                # Run with Socket.IO support
                self.socketio.run(
                    self.app,
                    host=host,
                    port=port,
                    debug=debug,
                    **kwargs
                )
            else:
                # Run standard Flask app
                self.app.run(
                    host=host,
                    port=port,
                    debug=debug,
                    **kwargs
                )
                
        except Exception as e:
            logger.error(f"Failed to start application: {e}")
            raise

# Custom exceptions
class ValidationError(Exception):
    """Custom validation error."""
    pass

# Factory function to create application
def create_app(config=None):
    """Factory function to create Flask application."""
    try:
        quantum_app = QuantumTrailApplication(config)
        return quantum_app.app
    except Exception as e:
        logger.error(f"Failed to create application: {e}")
        raise

# Application instance for WSGI servers
def create_wsgi_app():
    """Create WSGI application instance."""
    config = get_config()
    quantum_app = QuantumTrailApplication(config)
    return quantum_app.app

# Main entry point
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Quantum Trail Application')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=int(os.environ.get('FLASK_RUN_PORT', 8000)), help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--validate-env', action='store_true', help='Validate environment first')
    
    args = parser.parse_args()
    
    # Validate environment if requested
    if args.validate_env:
        from validate_env import EnvValidator
        validator = EnvValidator()
        if not validator.validate():
            sys.exit(1)
    
    try:
        # Create and run application
        quantum_app = QuantumTrailApplication()
        quantum_app.run(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        sys.exit(1)