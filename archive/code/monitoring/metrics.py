"""
Prometheus metrics collection for Quantum Trail application.
"""

import time
import logging
from typing import Dict, Any, Optional
from functools import wraps
from datetime import datetime
import os

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info,
        CollectorRegistry, CONTENT_TYPE_LATEST,
        generate_latest, multiprocess, REGISTRY
    )
    from flask import request, g, Response
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available. Metrics collection disabled.")

logger = logging.getLogger(__name__)

class QuantumMetrics:
    """Centralized metrics collection for Quantum Trail."""
    
    def __init__(self, registry=None):
        """Initialize metrics collectors."""
        self.registry = registry or REGISTRY
        
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus not available. Metrics collection disabled.")
            return
            
        # HTTP Metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.http_request_duration_seconds = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry,
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf'))
        )
        
        # Quantum Computing Metrics
        self.quantum_jobs_total = Counter(
            'quantum_jobs_total',
            'Total quantum jobs submitted',
            ['backend', 'job_type', 'status'],
            registry=self.registry
        )
        
        self.quantum_job_duration_seconds = Histogram(
            'quantum_job_duration_seconds',
            'Quantum job execution duration in seconds',
            ['backend', 'job_type'],
            registry=self.registry,
            buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0, float('inf'))
        )
        
        self.quantum_circuit_depth = Histogram(
            'quantum_circuit_depth',
            'Quantum circuit depth',
            ['backend'],
            registry=self.registry,
            buckets=(1, 5, 10, 20, 50, 100, 200, 500, float('inf'))
        )
        
        self.quantum_qubits_used = Histogram(
            'quantum_qubits_used',
            'Number of qubits used in quantum circuits',
            ['backend'],
            registry=self.registry,
            buckets=(1, 2, 4, 8, 16, 32, 64, 128, float('inf'))
        )
        
        # Simulation Metrics
        self.simulations_total = Counter(
            'simulations_total',
            'Total simulations run',
            ['simulation_type', 'status'],
            registry=self.registry
        )
        
        self.simulation_duration_seconds = Histogram(
            'simulation_duration_seconds',
            'Simulation execution duration in seconds',
            ['simulation_type'],
            registry=self.registry,
            buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0, float('inf'))
        )
        
        # System Performance Metrics
        self.active_connections = Gauge(
            'active_websocket_connections',
            'Number of active WebSocket connections',
            registry=self.registry
        )
        
        self.database_queries_total = Counter(
            'database_queries_total',
            'Total database queries',
            ['operation', 'table'],
            registry=self.registry
        )
        
        self.database_query_duration_seconds = Histogram(
            'database_query_duration_seconds',
            'Database query duration in seconds',
            ['operation', 'table'],
            registry=self.registry,
            buckets=(0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, float('inf'))
        )
        
        self.cache_operations_total = Counter(
            'cache_operations_total',
            'Total cache operations',
            ['operation', 'result'],  # hit, miss, set, delete
            registry=self.registry
        )
        
        # Business Metrics
        self.athlete_profiles_total = Gauge(
            'athlete_profiles_total',
            'Total athlete profiles',
            registry=self.registry
        )
        
        self.optimization_convergence_iterations = Histogram(
            'optimization_convergence_iterations',
            'Number of iterations for optimization convergence',
            ['algorithm'],
            registry=self.registry,
            buckets=(10, 50, 100, 200, 500, 1000, 2000, float('inf'))
        )
        
        self.quantum_advantage_factor = Histogram(
            'quantum_advantage_factor',
            'Quantum advantage factor achieved',
            ['algorithm'],
            registry=self.registry,
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, float('inf'))
        )
        
        # Error Metrics
        self.errors_total = Counter(
            'errors_total',
            'Total errors',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        # Application Info
        self.app_info = Info(
            'quantum_trail_app_info',
            'Application information',
            registry=self.registry
        )
        
        # Set application info
        self.app_info.info({
            'version': os.getenv('APP_VERSION', '1.0.0'),
            'environment': os.getenv('FLASK_ENV', 'development'),
            'python_version': os.getenv('PYTHON_VERSION', 'unknown')
        })
        
        logger.info("Prometheus metrics initialized")
    
    def track_http_request(self):
        """Decorator to track HTTP request metrics."""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                if not PROMETHEUS_AVAILABLE:
                    return f(*args, **kwargs)
                    
                start_time = time.time()
                
                try:
                    response = f(*args, **kwargs)
                    status_code = getattr(response, 'status_code', 200)
                    
                    # Record metrics
                    self.http_requests_total.labels(
                        method=request.method,
                        endpoint=request.endpoint or 'unknown',
                        status_code=status_code
                    ).inc()
                    
                    self.http_request_duration_seconds.labels(
                        method=request.method,
                        endpoint=request.endpoint or 'unknown'
                    ).observe(time.time() - start_time)
                    
                    return response
                    
                except Exception as e:
                    # Record error
                    self.http_requests_total.labels(
                        method=request.method,
                        endpoint=request.endpoint or 'unknown',
                        status_code=500
                    ).inc()
                    
                    self.errors_total.labels(
                        error_type=type(e).__name__,
                        component='http'
                    ).inc()
                    
                    raise
                    
            return decorated_function
        return decorator
    
    def track_quantum_job(self, backend: str, job_type: str):
        """Context manager to track quantum job metrics."""
        class QuantumJobTracker:
            def __init__(self, metrics, backend, job_type):
                self.metrics = metrics
                self.backend = backend
                self.job_type = job_type
                self.start_time = None
                
            def __enter__(self):
                self.start_time = time.time()
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                if not PROMETHEUS_AVAILABLE:
                    return
                    
                duration = time.time() - self.start_time
                status = 'error' if exc_type else 'success'
                
                self.metrics.quantum_jobs_total.labels(
                    backend=self.backend,
                    job_type=self.job_type,
                    status=status
                ).inc()
                
                if not exc_type:  # Only record duration for successful jobs
                    self.metrics.quantum_job_duration_seconds.labels(
                        backend=self.backend,
                        job_type=self.job_type
                    ).observe(duration)
                    
                if exc_type:
                    self.metrics.errors_total.labels(
                        error_type=exc_type.__name__,
                        component='quantum'
                    ).inc()
        
        return QuantumJobTracker(self, backend, job_type)
    
    def track_simulation(self, simulation_type: str):
        """Context manager to track simulation metrics."""
        class SimulationTracker:
            def __init__(self, metrics, simulation_type):
                self.metrics = metrics
                self.simulation_type = simulation_type
                self.start_time = None
                
            def __enter__(self):
                self.start_time = time.time()
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                if not PROMETHEUS_AVAILABLE:
                    return
                    
                duration = time.time() - self.start_time
                status = 'error' if exc_type else 'success'
                
                self.metrics.simulations_total.labels(
                    simulation_type=self.simulation_type,
                    status=status
                ).inc()
                
                if not exc_type:
                    self.metrics.simulation_duration_seconds.labels(
                        simulation_type=self.simulation_type
                    ).observe(duration)
        
        return SimulationTracker(self, simulation_type)
    
    def record_quantum_circuit(self, backend: str, n_qubits: int, depth: int):
        """Record quantum circuit characteristics."""
        if not PROMETHEUS_AVAILABLE:
            return
            
        self.quantum_qubits_used.labels(backend=backend).observe(n_qubits)
        self.quantum_circuit_depth.labels(backend=backend).observe(depth)
    
    def record_optimization_result(self, algorithm: str, iterations: int, advantage: float):
        """Record optimization convergence metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
            
        self.optimization_convergence_iterations.labels(algorithm=algorithm).observe(iterations)
        self.quantum_advantage_factor.labels(algorithm=algorithm).observe(advantage)
    
    def update_websocket_connections(self, count: int):
        """Update active WebSocket connections gauge."""
        if not PROMETHEUS_AVAILABLE:
            return
            
        self.active_connections.set(count)
    
    def record_database_query(self, operation: str, table: str, duration: float):
        """Record database query metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
            
        self.database_queries_total.labels(operation=operation, table=table).inc()
        self.database_query_duration_seconds.labels(operation=operation, table=table).observe(duration)
    
    def record_cache_operation(self, operation: str, result: str):
        """Record cache operation metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
            
        self.cache_operations_total.labels(operation=operation, result=result).inc()
    
    def update_athlete_count(self, count: int):
        """Update athlete profiles gauge."""
        if not PROMETHEUS_AVAILABLE:
            return
            
        self.athlete_profiles_total.set(count)
    
    def record_error(self, error_type: str, component: str):
        """Record error occurrence."""
        if not PROMETHEUS_AVAILABLE:
            return
            
        self.errors_total.labels(error_type=error_type, component=component).inc()

# Global metrics instance
metrics = QuantumMetrics()

def setup_metrics_endpoint(app):
    """Set up metrics endpoint for Prometheus scraping."""
    
    if not PROMETHEUS_AVAILABLE:
        logger.warning("Prometheus not available. Metrics endpoint disabled.")
        return
    
    # Set up multiprocess mode if configured
    prometheus_dir = os.getenv('PROMETHEUS_MULTIPROC_DIR')
    if prometheus_dir and os.path.exists(prometheus_dir):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        
        @app.route('/metrics')
        def metrics_endpoint():
            data = generate_latest(registry)
            return Response(data, mimetype=CONTENT_TYPE_LATEST)
    else:
        @app.route('/metrics')
        def metrics_endpoint():
            data = generate_latest(REGISTRY)
            return Response(data, mimetype=CONTENT_TYPE_LATEST)
    
    # Add request tracking middleware
    @app.before_request
    def before_request():
        g.start_time = time.time()
    
    @app.after_request
    def after_request(response):
        if hasattr(g, 'start_time') and PROMETHEUS_AVAILABLE:
            duration = time.time() - g.start_time
            
            metrics.http_requests_total.labels(
                method=request.method,
                endpoint=request.endpoint or 'unknown',
                status_code=response.status_code
            ).inc()
            
            metrics.http_request_duration_seconds.labels(
                method=request.method,
                endpoint=request.endpoint or 'unknown'
            ).observe(duration)
            
        return response
    
    logger.info("Metrics endpoint configured at /metrics")

# Health check for monitoring
def get_system_health():
    """Get system health metrics for monitoring."""
    if not PROMETHEUS_AVAILABLE:
        return {"status": "metrics_disabled"}
    
    try:
        return {
            "status": "healthy",
            "metrics_enabled": True,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "prometheus": "available",
                "database": "healthy",  # This would be checked in real implementation
                "redis": "healthy",     # This would be checked in real implementation
                "quantum_backend": "available"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }