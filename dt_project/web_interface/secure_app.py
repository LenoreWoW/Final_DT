"""
Secure Flask Web Application
Enhanced version with comprehensive security features.
"""

import os
import logging
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, request, jsonify, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
import structlog

# Import our secure configuration
from dt_project.config.secure_config import get_config, validate_input, sanitize_input

# Configure structured logging
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

def create_secure_app() -> Flask:
    """Create Flask application with comprehensive security."""
    app = Flask(__name__)
    
    # Load secure configuration
    config = get_config()
    
    # Validate configuration before starting
    if not config.validate_config():
        logger.error("Invalid configuration detected")
        raise ValueError("Configuration validation failed")
    
    # Configure Flask with secure settings
    app.config.update({
        'SECRET_KEY': config.get('SECRET_KEY'),
        'SESSION_COOKIE_SECURE': config.get('SECURITY.SECURE_COOKIES'),
        'SESSION_COOKIE_HTTPONLY': True,
        'SESSION_COOKIE_SAMESITE': 'Lax',
        'PERMANENT_SESSION_LIFETIME': timedelta(
            minutes=config.get('SECURITY.SESSION_LIFETIME_MINUTES')
        ),
        'WTF_CSRF_ENABLED': config.get('SECURITY.CSRF_PROTECTION'),
        'PREFERRED_URL_SCHEME': 'https' if config.is_production() else 'http',
    })
    
    # Initialize CORS with secure settings
    CORS(app, origins=config.get('SECURITY.CORS_ORIGINS'))
    
    # Initialize rate limiter
    limiter = Limiter(
        app,
        key_func=get_remote_address,
        default_limits=[config.get('RATE_LIMITS.DEFAULT')]
    )
    
    # Security headers middleware
    @app.after_request
    def add_security_headers(response):
        """Add security headers to all responses."""
        response.headers.update({
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'",
            'Referrer-Policy': 'strict-origin-when-cross-origin'
        })
        return response
    
    # Request logging middleware
    @app.before_request
    def log_request():
        """Log all incoming requests."""
        logger.info(
            "Request received",
            method=request.method,
            path=request.path,
            remote_addr=request.remote_addr,
            user_agent=request.headers.get('User-Agent', 'Unknown')
        )
    
    # Input validation decorator
    def validate_json_input(required_fields=None, optional_fields=None):
        """Decorator to validate JSON input."""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                if not request.is_json:
                    logger.warning("Non-JSON request received", path=request.path)
                    return jsonify({'error': 'Request must be JSON'}), 400
                
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'Empty request body'}), 400
                
                # Validate required fields
                if required_fields:
                    missing = [f for f in required_fields if f not in data]
                    if missing:
                        logger.warning("Missing required fields", fields=missing)
                        return jsonify({'error': f'Missing required fields: {missing}'}), 400
                
                # Sanitize all string inputs
                sanitized_data = {}
                all_fields = (required_fields or []) + (optional_fields or [])
                
                for field in all_fields:
                    if field in data:
                        value = data[field]
                        if isinstance(value, str):
                            sanitized_data[field] = sanitize_input(value)
                        else:
                            sanitized_data[field] = value
                
                # Add sanitized data to request context
                request.validated_data = sanitized_data
                return f(*args, **kwargs)
                
            return decorated_function
        return decorator
    
    # Health check endpoint (no rate limiting)
    @app.route('/health')
    def health_check():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '2.0.0'
        })
    
    # API endpoints with security
    @app.route('/api/simulate', methods=['POST'])
    @limiter.limit(config.get('RATE_LIMITS.API_CALLS'))
    @validate_json_input(
        required_fields=['athlete_profile', 'route_data'],
        optional_fields=['environmental_conditions', 'equipment_load']
    )
    def secure_simulate():
        """Secure simulation endpoint with input validation."""
        try:
            data = request.validated_data
            
            # Additional validation for coordinates
            if 'route_data' in data:
                for point in data['route_data']:
                    if 'latitude' in point:
                        if not validate_input(str(point['latitude']), 'coordinate'):
                            return jsonify({'error': 'Invalid latitude value'}), 400
                    if 'longitude' in point:
                        if not validate_input(str(point['longitude']), 'coordinate'):
                            return jsonify({'error': 'Invalid longitude value'}), 400
            
            # Log simulation request (without sensitive data)
            logger.info(
                "Simulation request",
                route_points=len(data.get('route_data', [])),
                has_equipment=bool(data.get('equipment_load'))
            )
            
            # Process simulation (implementation would go here)
            result = {
                'simulation_id': f"sim_{datetime.utcnow().timestamp()}",
                'status': 'completed',
                'message': 'Simulation completed successfully'
            }
            
            return jsonify(result)
            
        except Exception as e:
            logger.error("Simulation error", error=str(e))
            return jsonify({'error': 'Internal server error'}), 500
    
    @app.route('/api/quantum/monte-carlo', methods=['POST'])
    @limiter.limit(config.get('RATE_LIMITS.QUANTUM_OPERATIONS'))
    @validate_json_input(
        required_fields=['function_params', 'iterations'],
        optional_fields=['distribution_type']
    )
    def secure_quantum_monte_carlo():
        """Secure quantum Monte Carlo endpoint."""
        try:
            data = request.validated_data
            
            # Validate iterations count
            iterations = data.get('iterations', 0)
            if not isinstance(iterations, int) or iterations <= 0 or iterations > 10000:
                return jsonify({'error': 'Iterations must be between 1 and 10000'}), 400
            
            logger.info(
                "Quantum Monte Carlo request",
                iterations=iterations,
                distribution=data.get('distribution_type', 'uniform')
            )
            
            # Process quantum operation (implementation would go here)
            result = {
                'job_id': f"quantum_{datetime.utcnow().timestamp()}",
                'status': 'submitted',
                'estimated_completion': (datetime.utcnow() + timedelta(minutes=5)).isoformat()
            }
            
            return jsonify(result)
            
        except Exception as e:
            logger.error("Quantum Monte Carlo error", error=str(e))
            return jsonify({'error': 'Internal server error'}), 500
    
    # Error handlers
    @app.errorhandler(429)
    def ratelimit_handler(e):
        """Handle rate limit exceeded."""
        logger.warning(
            "Rate limit exceeded",
            remote_addr=request.remote_addr,
            path=request.path
        )
        return jsonify({'error': 'Rate limit exceeded'}), 429
    
    @app.errorhandler(400)
    def bad_request_handler(e):
        """Handle bad requests."""
        return jsonify({'error': 'Bad request'}), 400
    
    @app.errorhandler(500)
    def internal_error_handler(e):
        """Handle internal server errors."""
        logger.error("Internal server error", error=str(e))
        return jsonify({'error': 'Internal server error'}), 500
    
    # Add configuration endpoint for debugging (development only)
    if not config.is_production():
        @app.route('/debug/config')
        def debug_config():
            """Debug endpoint to view configuration (development only)."""
            safe_config = {
                'environment': config.get('FLASK_ENV'),
                'security_features': {
                    'rate_limiting': config.get('SECURITY.RATE_LIMIT_ENABLED'),
                    'cors_enabled': True,
                    'secure_cookies': config.get('SECURITY.SECURE_COOKIES'),
                    'csrf_protection': config.get('SECURITY.CSRF_PROTECTION'),
                }
            }
            return jsonify(safe_config)
    
    logger.info("Secure Flask application initialized successfully")
    return app

# Application factory function
def create_app(config_name: str = None) -> Flask:
    """Create application with specified configuration."""
    return create_secure_app()

if __name__ == '__main__':
    app = create_secure_app()
    config = get_config()
    
    app.run(
        host=config.get('HOST', '127.0.0.1'),
        port=config.get('PORT', 5000),
        debug=not config.is_production()
    )