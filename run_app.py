#!/usr/bin/env python3
"""
Simple Flask app runner with all endpoints
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set minimal environment variables
os.environ.setdefault('SECRET_KEY', 'quantum-trail-secret-key-2024')
os.environ.setdefault('DATABASE_URL', 'sqlite:///quantum_trail.db')
os.environ.setdefault('REDIS_URL', 'redis://localhost:6379/0')
os.environ.setdefault('IBM_QUANTUM_TOKEN', 'dummy-token-for-testing')

from flask import Flask, jsonify
from flask_cors import CORS

def create_app():
    """Create Flask app with all routes"""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
    
    # Enable CORS
    CORS(app)
    
    try:
        # Import and register route blueprints
        from dt_project.web_interface.routes.main_routes import create_main_routes
        from dt_project.web_interface.routes.admin_routes import create_admin_routes  
        from dt_project.web_interface.routes.docs_routes import create_docs_routes
        
        # Register blueprints
        main_bp = create_main_routes()
        admin_bp = create_admin_routes()
        docs_bp = create_docs_routes()
        
        app.register_blueprint(main_bp)
        app.register_blueprint(admin_bp)
        app.register_blueprint(docs_bp)
        
        # Health endpoint
        @app.route('/health')
        def health():
            return jsonify({
                'status': 'healthy',
                'message': 'Quantum Trail Platform is operational',
                'endpoints': {
                    'main': '/',
                    'dashboard': '/dashboard',
                    'admin': '/admin/',
                    'docs': '/docs/',
                    'graphql': '/graphql',
                    'playground': '/quantum-playground'
                }
            })
        
        # API endpoints (mock for now)
        @app.route('/api/quantum_twins', methods=['GET', 'POST'])
        def quantum_twins():
            if request.method == 'POST':
                return jsonify({
                    'id': 1,
                    'name': 'New Quantum Twin',
                    'status': 'created',
                    'message': 'Quantum twin created successfully'
                })
            return jsonify({
                'twins': [
                    {
                        'id': 1,
                        'name': 'Sample Portfolio Twin',
                        'type': 'portfolio_optimization',
                        'status': 'active'
                    }
                ]
            })
        
        @app.route('/api/quantum_twins/<int:twin_id>/simulate', methods=['POST'])
        def simulate_twin(twin_id):
            return jsonify({
                'id': f'sim_{twin_id}',
                'status': 'completed',
                'result': {
                    'optimal_weights': [0.25, 0.35, 0.20, 0.20],
                    'expected_return': 0.138,
                    'portfolio_risk': 0.089
                }
            })
        
        print("✅ All routes registered successfully")
        return app
        
    except Exception as e:
        print(f"❌ Failed to create app: {e}")
        import traceback
        traceback.print_exc()
        
        # Return minimal app if route import fails
        app = Flask(__name__)
        CORS(app)
        
        @app.route('/')
        def error_page():
            return f"""
            <h1>⚠️ Quantum Trail Platform - Setup Required</h1>
            <p>Some modules couldn't be loaded: {str(e)}</p>
            <p>The application is running on port 8000 but needs additional setup.</p>
            <hr>
            <p>Available: <a href="/health">Health Check</a></p>
            """
        
        @app.route('/health')
        def health():
            return jsonify({'status': 'partial', 'error': str(e)})
        
        return app

if __name__ == '__main__':
    from flask import request
    
    print("🚀 Starting Quantum Trail Platform...")
    print("=" * 50)
    
    app = create_app()
    
    if app:
        print("\n📱 Starting server on http://localhost:8000")
        print("🌐 Open your browser and visit:")
        print("   • Main page: http://localhost:8000")
        print("   • Dashboard: http://localhost:8000/dashboard")
        print("   • Admin: http://localhost:8000/admin")
        print("   • API Docs: http://localhost:8000/docs")
        print("   • GraphQL: http://localhost:8000/graphql")
        print("   • Playground: http://localhost:8000/quantum-playground")
        print("\n💡 Press Ctrl+C to stop the server")
        print("=" * 50)
        
        app.run(host='0.0.0.0', port=8000, debug=True)
    else:
        print("❌ Failed to start application")