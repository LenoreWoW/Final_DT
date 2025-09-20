"""
Admin interface routes for Quantum Trail Platform
"""

from flask import Blueprint, render_template_string, request, jsonify, redirect, url_for, flash
from flask import current_app as app
from datetime import datetime, timedelta
import json

# Create admin blueprint
admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_bp.route('/')
def dashboard():
    """Admin dashboard main page"""
    try:
        # Get system stats
        stats = {
            'active_twins': 0,
            'total_simulations': 0,
            'total_users': 0,
            'system_health': 'healthy',
            'last_update': datetime.utcnow().isoformat()
        }
        
        # Try to get real stats from database
        try:
            from dt_project.database.models import QuantumTwin, User, SimulationResult
            from dt_project.database import db
            
            stats['active_twins'] = QuantumTwin.query.filter_by(is_active=True).count()
            stats['total_simulations'] = SimulationResult.query.count()
            stats['total_users'] = User.query.count()
        except Exception as e:
            app.logger.warning(f"Could not fetch database stats: {e}")
        
        return render_template_string(ADMIN_DASHBOARD_TEMPLATE, stats=stats)
        
    except Exception as e:
        app.logger.error(f"Admin dashboard error: {e}")
        return jsonify({'error': 'Admin dashboard temporarily unavailable'}), 500

@admin_bp.route('/system')
def system_status():
    """System status page"""
    try:
        system_info = {
            'timestamp': datetime.utcnow().isoformat(),
            'services': {
                'web_server': 'running',
                'database': 'unknown',
                'cache': 'unknown',
                'quantum_backend': 'unknown'
            },
            'performance': {
                'uptime': 'N/A',
                'memory_usage': 'N/A',
                'cpu_usage': 'N/A'
            }
        }
        
        return render_template_string(SYSTEM_STATUS_TEMPLATE, system_info=system_info)
        
    except Exception as e:
        app.logger.error(f"System status error: {e}")
        return jsonify({'error': 'System status temporarily unavailable'}), 500

@admin_bp.route('/quantum-twins')
def quantum_twins():
    """Quantum twins management page"""
    try:
        twins = []
        
        # Try to get twins from database
        try:
            from dt_project.database.models import QuantumTwin
            twins_query = QuantumTwin.query.all()
            twins = [
                {
                    'id': twin.id,
                    'name': twin.name,
                    'type': twin.twin_type,
                    'status': 'active' if twin.is_active else 'inactive',
                    'created': twin.created_at.isoformat() if twin.created_at else 'N/A',
                    'description': twin.description or 'No description'
                }
                for twin in twins_query
            ]
        except Exception as e:
            app.logger.warning(f"Could not fetch twins: {e}")
            
        return render_template_string(QUANTUM_TWINS_TEMPLATE, twins=twins)
        
    except Exception as e:
        app.logger.error(f"Quantum twins page error: {e}")
        return jsonify({'error': 'Quantum twins page temporarily unavailable'}), 500

@admin_bp.route('/api/stats')
def api_stats():
    """API endpoint for admin statistics"""
    try:
        stats = {
            'timestamp': datetime.utcnow().isoformat(),
            'active_twins': 0,
            'recent_simulations': 0,
            'system_health': 'healthy',
            'database_status': 'unknown'
        }
        
        # Try to get real stats
        try:
            from dt_project.database.models import QuantumTwin, SimulationResult
            
            stats['active_twins'] = QuantumTwin.query.filter_by(is_active=True).count()
            
            # Count simulations in last 24 hours
            yesterday = datetime.utcnow() - timedelta(days=1)
            stats['recent_simulations'] = SimulationResult.query.filter(
                SimulationResult.created_at >= yesterday
            ).count()
            
            stats['database_status'] = 'connected'
            
        except Exception as e:
            app.logger.warning(f"Could not fetch API stats: {e}")
            stats['database_status'] = 'error'
        
        return jsonify(stats)
        
    except Exception as e:
        app.logger.error(f"API stats error: {e}")
        return jsonify({'error': 'Stats unavailable'}), 500

# HTML Templates (inline for simplicity)
ADMIN_DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Trail - Admin Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .stat-card { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .stat-number { font-size: 2em; font-weight: bold; color: #3498db; }
        .stat-label { color: #666; margin-top: 10px; }
        .nav-links { margin: 20px 0; }
        .nav-links a { display: inline-block; margin-right: 15px; padding: 10px 15px; background: #3498db; color: white; text-decoration: none; border-radius: 3px; }
        .nav-links a:hover { background: #2980b9; }
        .status-healthy { color: #27ae60; font-weight: bold; }
        .footer { margin-top: 40px; text-align: center; color: #666; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Quantum Trail Admin Dashboard</h1>
        <p>System administration and monitoring interface</p>
    </div>
    
    <div class="nav-links">
        <a href="/admin/">Dashboard</a>
        <a href="/admin/system">System Status</a>
        <a href="/admin/quantum-twins">Quantum Twins</a>
        <a href="/">← Back to Main App</a>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-number">{{ stats.active_twins }}</div>
            <div class="stat-label">Active Quantum Twins</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{{ stats.total_simulations }}</div>
            <div class="stat-label">Total Simulations</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{{ stats.total_users }}</div>
            <div class="stat-label">Registered Users</div>
        </div>
        <div class="stat-card">
            <div class="stat-number status-healthy">{{ stats.system_health.title() }}</div>
            <div class="stat-label">System Health</div>
        </div>
    </div>
    
    <div class="stat-card">
        <h3>Quick Actions</h3>
        <p><a href="/health">🔍 Health Check</a></p>
        <p><a href="/docs">📚 API Documentation</a></p>
        <p><a href="/graphql">🔧 GraphQL Playground</a></p>
        <p><strong>Last Updated:</strong> {{ stats.last_update }}</p>
    </div>
    
    <div class="footer">
        <p>Quantum Trail Platform v2.0.0 | Last updated: {{ stats.last_update }}</p>
    </div>
    
    <script>
        // Auto-refresh stats every 30 seconds
        setInterval(() => {
            fetch('/admin/api/stats')
                .then(response => response.json())
                .then(data => {
                    console.log('Updated stats:', data);
                    // Could update the UI here
                })
                .catch(err => console.error('Stats update failed:', err));
        }, 30000);
    </script>
</body>
</html>
"""

SYSTEM_STATUS_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Trail - System Status</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .nav-links { margin: 20px 0; }
        .nav-links a { display: inline-block; margin-right: 15px; padding: 10px 15px; background: #3498db; color: white; text-decoration: none; border-radius: 3px; }
        .service-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .service-card { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status-running { color: #27ae60; font-weight: bold; }
        .status-unknown { color: #f39c12; font-weight: bold; }
        .status-error { color: #e74c3c; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔧 System Status</h1>
        <p>Real-time system monitoring and service status</p>
    </div>
    
    <div class="nav-links">
        <a href="/admin/">Dashboard</a>
        <a href="/admin/system">System Status</a>
        <a href="/admin/quantum-twins">Quantum Twins</a>
        <a href="/">← Back to Main App</a>
    </div>
    
    <div class="service-grid">
        {% for service_name, status in system_info.services.items() %}
        <div class="service-card">
            <h3>{{ service_name.replace('_', ' ').title() }}</h3>
            <p class="status-{{ status }}">Status: {{ status.title() }}</p>
        </div>
        {% endfor %}
    </div>
    
    <div class="service-card" style="margin-top: 20px;">
        <h3>Performance Metrics</h3>
        {% for metric, value in system_info.performance.items() %}
        <p><strong>{{ metric.replace('_', ' ').title() }}:</strong> {{ value }}</p>
        {% endfor %}
    </div>
    
    <div class="service-card" style="margin-top: 20px;">
        <h3>System Information</h3>
        <p><strong>Timestamp:</strong> {{ system_info.timestamp }}</p>
        <p><strong>Platform:</strong> Quantum Trail v2.0.0</p>
        <p><strong>Environment:</strong> Development</p>
    </div>
</body>
</html>
"""

QUANTUM_TWINS_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Trail - Quantum Twins</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .nav-links { margin: 20px 0; }
        .nav-links a { display: inline-block; margin-right: 15px; padding: 10px 15px; background: #3498db; color: white; text-decoration: none; border-radius: 3px; }
        table { width: 100%; background: white; border-collapse: collapse; border-radius: 5px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #34495e; color: white; }
        .status-active { color: #27ae60; font-weight: bold; }
        .status-inactive { color: #e74c3c; font-weight: bold; }
        .empty-state { background: white; padding: 40px; text-align: center; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    </style>
</head>
<body>
    <div class="header">
        <h1>⚛️ Quantum Twins Management</h1>
        <p>Manage and monitor quantum digital twins</p>
    </div>
    
    <div class="nav-links">
        <a href="/admin/">Dashboard</a>
        <a href="/admin/system">System Status</a>
        <a href="/admin/quantum-twins">Quantum Twins</a>
        <a href="/">← Back to Main App</a>
    </div>
    
    {% if twins %}
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Type</th>
                <th>Status</th>
                <th>Created</th>
                <th>Description</th>
            </tr>
        </thead>
        <tbody>
            {% for twin in twins %}
            <tr>
                <td>{{ twin.id }}</td>
                <td>{{ twin.name }}</td>
                <td>{{ twin.type }}</td>
                <td class="status-{{ twin.status }}">{{ twin.status.title() }}</td>
                <td>{{ twin.created[:19] if twin.created != 'N/A' else 'N/A' }}</td>
                <td>{{ twin.description }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    {% else %}
    <div class="empty-state">
        <h3>No Quantum Twins Found</h3>
        <p>No quantum digital twins are currently registered in the system.</p>
        <p>Create your first quantum twin through the API or main interface.</p>
    </div>
    {% endif %}
</body>
</html>
"""

def create_admin_routes():
    """Factory function to create admin routes"""
    return admin_bp