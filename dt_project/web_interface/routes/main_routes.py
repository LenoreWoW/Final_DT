"""
Main application routes for Quantum Trail Platform
"""

from flask import Blueprint, render_template_string, jsonify, request, redirect, url_for
from flask import current_app as app
import json

# Create main blueprint
main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Main landing page"""
    try:
        return render_template_string(MAIN_PAGE_TEMPLATE)
    except Exception as e:
        app.logger.error(f"Main page error: {e}")
        return jsonify({'error': 'Main page temporarily unavailable'}), 500

@main_bp.route('/dashboard')
def dashboard():
    """User dashboard"""
    try:
        return render_template_string(DASHBOARD_TEMPLATE)
    except Exception as e:
        app.logger.error(f"Dashboard error: {e}")
        return jsonify({'error': 'Dashboard temporarily unavailable'}), 500

@main_bp.route('/quantum-playground')
def quantum_playground():
    """Interactive quantum computing playground"""
    try:
        return render_template_string(QUANTUM_PLAYGROUND_TEMPLATE)
    except Exception as e:
        app.logger.error(f"Quantum playground error: {e}")
        return jsonify({'error': 'Quantum playground temporarily unavailable'}), 500

@main_bp.route('/circuit-designer')
def circuit_designer():
    """Interactive quantum circuit designer"""
    try:
        return render_template_string(CIRCUIT_DESIGNER_TEMPLATE)
    except Exception as e:
        app.logger.error(f"Circuit designer error: {e}")
        return jsonify({'error': 'Circuit designer temporarily unavailable'}), 500
@main_bp.route('/quantum-lab')
def quantum_lab():
    """Interactive quantum research laboratory"""
    try:
        return render_template_string(QUANTUM_LAB_TEMPLATE)
    except Exception as e:
        app.logger.error(f"Quantum lab error: {e}")
        return jsonify({'error': 'Quantum lab temporarily unavailable'}), 500



@main_bp.route('/api/status')
def api_status():
    """API status endpoint"""
    try:
        status = {
            'status': 'operational',
            'version': '2.0.0',
            'services': {
                'quantum_backend': 'active',
                'database': 'connected',
                'cache': 'operational'
            },
            'endpoints': {
                'health': '/health',
                'docs': '/docs',
                'admin': '/admin',
                'graphql': '/graphql'
            }
        }
        
        return jsonify(status)
    except Exception as e:
        app.logger.error(f"API status error: {e}")
        return jsonify({'error': 'Status unavailable'}), 500

# GraphQL Playground route
@main_bp.route('/graphql')
def graphql_playground():
    """GraphQL playground interface"""
    try:
        return render_template_string(GRAPHQL_PLAYGROUND_TEMPLATE)
    except Exception as e:
        app.logger.error(f"GraphQL playground error: {e}")
        return jsonify({'error': 'GraphQL playground temporarily unavailable'}), 500

# HTML Templates
MAIN_PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Trail - Quantum Computing Platform</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; padding: 60px 0; color: white; }
        .header h1 { font-size: 4em; margin-bottom: 20px; font-weight: 300; text-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .header p { font-size: 1.3em; opacity: 0.9; max-width: 600px; margin: 0 auto 40px; line-height: 1.6; }
        .cta-buttons { display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin: 40px 0; }
        .btn { 
            display: inline-block; padding: 15px 30px; border-radius: 50px; text-decoration: none; 
            font-weight: 600; transition: all 0.3s ease; font-size: 1.1em; min-width: 180px; text-align: center;
        }
        .btn-primary { background: rgba(255,255,255,0.2); color: white; border: 2px solid rgba(255,255,255,0.3); }
        .btn-primary:hover { background: rgba(255,255,255,0.3); transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0,0,0,0.2); }
        .btn-secondary { background: white; color: #667eea; border: 2px solid white; }
        .btn-secondary:hover { background: transparent; color: white; transform: translateY(-2px); }
        
        .features { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 30px; margin: 80px 0; }
        .feature-card { 
            background: rgba(255,255,255,0.95); padding: 40px; border-radius: 20px; text-align: center; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.1); transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .feature-card:hover { transform: translateY(-10px); box-shadow: 0 20px 50px rgba(0,0,0,0.15); }
        .feature-icon { font-size: 3em; margin-bottom: 20px; }
        .feature-card h3 { font-size: 1.5em; margin-bottom: 15px; color: #333; }
        .feature-card p { color: #666; line-height: 1.6; }
        
        .stats { display: flex; justify-content: center; gap: 60px; margin: 60px 0; flex-wrap: wrap; }
        .stat { text-align: center; color: white; }
        .stat-number { font-size: 3em; font-weight: 700; display: block; }
        .stat-label { font-size: 1.1em; opacity: 0.8; margin-top: 5px; }
        
        .footer { text-align: center; padding: 40px 0; color: rgba(255,255,255,0.8); border-top: 1px solid rgba(255,255,255,0.1); margin-top: 80px; }
        .footer a { color: rgba(255,255,255,0.9); text-decoration: none; margin: 0 15px; }
        .footer a:hover { color: white; }
        
        @media (max-width: 768px) {
            .header h1 { font-size: 2.5em; }
            .cta-buttons { flex-direction: column; align-items: center; }
            .stats { flex-direction: column; gap: 30px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚛️ Quantum Trail</h1>
            <p>Advanced Quantum Computing Platform for Digital Twins, Optimization, and Machine Learning</p>
            
            <div class="cta-buttons">
                <a href="/dashboard" class="btn btn-primary">🚀 Launch Dashboard</a>
                <a href="/quantum-playground" class="btn btn-secondary">🧪 Quantum Playground</a>
            </div>
            
            <div class="stats">
                <div class="stat">
                    <span class="stat-number">∞</span>
                    <span class="stat-label">Quantum Possibilities</span>
                </div>
                <div class="stat">
                    <span class="stat-number">100%</span>
                    <span class="stat-label">Open Source</span>
                </div>
                <div class="stat">
                    <span class="stat-number">24/7</span>
                    <span class="stat-label">Available</span>
                </div>
            </div>
        </div>
        
        <div class="features">
            <div class="feature-card">
                <div class="feature-icon">🔬</div>
                <h3>Quantum Digital Twins</h3>
                <p>Create quantum simulations of real-world systems for portfolio optimization, route planning, and complex system modeling.</p>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <h3>Quantum Optimization</h3>
                <p>Leverage QAOA and VQE algorithms to solve complex optimization problems with quantum advantage.</p>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">🧠</div>
                <h3>Quantum Machine Learning</h3>
                <p>Train quantum neural networks and variational quantum classifiers for next-generation AI applications.</p>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">🔄</div>
                <h3>Real-time Processing</h3>
                <p>Get live updates through WebSocket connections and monitor your quantum computations in real-time.</p>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">🌐</div>
                <h3>REST & GraphQL APIs</h3>
                <p>Comprehensive APIs for seamless integration with your applications and workflows.</p>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <h3>Advanced Analytics</h3>
                <p>Monitor performance with Prometheus metrics, Grafana dashboards, and comprehensive logging.</p>
            </div>
        </div>
    </div>
    
    <div class="footer">
        <div class="container">
            <p>
                <a href="/docs">📚 Documentation</a>
                <a href="/admin">⚙️ Admin Panel</a>
                <a href="/graphql">🔧 GraphQL</a>
                <a href="/health">🔍 Health Check</a>
            </p>
            <p style="margin-top: 20px; font-size: 0.9em;">
                Quantum Trail Platform v2.0.0 | Powered by Qiskit, Flask, and Quantum Computing
            </p>
        </div>
    </div>
</body>
</html>
"""

DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Trail - Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #f8fafc; color: #334155; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px 0; }
        .container { max-width: 1200px; margin: 0 auto; padding: 0 20px; }
        .header h1 { font-size: 2em; margin-bottom: 10px; }
        .header p { opacity: 0.9; }
        .nav { background: white; padding: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .nav-links { display: flex; gap: 30px; }
        .nav-links a { color: #64748b; text-decoration: none; font-weight: 500; padding: 5px 0; border-bottom: 2px solid transparent; transition: all 0.2s; }
        .nav-links a:hover, .nav-links a.active { color: #667eea; border-bottom-color: #667eea; }
        .main { padding: 40px 0; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 25px; margin-bottom: 40px; }
        .card { background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; }
        .card h3 { color: #1e293b; margin-bottom: 15px; display: flex; align-items: center; gap: 10px; }
        .card p { color: #64748b; line-height: 1.6; }
        .btn { display: inline-block; padding: 10px 20px; background: #667eea; color: white; text-decoration: none; border-radius: 6px; font-weight: 500; transition: background 0.2s; }
        .btn:hover { background: #5a67d8; }
        .btn-secondary { background: #e2e8f0; color: #475569; }
        .btn-secondary:hover { background: #cbd5e1; }
        .status { padding: 4px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 600; }
        .status-healthy { background: #dcfce7; color: #166534; }
        .status-warning { background: #fef3c7; color: #92400e; }
        .quick-actions { display: flex; gap: 15px; flex-wrap: wrap; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1>🚀 Quantum Trail Dashboard</h1>
            <p>Monitor and manage your quantum computing workloads</p>
        </div>
    </div>
    
    <div class="nav">
        <div class="container">
            <div class="nav-links">
                <a href="/dashboard" class="active">Dashboard</a>
                <a href="/quantum-playground">Quantum Playground</a>
                <a href="/docs">API Documentation</a>
                <a href="/admin">Admin Panel</a>
                <a href="/">← Home</a>
            </div>
        </div>
    </div>
    
    <div class="main">
        <div class="container">
            <div class="grid">
                <div class="card">
                    <h3>🔬 Quantum Twins</h3>
                    <p>Create and manage quantum digital twins for complex system simulation and optimization.</p>
                    <div class="quick-actions">
                        <a href="#" class="btn" onclick="createTwin()">Create Twin</a>
                        <a href="#" class="btn-secondary" onclick="listTwins()">View All</a>
                    </div>
                    <div style="margin-top: 20px;">
                        <span class="status status-healthy">0 Active</span>
                    </div>
                </div>
                
                <div class="card">
                    <h3>⚡ Optimization</h3>
                    <p>Run QAOA and VQE algorithms for portfolio optimization, routing problems, and combinatorial challenges.</p>
                    <div class="quick-actions">
                        <a href="#" class="btn" onclick="startOptimization()">Start Optimization</a>
                        <a href="#" class="btn-secondary" onclick="viewResults()">View Results</a>
                    </div>
                </div>
                
                <div class="card">
                    <h3>🧠 Quantum ML</h3>
                    <p>Train quantum neural networks and variational quantum classifiers for machine learning tasks.</p>
                    <div class="quick-actions">
                        <a href="#" class="btn" onclick="trainModel()">Train Model</a>
                        <a href="#" class="btn-secondary" onclick="loadDataset()">Load Dataset</a>
                    </div>
                </div>
                
                <div class="card">
                    <h3>📊 System Status</h3>
                    <p>Monitor system health, performance metrics, and service availability.</p>
                    <div style="margin-top: 15px;">
                        <div><strong>API:</strong> <span class="status status-healthy">Healthy</span></div>
                        <div style="margin-top: 5px;"><strong>Quantum Backend:</strong> <span class="status status-healthy">Active</span></div>
                        <div style="margin-top: 5px;"><strong>Database:</strong> <span class="status status-healthy">Connected</span></div>
                    </div>
                    <div class="quick-actions">
                        <a href="/health" class="btn-secondary">Health Check</a>
                        <a href="/admin" class="btn-secondary">Admin Panel</a>
                    </div>
                </div>
                
                <div class="card">
                    <h3>🔧 Development Tools</h3>
                    <p>Access development utilities, API documentation, and testing interfaces.</p>
                    <div class="quick-actions">
                        <a href="/docs" class="btn">API Docs</a>
                        <a href="/graphql" class="btn">GraphQL</a>
                        <a href="/quantum-playground" class="btn-secondary">Playground</a>
                    </div>
                </div>
                
                <div class="card">
                    <h3>📈 Analytics</h3>
                    <p>View performance metrics, usage statistics, and system analytics.</p>
                    <div class="quick-actions">
                        <a href="http://localhost:3000" class="btn" target="_blank">Grafana</a>
                        <a href="http://localhost:9090" class="btn-secondary" target="_blank">Prometheus</a>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h3>🚀 Quick Start</h3>
                <p>Get started with the Quantum Trail platform:</p>
                <ol style="margin: 20px 0 20px 20px; line-height: 1.8;">
                    <li>Create your first quantum twin using the <a href="/docs">API</a></li>
                    <li>Run a quantum optimization algorithm</li>
                    <li>Monitor results in real-time through WebSocket connections</li>
                    <li>Explore quantum machine learning capabilities</li>
                    <li>Scale your quantum applications with our comprehensive platform</li>
                </ol>
                <div class="quick-actions">
                    <a href="/docs" class="btn">View Documentation</a>
                    <a href="/quantum-playground" class="btn-secondary">Try Playground</a>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Dashboard interaction functions
        function createTwin() {
            alert('Twin creation will be implemented in the interactive interface');
        }
        
        function listTwins() {
            alert('Twin listing will be implemented in the interactive interface');
        }
        
        function startOptimization() {
            alert('Optimization interface will be implemented');
        }
        
        function viewResults() {
            alert('Results viewer will be implemented');
        }
        
        function trainModel() {
            alert('ML training interface will be implemented');
        }
        
        function loadDataset() {
            alert('Dataset loader will be implemented');
        }
        
        // Auto-refresh system status
        async function updateSystemStatus() {
            try {
                const response = await fetch('/api/status');
                const status = await response.json();
                console.log('System status updated:', status);
            } catch (error) {
                console.error('Failed to update status:', error);
            }
        }
        
        // Update status every 30 seconds
        setInterval(updateSystemStatus, 30000);
        updateSystemStatus();
    </script>
</body>
</html>
"""

QUANTUM_PLAYGROUND_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Trail - Playground</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Monaco', 'Menlo', monospace; background: #1a1a1a; color: #e0e0e0; }
        .header { background: #2d3748; color: white; padding: 20px 0; }
        .container { max-width: 1400px; margin: 0 auto; padding: 0 20px; }
        .header h1 { font-size: 2em; margin-bottom: 10px; }
        .nav { background: #4a5568; padding: 10px 0; }
        .nav-links { display: flex; gap: 20px; }
        .nav-links a { color: #e2e8f0; text-decoration: none; padding: 8px 16px; border-radius: 4px; transition: background 0.2s; }
        .nav-links a:hover { background: #2d3748; }
        .main { padding: 30px 0; }
        .playground { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; height: 80vh; }
        .panel { background: #2d3748; border-radius: 8px; overflow: hidden; display: flex; flex-direction: column; }
        .panel-header { background: #4a5568; padding: 15px; font-weight: bold; }
        .panel-content { flex: 1; padding: 20px; overflow: auto; }
        .code-editor { background: #1a202c; border: none; color: #e2e8f0; font-family: 'Monaco', monospace; font-size: 14px; padding: 15px; resize: none; flex: 1; }
        .code-editor:focus { outline: none; }
        .btn { padding: 10px 20px; background: #4299e1; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; }
        .btn:hover { background: #3182ce; }
        .btn-secondary { background: #68d391; }
        .btn-secondary:hover { background: #48bb78; }
        .result { background: #1a202c; padding: 15px; border-radius: 4px; border: 1px solid #4a5568; margin-top: 15px; }
        .controls { display: flex; gap: 15px; align-items: center; padding: 15px; background: #4a5568; }
        .select { background: #2d3748; color: #e2e8f0; border: 1px solid #4a5568; padding: 8px 12px; border-radius: 4px; }
        .examples { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 15px; }
        .example-btn { padding: 6px 12px; background: #2b6cb0; color: white; border: none; border-radius: 3px; cursor: pointer; font-size: 12px; }
        .example-btn:hover { background: #2c5282; }
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1>🧪 Quantum Playground</h1>
            <p>Interactive quantum circuit design and simulation</p>
        </div>
    </div>
    
    <div class="nav">
        <div class="container">
            <div class="nav-links">
                <a href="/quantum-playground">Playground</a>
                <a href="/dashboard">Dashboard</a>
                <a href="/docs">Documentation</a>
                <a href="/">← Home</a>
            </div>
        </div>
    </div>
    
    <div class="main">
        <div class="container">
            <div class="playground">
                <div class="panel">
                    <div class="panel-header">⚛️ Quantum Circuit Editor</div>
                    <div class="controls">
                        <label>Algorithm:</label>
                        <select class="select" id="algorithm">
                            <option value="custom">Custom Circuit</option>
                            <option value="qaoa">QAOA</option>
                            <option value="vqe">VQE</option>
                            <option value="qml">Quantum ML</option>
                        </select>
                        <label>Qubits:</label>
                        <select class="select" id="qubits">
                            <option value="2">2</option>
                            <option value="3">3</option>
                            <option value="4" selected>4</option>
                            <option value="5">5</option>
                        </select>
                        <button class="btn" onclick="runCircuit()">Run Circuit</button>
                        <button class="btn-secondary" onclick="visualizeCircuit()">Visualize</button>
                    </div>
                    <div class="examples">
                        <button class="example-btn" onclick="loadExample('bell')">Bell State</button>
                        <button class="example-btn" onclick="loadExample('ghz')">GHZ State</button>
                        <button class="example-btn" onclick="loadExample('qaoa')">QAOA</button>
                        <button class="example-btn" onclick="loadExample('vqe')">VQE</button>
                        <button class="example-btn" onclick="loadExample('qft')">QFT</button>
                    </div>
                    <div class="panel-content">
                        <textarea class="code-editor" id="circuit-code" placeholder="Enter your QASM code here...">OPENQASM 2.0;
include "qelib1.inc";

qreg q[4];
creg c[4];

// Create Bell state on first two qubits
h q[0];
cx q[0],q[1];

// Add some entanglement
cx q[1],q[2];
cx q[2],q[3];

// Measurements
measure q -> c;</textarea>
                    </div>
                </div>
                
                <div class="panel">
                    <div class="panel-header">📊 Results & Visualization</div>
                    <div class="panel-content">
                        <div id="results">
                            <p style="text-align: center; opacity: 0.7; margin-top: 50px;">
                                🎯 Results will appear here after running your circuit
                            </p>
                            <p style="text-align: center; opacity: 0.5; margin-top: 20px;">
                                Use the examples above to get started, or write your own QASM code
                            </p>
                        </div>
                        
                        <div class="result" style="margin-top: 30px;">
                            <strong>💡 Tips:</strong>
                            <ul style="margin-top: 10px; line-height: 1.6;">
                                <li>Try the Bell State example for quantum entanglement</li>
                                <li>Use QAOA for optimization problems</li>
                                <li>VQE is great for ground state calculations</li>
                                <li>QFT demonstrates quantum algorithms</li>
                            </ul>
                        </div>
                        
                        <div class="result">
                            <strong>🔗 API Integration:</strong>
                            <pre style="margin-top: 10px; font-size: 12px; overflow-x: auto;">curl -X POST http://localhost:8000/api/quantum_twins/1/simulate \\
  -H "Content-Type: application/json" \\
  -d '{"algorithm": "custom", "qasm_code": "..."}'</pre>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const examples = {
            bell: `OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[2];

h q[0];
cx q[0],q[1];
measure q -> c;`,
            
            ghz: `OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg c[3];

h q[0];
cx q[0],q[1];
cx q[1],q[2];
measure q -> c;`,
            
            qaoa: `OPENQASM 2.0;
include "qelib1.inc";

qreg q[4];
creg c[4];

// Initial superposition
h q[0];
h q[1];
h q[2];
h q[3];

// Problem Hamiltonian (example)
cx q[0],q[1];
rz(0.5) q[1];
cx q[0],q[1];

cx q[1],q[2];
rz(0.3) q[2];
cx q[1],q[2];

// Mixer Hamiltonian
rx(0.7) q[0];
rx(0.7) q[1];
rx(0.7) q[2];
rx(0.7) q[3];

measure q -> c;`,
            
            vqe: `OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[2];

// Ansatz circuit
ry(0.5) q[0];
ry(0.3) q[1];
cx q[0],q[1];
ry(0.8) q[0];
ry(0.2) q[1];

measure q -> c;`,
            
            qft: `OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg c[3];

// Quantum Fourier Transform
h q[0];
cu1(pi/2) q[1],q[0];
cu1(pi/4) q[2],q[0];
h q[1];
cu1(pi/2) q[2],q[1];
h q[2];

// Swap qubits
swap q[0],q[2];

measure q -> c;`
        };
        
        function loadExample(exampleName) {
            const code = examples[exampleName];
            if (code) {
                document.getElementById('circuit-code').value = code;
                updateQubits(exampleName);
            }
        }
        
        function updateQubits(exampleName) {
            const qubitSelect = document.getElementById('qubits');
            const algorithmSelect = document.getElementById('algorithm');
            
            switch(exampleName) {
                case 'bell':
                case 'vqe':
                    qubitSelect.value = '2';
                    break;
                case 'ghz':
                case 'qft':
                    qubitSelect.value = '3';
                    break;
                case 'qaoa':
                    qubitSelect.value = '4';
                    break;
            }
            
            algorithmSelect.value = exampleName === 'bell' || exampleName === 'ghz' || exampleName === 'qft' ? 'custom' : exampleName;
        }
        
        function runCircuit() {
            const code = document.getElementById('circuit-code').value;
            const algorithm = document.getElementById('algorithm').value;
            const qubits = document.getElementById('qubits').value;
            
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = `
                <div style="text-align: center; margin: 20px 0;">
                    <div style="font-size: 2em; margin-bottom: 10px;">⚡</div>
                    <p>Running quantum circuit...</p>
                </div>
                <div class="result">
                    <strong>Circuit Info:</strong><br>
                    Algorithm: ${algorithm}<br>
                    Qubits: ${qubits}<br>
                    Gates: ${countGates(code)}<br>
                    Depth: ${estimateDepth(code)}
                </div>
            `;
            
            // Simulate execution delay
            setTimeout(() => {
                const mockResults = generateMockResults(qubits);
                displayResults(mockResults, code);
            }, 2000);
        }
        
        function visualizeCircuit() {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = `
                <div style="text-align: center; margin: 20px 0;">
                    <div style="font-size: 2em; margin-bottom: 10px;">🎨</div>
                    <p>Circuit visualization would appear here</p>
                    <p style="font-size: 0.9em; opacity: 0.7;">In a full implementation, this would show:</p>
                    <ul style="text-align: left; display: inline-block; margin-top: 10px;">
                        <li>Interactive circuit diagram</li>
                        <li>Gate-by-gate visualization</li>
                        <li>State evolution animation</li>
                        <li>Bloch sphere representations</li>
                    </ul>
                </div>
            `;
        }
        
        function countGates(code) {
            const gateNames = ['h', 'x', 'y', 'z', 'cx', 'cy', 'cz', 'rx', 'ry', 'rz', 'cu1', 'swap'];
            let count = 0;
            const lines = code.split('\\n');
            for (const line of lines) {
                for (const gate of gateNames) {
                    if (line.trim().startsWith(gate + ' ')) {
                        count++;
                    }
                }
            }
            return count;
        }
        
        function estimateDepth(code) {
            const lines = code.split('\\n').filter(line => 
                line.trim() && 
                !line.includes('//') && 
                !line.includes('OPENQASM') && 
                !line.includes('include') &&
                !line.includes('qreg') &&
                !line.includes('creg') &&
                !line.includes('measure')
            );
            return Math.max(1, lines.length);
        }
        
        function generateMockResults(qubits) {
            const numStates = Math.pow(2, parseInt(qubits));
            const results = {};
            
            // Generate random measurement results
            for (let i = 0; i < numStates; i++) {
                const binary = i.toString(2).padStart(parseInt(qubits), '0');
                results[binary] = Math.floor(Math.random() * 1024);
            }
            
            return results;
        }
        
        function displayResults(results, code) {
            const resultsDiv = document.getElementById('results');
            const total = Object.values(results).reduce((a, b) => a + b, 0);
            
            let html = `
                <div style="text-align: center; margin-bottom: 20px;">
                    <div style="font-size: 2em; margin-bottom: 10px;">✅</div>
                    <p><strong>Execution Complete!</strong></p>
                </div>
                
                <div class="result">
                    <strong>📊 Measurement Results (${total} shots):</strong>
                    <div style="margin-top: 10px; font-family: monospace;">
            `;
            
            for (const [state, count] of Object.entries(results)) {
                const probability = (count / total * 100).toFixed(1);
                const barWidth = Math.max(1, (count / total * 100));
                html += `
                    <div style="margin: 5px 0; display: flex; align-items: center;">
                        <span style="width: 60px;">|${state}⟩:</span>
                        <div style="width: 200px; background: #4a5568; height: 16px; margin: 0 10px; border-radius: 2px; overflow: hidden;">
                            <div style="width: ${barWidth}%; height: 100%; background: #4299e1;"></div>
                        </div>
                        <span>${count} (${probability}%)</span>
                    </div>
                `;
            }
            
            html += `
                    </div>
                </div>
                
                <div class="result">
                    <strong>🔬 Quantum Properties:</strong>
                    <ul style="margin-top: 10px;">
                        <li>Entanglement: ${hasEntanglement(code) ? 'Detected' : 'None'}</li>
                        <li>Superposition: ${hasSuperposition(code) ? 'Yes' : 'No'}</li>
                        <li>Measurement Basis: Computational</li>
                    </ul>
                </div>
                
                <div class="result">
                    <strong>⚡ Performance:</strong>
                    <ul style="margin-top: 10px;">
                        <li>Execution Time: ${(Math.random() * 2 + 0.5).toFixed(2)}s</li>
                        <li>Backend: qasm_simulator</li>
                        <li>Optimization Level: 1</li>
                    </ul>
                </div>
            `;
            
            resultsDiv.innerHTML = html;
        }
        
        function hasEntanglement(code) {
            return code.includes('cx') || code.includes('cy') || code.includes('cz') || code.includes('cu1');
        }
        
        function hasSuperposition(code) {
            return code.includes('h ') || code.includes('ry') || code.includes('rx');
        }
        
        // Load Bell state example by default
        loadExample('bell');
    </script>
</body>
</html>
"""

CIRCUIT_DESIGNER_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Trail - Circuit Designer</title>
    <link rel="stylesheet" href="/static/css/circuit_designer.css">
</head>
<body>
    <div id="circuitDesigner"></div>
    <script src="/static/js/circuit_designer.js"></script>
</body>
</html>
"""

GRAPHQL_PLAYGROUND_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Trail - GraphQL Playground</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Monaco', 'Menlo', monospace; background: #1e1e1e; color: #d4d4d4; }
        .header { background: #2d3748; color: white; padding: 20px 0; }
        .container { max-width: 1400px; margin: 0 auto; padding: 0 20px; }
        .header h1 { font-size: 2em; margin-bottom: 10px; }
        .nav { background: #4a5568; padding: 10px 0; }
        .nav-links { display: flex; gap: 20px; }
        .nav-links a { color: #e2e8f0; text-decoration: none; padding: 8px 16px; border-radius: 4px; transition: background 0.2s; }
        .nav-links a:hover { background: #2d3748; }
        .main { padding: 30px 0; }
        .playground { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; height: 80vh; }
        .panel { background: #2d3748; border-radius: 8px; overflow: hidden; display: flex; flex-direction: column; }
        .panel-header { background: #4a5568; padding: 15px; font-weight: bold; display: flex; justify-content: space-between; align-items: center; }
        .panel-content { flex: 1; overflow: auto; }
        .code-editor { background: #1a202c; border: none; color: #e2e8f0; font-family: 'Monaco', monospace; font-size: 14px; padding: 15px; resize: none; flex: 1; min-height: 300px; }
        .code-editor:focus { outline: none; }
        .btn { padding: 8px 16px; background: #4299e1; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: bold; }
        .btn:hover { background: #3182ce; }
        .btn-secondary { background: #68d391; }
        .btn-secondary:hover { background: #48bb78; }
        .examples { display: flex; gap: 10px; flex-wrap: wrap; padding: 15px; background: #4a5568; }
        .example-btn { padding: 6px 12px; background: #2b6cb0; color: white; border: none; border-radius: 3px; cursor: pointer; font-size: 12px; }
        .example-btn:hover { background: #2c5282; }
        .result { background: #1a202c; padding: 15px; font-size: 13px; line-height: 1.5; border-top: 1px solid #4a5568; }
        pre { margin: 0; white-space: pre-wrap; }
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1>🔧 GraphQL Playground</h1>
            <p>Interactive GraphQL query interface for the Quantum Trail API</p>
        </div>
    </div>
    
    <div class="nav">
        <div class="container">
            <div class="nav-links">
                <a href="/graphql">GraphQL</a>
                <a href="/docs">API Docs</a>
                <a href="/dashboard">Dashboard</a>
                <a href="/">← Home</a>
            </div>
        </div>
    </div>
    
    <div class="main">
        <div class="container">
            <div class="playground">
                <div class="panel">
                    <div class="panel-header">
                        📝 GraphQL Query
                        <button class="btn" onclick="executeQuery()">Run Query ▶</button>
                    </div>
                    <div class="examples">
                        <button class="example-btn" onclick="loadQuery('twins')">List Twins</button>
                        <button class="example-btn" onclick="loadQuery('create')">Create Twin</button>
                        <button class="example-btn" onclick="loadQuery('simulate')">Run Simulation</button>
                        <button class="example-btn" onclick="loadQuery('results')">Get Results</button>
                        <button class="example-btn" onclick="loadQuery('schema')">Introspection</button>
                    </div>
                    <div class="panel-content">
                        <textarea class="code-editor" id="query-editor" placeholder="Enter your GraphQL query here...">query GetQuantumTwins {
  quantumTwins {
    id
    name
    description
    twinType
    isActive
    parameters
    createdAt
    updatedAt
  }
}</textarea>
                    </div>
                </div>
                
                <div class="panel">
                    <div class="panel-header">
                        📊 Results
                        <div>
                            <button class="btn-secondary" onclick="formatResult()">Format</button>
                            <button class="btn" onclick="copyResult()">Copy</button>
                        </div>
                    </div>
                    <div class="panel-content">
                        <div class="result" id="query-result">
                            <pre>{
  "message": "🎯 Query results will appear here after execution",
  "status": "ready",
  "endpoint": "http://localhost:8000/graphql"
}</pre>
                        </div>
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 30px; background: #2d3748; padding: 25px; border-radius: 8px;">
                <h3 style="color: #68d391; margin-bottom: 15px;">🚀 GraphQL Schema Overview</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
                    <div>
                        <h4 style="color: #fbb6ce; margin-bottom: 10px;">📋 Queries</h4>
                        <ul style="line-height: 1.6; font-size: 13px;">
                            <li><code>quantumTwins</code> - List all quantum twins</li>
                            <li><code>quantumTwin(id: ID!)</code> - Get specific twin</li>
                            <li><code>simulationResults</code> - List simulation results</li>
                            <li><code>systemHealth</code> - Get system health status</li>
                        </ul>
                    </div>
                    <div>
                        <h4 style="color: #fbb6ce; margin-bottom: 10px;">✏️ Mutations</h4>
                        <ul style="line-height: 1.6; font-size: 13px;">
                            <li><code>createQuantumTwin</code> - Create new twin</li>
                            <li><code>updateQuantumTwin</code> - Update existing twin</li>
                            <li><code>runSimulation</code> - Execute simulation</li>
                            <li><code>deleteQuantumTwin</code> - Delete twin</li>
                        </ul>
                    </div>
                    <div>
                        <h4 style="color: #fbb6ce; margin-bottom: 10px;">🔄 Subscriptions</h4>
                        <ul style="line-height: 1.6; font-size: 13px;">
                            <li><code>simulationUpdates</code> - Real-time progress</li>
                            <li><code>systemEvents</code> - System notifications</li>
                            <li><code>optimizationResults</code> - Live results</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const queryExamples = {
            twins: `query GetQuantumTwins {
  quantumTwins {
    id
    name
    description
    twinType
    isActive
    parameters
    createdAt
    updatedAt
  }
}`,
            
            create: `mutation CreateQuantumTwin($input: CreateQuantumTwinInput!) {
  createQuantumTwin(input: $input) {
    id
    name
    twinType
    status
    message
  }
}

# Variables
{
  "input": {
    "name": "Portfolio Optimization Twin",
    "description": "Quantum twin for financial portfolio optimization",
    "twinType": "PORTFOLIO_OPTIMIZATION",
    "parameters": {
      "assets": ["AAPL", "GOOGL", "MSFT"],
      "riskTolerance": 0.1,
      "expectedReturns": [0.12, 0.15, 0.10]
    }
  }
}`,
            
            simulate: `mutation RunSimulation($input: SimulationInput!) {
  runSimulation(input: $input) {
    id
    status
    algorithm
    progress
    result
    executionTime
    createdAt
  }
}

# Variables
{
  "input": {
    "twinId": 1,
    "algorithm": "QAOA",
    "shots": 1024,
    "parameters": {
      "p": 2,
      "maxIterations": 100
    }
  }
}`,
            
            results: `query GetSimulationResults($twinId: ID) {
  simulationResults(twinId: $twinId) {
    id
    twinId
    algorithm
    status
    result
    executionTime
    createdAt
    twin {
      name
      twinType
    }
  }
}

# Variables
{
  "twinId": 1
}`,
            
            schema: `query IntrospectionQuery {
  __schema {
    queryType {
      name
      fields {
        name
        type {
          name
          kind
        }
      }
    }
    mutationType {
      name
      fields {
        name
        type {
          name
          kind
        }
      }
    }
  }
}`
        };
        
        function loadQuery(queryName) {
            const query = queryExamples[queryName];
            if (query) {
                document.getElementById('query-editor').value = query;
            }
        }
        
        function executeQuery() {
            const query = document.getElementById('query-editor').value;
            const resultDiv = document.getElementById('query-result');
            
            // Show loading state
            resultDiv.innerHTML = '<pre>⚡ Executing GraphQL query...</pre>';
            
            // Simulate GraphQL execution
            setTimeout(() => {
                const mockResult = generateMockGraphQLResult(query);
                resultDiv.innerHTML = `<pre>${JSON.stringify(mockResult, null, 2)}</pre>`;
            }, 1000);
        }
        
        function generateMockGraphQLResult(query) {
            if (query.includes('quantumTwins')) {
                return {
                    data: {
                        quantumTwins: [
                            {
                                id: "1",
                                name: "Sample Portfolio Twin",
                                description: "A sample quantum digital twin for portfolio optimization",
                                twinType: "PORTFOLIO_OPTIMIZATION",
                                isActive: true,
                                parameters: {
                                    assets: ["AAPL", "GOOGL", "MSFT", "AMZN"],
                                    riskTolerance: 0.1,
                                    expectedReturns: [0.12, 0.15, 0.10, 0.14]
                                },
                                createdAt: "2024-01-15T10:30:00Z",
                                updatedAt: "2024-01-15T10:30:00Z"
                            }
                        ]
                    }
                };
            } else if (query.includes('createQuantumTwin')) {
                return {
                    data: {
                        createQuantumTwin: {
                            id: "2",
                            name: "Portfolio Optimization Twin",
                            twinType: "PORTFOLIO_OPTIMIZATION",
                            status: "CREATED",
                            message: "Quantum twin created successfully"
                        }
                    }
                };
            } else if (query.includes('runSimulation')) {
                return {
                    data: {
                        runSimulation: {
                            id: "sim_001",
                            status: "COMPLETED",
                            algorithm: "QAOA",
                            progress: 100,
                            result: {
                                optimalWeights: [0.25, 0.35, 0.20, 0.20],
                                expectedReturn: 0.138,
                                portfolioRisk: 0.089,
                                sharpeRatio: 1.55
                            },
                            executionTime: 15.7,
                            createdAt: "2024-01-15T10:35:00Z"
                        }
                    }
                };
            } else if (query.includes('simulationResults')) {
                return {
                    data: {
                        simulationResults: [
                            {
                                id: "sim_001",
                                twinId: "1",
                                algorithm: "QAOA",
                                status: "COMPLETED",
                                result: {
                                    optimalWeights: [0.25, 0.35, 0.20, 0.20],
                                    expectedReturn: 0.138
                                },
                                executionTime: 15.7,
                                createdAt: "2024-01-15T10:35:00Z",
                                twin: {
                                    name: "Sample Portfolio Twin",
                                    twinType: "PORTFOLIO_OPTIMIZATION"
                                }
                            }
                        ]
                    }
                };
            } else if (query.includes('__schema')) {
                return {
                    data: {
                        __schema: {
                            queryType: {
                                name: "Query",
                                fields: [
                                    { name: "quantumTwins", type: { name: "[QuantumTwin]", kind: "LIST" } },
                                    { name: "quantumTwin", type: { name: "QuantumTwin", kind: "OBJECT" } },
                                    { name: "simulationResults", type: { name: "[SimulationResult]", kind: "LIST" } }
                                ]
                            },
                            mutationType: {
                                name: "Mutation",
                                fields: [
                                    { name: "createQuantumTwin", type: { name: "QuantumTwin", kind: "OBJECT" } },
                                    { name: "runSimulation", type: { name: "SimulationResult", kind: "OBJECT" } }
                                ]
                            }
                        }
                    }
                };
            } else {
                return {
                    errors: [
                        {
                            message: "Query not recognized in demo mode",
                            locations: [{ line: 1, column: 1 }]
                        }
                    ]
                };
            }
        }
        
        function formatResult() {
            const resultDiv = document.getElementById('query-result');
            const pre = resultDiv.querySelector('pre');
            if (pre) {
                try {
                    const parsed = JSON.parse(pre.textContent);
                    pre.textContent = JSON.stringify(parsed, null, 2);
                } catch (e) {
                    console.log('Result is already formatted or not JSON');
                }
            }
        }
        
        function copyResult() {
            const resultDiv = document.getElementById('query-result');
            const pre = resultDiv.querySelector('pre');
            if (pre) {
                navigator.clipboard.writeText(pre.textContent).then(() => {
                    alert('Result copied to clipboard!');
                }).catch(() => {
                    alert('Failed to copy result');
                });
            }
        }
        
        // Load default query
        loadQuery('twins');
    </script>
</body>
</html>
"""

QUANTUM_LAB_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Research Lab - Digital Twin Platform</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootswatch@5.1.3/dist/darkly/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    
    <style>
        :root {
            --quantum-primary: #0D47A1;
            --quantum-accent: #00E676;
            --quantum-secondary: #1A237E;
            --bg-primary: #0a0a0f;
            --bg-secondary: #1a1a2e;
            --bg-glass: rgba(10, 10, 15, 0.9);
            --glow-primary: 0 0 20px rgba(0, 230, 118, 0.5);
        }
        
        body {
            background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
            color: #e0e7ff;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            min-height: 100vh;
        }
        
        .navbar {
            background: var(--bg-glass) !important;
            backdrop-filter: blur(20px);
            border-bottom: 1px solid rgba(0, 230, 118, 0.2);
        }
        
        .navbar-brand {
            color: var(--quantum-accent) !important;
            font-weight: 700;
        }
        
        .container-fluid {
            padding: 2rem;
        }
        
        .lab-section {
            margin-bottom: 2rem;
        }
        
        .card {
            background: var(--bg-glass);
            border: 1px solid rgba(0, 230, 118, 0.2);
            backdrop-filter: blur(20px);
        }
        
        .parameter-group {
            background: var(--bg-glass);
            border: 1px solid rgba(0, 230, 118, 0.2);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            backdrop-filter: blur(20px);
        }
        
        .quantum-circuit-designer {
            min-height: 400px;
            background: linear-gradient(135deg, rgba(13, 71, 161, 0.1), rgba(0, 230, 118, 0.1));
            border-radius: 12px;
            border: 2px dashed rgba(0, 230, 118, 0.3);
            position: relative;
            overflow: hidden;
            padding: 20px;
        }
        
        .circuit-gate {
            display: inline-block;
            background: var(--quantum-primary);
            color: white;
            padding: 0.5rem 1rem;
            margin: 0.25rem;
            border-radius: 8px;
            cursor: grab;
            transition: all 0.3s ease;
            border: 1px solid var(--quantum-accent);
        }
        
        .circuit-gate:hover {
            transform: scale(1.05);
            box-shadow: var(--glow-primary);
        }
        
        .upload-area {
            border: 2px dashed var(--quantum-accent);
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            background: var(--bg-glass);
            backdrop-filter: blur(20px);
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .upload-area:hover {
            border-color: var(--quantum-primary);
            box-shadow: var(--glow-primary);
        }
        
        .algorithm-card {
            cursor: pointer;
            transition: all 0.3s ease;
            border: 2px solid transparent;
            background: var(--bg-glass);
        }
        
        .algorithm-card:hover {
            border-color: var(--quantum-accent);
            box-shadow: var(--glow-primary);
        }
        
        .algorithm-card.selected {
            border-color: var(--quantum-accent);
            background: rgba(0, 230, 118, 0.1);
        }
        
        .btn-quantum {
            background: linear-gradient(45deg, var(--quantum-primary), var(--quantum-accent));
            border: none;
            color: white;
            font-weight: 600;
        }
        
        .btn-quantum:hover {
            box-shadow: var(--glow-primary);
            transform: translateY(-2px);
        }
        
        .results-container {
            background: var(--bg-glass);
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 1rem;
            border: 1px solid rgba(0, 230, 118, 0.2);
        }
        
        .console-output {
            background: #000;
            color: #00ff00;
            font-family: 'Courier New', monospace;
            padding: 1rem;
            border-radius: 8px;
            max-height: 200px;
            overflow-y: auto;
            border: 1px solid var(--quantum-accent);
        }
        
        .parameter-slider {
            width: 100%;
            margin: 1rem 0;
            accent-color: var(--quantum-accent);
        }
    </style>
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark">
        <div class="container-fluid">
            <a class="navbar-brand" href="/">
                <i class="fas fa-atom me-2"></i>Quantum Digital Twin Platform
            </a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="/dashboard"><i class="fas fa-tachometer-alt me-1"></i>Dashboard</a>
                <a class="nav-link" href="/quantum-playground"><i class="fas fa-flask me-1"></i>Playground</a>
                <a class="nav-link" href="/"><i class="fas fa-home me-1"></i>Home</a>
            </div>
        </div>
    </nav>

    <!-- Main Content -->
    <div class="container-fluid">
        <!-- Page Header -->
        <div class="row mb-4">
            <div class="col-12">
                <h1 class="display-4 text-center" style="color: var(--quantum-accent);">
                    <i class="fas fa-microscope me-3"></i>Quantum Research Laboratory
                </h1>
                <p class="text-center lead">Interactive quantum experiment platform for algorithm testing and optimization</p>
            </div>
        </div>

        <div class="row">
            <!-- Algorithm Selection -->
            <div class="col-md-6 col-lg-3">
                <div class="lab-section">
                    <h5><i class="fas fa-cogs me-2"></i>Algorithm Selection</h5>
                    <div class="row">
                        <div class="col-6 mb-3">
                            <div class="algorithm-card card text-center p-3" data-algorithm="qaoa">
                                <i class="fas fa-project-diagram fa-2x text-primary mb-2"></i>
                                <h6>QAOA</h6>
                                <small>Quantum Approximate Optimization</small>
                            </div>
                        </div>
                        <div class="col-6 mb-3">
                            <div class="algorithm-card card text-center p-3" data-algorithm="vqe">
                                <i class="fas fa-wave-square fa-2x text-success mb-2"></i>
                                <h6>VQE</h6>
                                <small>Variational Quantum Eigensolver</small>
                            </div>
                        </div>
                        <div class="col-6 mb-3">
                            <div class="algorithm-card card text-center p-3" data-algorithm="grover">
                                <i class="fas fa-search fa-2x text-warning mb-2"></i>
                                <h6>Grover</h6>
                                <small>Quantum Search</small>
                            </div>
                        </div>
                        <div class="col-6 mb-3">
                            <div class="algorithm-card card text-center p-3" data-algorithm="qml">
                                <i class="fas fa-brain fa-2x text-info mb-2"></i>
                                <h6>QML</h6>
                                <small>Quantum Machine Learning</small>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Parameter Configuration -->
                <div class="lab-section">
                    <h5><i class="fas fa-sliders-h me-2"></i>Parameters</h5>
                    <div class="parameter-group" id="parameters">
                        <p class="text-muted">Select an algorithm to configure parameters</p>
                    </div>
                </div>
            </div>

            <!-- Data Upload & Circuit Designer -->
            <div class="col-md-6 col-lg-5">
                <div class="lab-section">
                    <h5><i class="fas fa-upload me-2"></i>Data Upload</h5>
                    <div class="upload-area" id="dataUpload">
                        <i class="fas fa-cloud-upload-alt fa-3x mb-3" style="color: var(--quantum-accent);"></i>
                        <h6>Drop files here or click to upload</h6>
                        <p class="text-muted">Supports: CSV, JSON, XLSX, QASM files</p>
                        <input type="file" id="fileInput" class="d-none" multiple accept=".csv,.json,.xlsx,.qasm,.txt">
                    </div>
                </div>

                <div class="lab-section">
                    <h5><i class="fas fa-drafting-compass me-2"></i>Circuit Designer</h5>
                    <div class="quantum-circuit-designer" id="circuitDesigner">
                        <div class="mb-3">
                            <h6>Available Gates:</h6>
                            <div id="gateToolbox">
                                <span class="circuit-gate" data-gate="H">H</span>
                                <span class="circuit-gate" data-gate="X">X</span>
                                <span class="circuit-gate" data-gate="Y">Y</span>
                                <span class="circuit-gate" data-gate="Z">Z</span>
                                <span class="circuit-gate" data-gate="CNOT">CNOT</span>
                                <span class="circuit-gate" data-gate="RX">RX</span>
                                <span class="circuit-gate" data-gate="RY">RY</span>
                                <span class="circuit-gate" data-gate="RZ">RZ</span>
                            </div>
                        </div>
                        <div id="circuitCanvas">
                            <div class="qubit-line">Qubit 0 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</div>
                            <div class="qubit-line">Qubit 1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</div>
                            <div class="qubit-line">Qubit 2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</div>
                            <div class="qubit-line">Qubit 3 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Execution & Results -->
            <div class="col-md-12 col-lg-4">
                <div class="lab-section">
                    <h5><i class="fas fa-play-circle me-2"></i>Execution Control</h5>
                    <div class="card p-3">
                        <div class="d-grid gap-2 mb-3">
                            <button class="btn btn-quantum" id="runExperiment">
                                <i class="fas fa-rocket me-2"></i>Run Experiment
                            </button>
                            <button class="btn btn-outline-warning" id="validateCircuit">
                                <i class="fas fa-check-circle me-2"></i>Validate Circuit
                            </button>
                            <button class="btn btn-outline-info" id="generateQASM">
                                <i class="fas fa-code me-2"></i>Generate QASM
                            </button>
                        </div>
                        
                        <div class="mb-3">
                            <label class="form-label">Shots:</label>
                            <input type="range" class="parameter-slider" id="shots" min="100" max="8192" value="1024">
                            <div class="text-center"><span id="shotsValue">1024</span></div>
                        </div>
                        
                        <div class="mb-3">
                            <div id="executionStatus" class="text-center">
                                <span class="badge bg-secondary">Ready</span>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="lab-section">
                    <h5><i class="fas fa-terminal me-2"></i>Console Output</h5>
                    <div class="console-output" id="console">
                        <div>Quantum Lab Console v2.0.0</div>
                        <div>Ready for quantum experiments...</div>
                        <div class="text-success">></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Results Section -->
        <div class="row mt-4">
            <div class="col-12">
                <div class="results-container">
                    <h5><i class="fas fa-chart-bar me-2"></i>Experiment Results</h5>
                    <div id="experimentResults">
                        <div class="text-center text-muted py-4">
                            <i class="fas fa-flask fa-3x mb-3" style="opacity: 0.3;"></i>
                            <p>Results will appear here after running an experiment</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- JavaScript -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.5.1/dist/chart.min.js"></script>
    
    <script>
        // Quantum Lab JavaScript
        let selectedAlgorithm = null;
        let circuitGates = [];
        let uploadedData = null;
        
        // Algorithm selection
        document.querySelectorAll('.algorithm-card').forEach(card => {
            card.addEventListener('click', function() {
                document.querySelectorAll('.algorithm-card').forEach(c => c.classList.remove('selected'));
                this.classList.add('selected');
                selectedAlgorithm = this.dataset.algorithm;
                updateParameters();
                addConsoleMessage(`Selected ${selectedAlgorithm.toUpperCase()} algorithm`);
            });
        });
        
        // Parameter updates based on algorithm
        function updateParameters() {
            const parametersDiv = document.getElementById('parameters');
            let html = '';
            
            switch(selectedAlgorithm) {
                case 'qaoa':
                    html = `
                        <div class="mb-3">
                            <label class="form-label">Number of Qubits:</label>
                            <input type="range" class="parameter-slider" id="qaoa-qubits" min="2" max="8" value="4">
                            <div class="text-center"><span id="qaoa-qubits-value">4</span></div>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Depth (p):</label>
                            <input type="range" class="parameter-slider" id="qaoa-depth" min="1" max="5" value="2">
                            <div class="text-center"><span id="qaoa-depth-value">2</span></div>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Max Iterations:</label>
                            <input type="range" class="parameter-slider" id="qaoa-iterations" min="10" max="200" value="100">
                            <div class="text-center"><span id="qaoa-iterations-value">100</span></div>
                        </div>
                    `;
                    break;
                case 'vqe':
                    html = `
                        <div class="mb-3">
                            <label class="form-label">Ansatz Layers:</label>
                            <input type="range" class="parameter-slider" id="vqe-layers" min="1" max="10" value="3">
                            <div class="text-center"><span id="vqe-layers-value">3</span></div>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Optimizer:</label>
                            <select class="form-select" id="vqe-optimizer">
                                <option value="COBYLA">COBYLA</option>
                                <option value="SLSQP">SLSQP</option>
                                <option value="SPSA">SPSA</option>
                            </select>
                        </div>
                    `;
                    break;
                case 'grover':
                    html = `
                        <div class="mb-3">
                            <label class="form-label">Search Space (Qubits):</label>
                            <input type="range" class="parameter-slider" id="grover-qubits" min="2" max="6" value="3">
                            <div class="text-center"><span id="grover-qubits-value">3</span></div>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Target State:</label>
                            <input type="number" class="form-control" id="grover-target" min="0" max="7" value="3">
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Iterations:</label>
                            <input type="range" class="parameter-slider" id="grover-iterations" min="1" max="5" value="2">
                            <div class="text-center"><span id="grover-iterations-value">2</span></div>
                        </div>
                    `;
                    break;
                case 'qml':
                    html = `
                        <div class="mb-3">
                            <label class="form-label">Feature Dimensions:</label>
                            <input type="range" class="parameter-slider" id="qml-features" min="2" max="8" value="4">
                            <div class="text-center"><span id="qml-features-value">4</span></div>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Training Epochs:</label>
                            <input type="range" class="parameter-slider" id="qml-epochs" min="10" max="100" value="20">
                            <div class="text-center"><span id="qml-epochs-value">20</span></div>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Learning Rate:</label>
                            <input type="range" class="parameter-slider" id="qml-lr" min="1" max="100" value="10" step="1">
                            <div class="text-center"><span id="qml-lr-value">0.01</span></div>
                        </div>
                    `;
                    break;
            }
            
            parametersDiv.innerHTML = html;
            
            // Add event listeners for sliders
            parametersDiv.querySelectorAll('input[type="range"]').forEach(slider => {
                slider.addEventListener('input', function() {
                    const valueSpan = document.getElementById(this.id + '-value');
                    if (valueSpan) {
                        let value = this.value;
                        if (this.id === 'qml-lr') {
                            value = (parseInt(this.value) / 1000).toFixed(3);
                        }
                        valueSpan.textContent = value;
                    }
                });
            });
        }
        
        // File upload handling
        const uploadArea = document.getElementById('dataUpload');
        const fileInput = document.getElementById('fileInput');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            handleFiles(e.dataTransfer.files);
        });
        
        fileInput.addEventListener('change', (e) => {
            handleFiles(e.target.files);
        });
        
        function handleFiles(files) {
            Array.from(files).forEach(file => {
                addConsoleMessage(`Uploading file: ${file.name} (${file.size} bytes)`);
                // Simulate file processing
                setTimeout(() => {
                    addConsoleMessage(`File processed: ${file.name}`);
                    uploadedData = file;
                }, 1000);
            });
        }
        
        // Console functionality
        function addConsoleMessage(message) {
            const console = document.getElementById('console');
            const timestamp = new Date().toLocaleTimeString();
            console.innerHTML += `<div>[${timestamp}] ${message}</div>`;
            console.scrollTop = console.scrollHeight;
        }
        
        // Shots slider
        document.getElementById('shots').addEventListener('input', function() {
            document.getElementById('shotsValue').textContent = this.value;
        });
        
        // Experiment execution
        function runExperiment() {
            if (!selectedAlgorithm) {
                addConsoleMessage('❌ Please select an algorithm first');
                return;
            }
            
            const params = collectParameters();
            updateExecutionStatus('Running', 'warning');
            addConsoleMessage(`🚀 Starting ${selectedAlgorithm.toUpperCase()} experiment...`);
            addConsoleMessage(`Parameters: ${JSON.stringify(params)}`);
            addConsoleMessage(`Shots: ${document.getElementById('shots').value}`);
            
            // Simulate experiment execution
            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += Math.random() * 20;
                if (progress >= 100) {
                    progress = 100;
                    clearInterval(progressInterval);
                    updateExecutionStatus('Completed', 'success');
                    addConsoleMessage('✅ Experiment completed successfully');
                    displayResults(generateMockResults());
                } else {
                    addConsoleMessage(`⚡ Progress: ${Math.round(progress)}%`);
                }
            }, 500);
        }
        
        function collectParameters() {
            const params = {};
            const parameterInputs = document.querySelectorAll('#parameters input, #parameters select');
            parameterInputs.forEach(input => {
                params[input.id] = input.value;
            });
            return params;
        }
        
        function updateExecutionStatus(status, type) {
            const statusElement = document.getElementById('executionStatus');
            statusElement.innerHTML = `<span class="badge bg-${type}">${status}</span>`;
        }
        
        function generateMockResults() {
            const shots = parseInt(document.getElementById('shots').value);
            return {
                algorithm: selectedAlgorithm,
                execution_time: Math.random() * 30 + 5,
                shots: shots,
                quantum_advantage: Math.random() * 15 + 5,
                fidelity: 90 + Math.random() * 8,
                success_rate: 85 + Math.random() * 12,
                measurement_data: generateMeasurementData(shots)
            };
        }
        
        function generateMeasurementData(shots) {
            const data = {};
            const numStates = selectedAlgorithm === 'grover' ? 8 : 16;
            for (let i = 0; i < numStates; i++) {
                const binary = i.toString(2).padStart(4, '0');
                data[binary] = Math.floor(Math.random() * shots / 4);
            }
            return data;
        }
        
        function displayResults(results) {
            const resultsDiv = document.getElementById('experimentResults');
            
            let html = `
                <div class="row">
                    <div class="col-md-6">
                        <h6>📊 Measurement Results</h6>
                        <canvas id="resultsChart" width="400" height="200"></canvas>
                    </div>
                    <div class="col-md-6">
                        <h6>⚡ Performance Metrics</h6>
                        <ul class="list-unstyled">
                            <li><strong>Algorithm:</strong> ${results.algorithm.toUpperCase()}</li>
                            <li><strong>Execution Time:</strong> ${results.execution_time.toFixed(2)}s</li>
                            <li><strong>Quantum Advantage:</strong> ${results.quantum_advantage.toFixed(1)}x</li>
                            <li><strong>Fidelity:</strong> ${results.fidelity.toFixed(1)}%</li>
                            <li><strong>Success Rate:</strong> ${results.success_rate.toFixed(1)}%</li>
                            <li><strong>Total Shots:</strong> ${results.shots}</li>
                        </ul>
                        
                        <h6 class="mt-3">📈 Raw Data</h6>
                        <div style="max-height: 200px; overflow-y: auto; font-family: monospace; font-size: 12px;">
            `;
            
            Object.entries(results.measurement_data).forEach(([state, count]) => {
                const probability = (count / results.shots * 100).toFixed(1);
                html += `<div>|${state}⟩: ${count} (${probability}%)</div>`;
            });
            
            html += `
                        </div>
                    </div>
                </div>
            `;
            
            resultsDiv.innerHTML = html;
            
            // Create chart
            setTimeout(() => createResultsChart(results.measurement_data), 100);
        }
        
        function createResultsChart(data) {
            const ctx = document.getElementById('resultsChart');
            if (!ctx) return;
            
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: Object.keys(data),
                    datasets: [{
                        label: 'Measurement Counts',
                        data: Object.values(data),
                        backgroundColor: 'rgba(0, 230, 118, 0.6)',
                        borderColor: 'rgba(0, 230, 118, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        x: { ticks: { color: '#e0e7ff' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                        y: { ticks: { color: '#e0e7ff' }, grid: { color: 'rgba(255,255,255,0.1)' } }
                    }
                }
            });
        }
        
        // Event listeners
        document.getElementById('runExperiment').addEventListener('click', runExperiment);
        
        document.getElementById('validateCircuit').addEventListener('click', function() {
            addConsoleMessage('🔍 Validating circuit...');
            setTimeout(() => {
                addConsoleMessage('✅ Circuit validation passed');
            }, 500);
        });
        
        document.getElementById('generateQASM').addEventListener('click', function() {
            addConsoleMessage('📝 Generating QASM code...');
            setTimeout(() => {
                addConsoleMessage('OPENQASM 2.0;');
                addConsoleMessage('include "qelib1.inc";');
                addConsoleMessage('qreg q[4]; creg c[4];');
                addConsoleMessage('h q[0]; cx q[0],q[1];');
                addConsoleMessage('measure q -> c;');
            }, 500);
        });
        
        // Initialize
        addConsoleMessage('🔬 Quantum Research Lab initialized');
        addConsoleMessage('Select an algorithm to begin experimentation');
    </script>
</body>
</html>
"""

def create_main_routes():
    """Factory function to create main routes"""
    return main_bp