"""
API Documentation routes for Quantum Trail Platform
"""

from flask import Blueprint, render_template_string, jsonify, request
from flask import current_app as app
import json

# Create docs blueprint
docs_bp = Blueprint('docs', __name__, url_prefix='/docs')

@docs_bp.route('/')
def api_documentation():
    """Main API documentation page"""
    try:
        return render_template_string(API_DOCS_TEMPLATE)
    except Exception as e:
        app.logger.error(f"API docs error: {e}")
        return jsonify({'error': 'API documentation temporarily unavailable'}), 500

@docs_bp.route('/openapi.json')
def openapi_spec():
    """OpenAPI 3.0 specification"""
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Quantum Trail API",
            "description": "Quantum Computing Platform API for Digital Twins and Optimization",
            "version": "2.0.0",
            "contact": {
                "name": "Quantum Trail Support",
                "email": "support@quantumtrail.com"
            }
        },
        "servers": [
            {
                "url": f"http://localhost:8000",
                "description": "Development server"
            }
        ],
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health Check",
                    "description": "Check the health status of the API",
                    "responses": {
                        "200": {
                            "description": "API is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string", "example": "healthy"},
                                            "timestamp": {"type": "string", "format": "date-time"},
                                            "version": {"type": "string", "example": "2.0.0"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/quantum_twins": {
                "get": {
                    "summary": "List Quantum Twins",
                    "description": "Retrieve all quantum digital twins",
                    "responses": {
                        "200": {
                            "description": "List of quantum twins",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {
                                            "$ref": "#/components/schemas/QuantumTwin"
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "post": {
                    "summary": "Create Quantum Twin",
                    "description": "Create a new quantum digital twin",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/CreateQuantumTwin"
                                }
                            }
                        }
                    },
                    "responses": {
                        "201": {
                            "description": "Quantum twin created successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/QuantumTwin"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/quantum_twins/{id}/simulate": {
                "post": {
                    "summary": "Run Quantum Simulation",
                    "description": "Execute a quantum simulation on a digital twin",
                    "parameters": [
                        {
                            "name": "id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "integer"},
                            "description": "Quantum twin ID"
                        }
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "algorithm": {"type": "string", "enum": ["qaoa", "vqe", "qml"], "example": "qaoa"},
                                        "shots": {"type": "integer", "minimum": 1, "maximum": 8192, "example": 1024},
                                        "parameters": {"type": "object", "description": "Algorithm-specific parameters"}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Simulation completed",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/SimulationResult"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/optimize": {
                "post": {
                    "summary": "Quantum Optimization",
                    "description": "Run quantum optimization algorithms",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "algorithm": {"type": "string", "enum": ["qaoa", "vqe"], "example": "qaoa"},
                                        "problem_type": {"type": "string", "enum": ["portfolio", "maxcut", "tsp"], "example": "portfolio"},
                                        "data": {"type": "object", "description": "Problem-specific data"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "QuantumTwin": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "example": 1},
                        "name": {"type": "string", "example": "Portfolio Twin"},
                        "description": {"type": "string", "example": "Quantum twin for portfolio optimization"},
                        "twin_type": {"type": "string", "example": "portfolio_optimization"},
                        "parameters": {"type": "object", "description": "Twin configuration parameters"},
                        "is_active": {"type": "boolean", "example": True},
                        "created_at": {"type": "string", "format": "date-time"},
                        "updated_at": {"type": "string", "format": "date-time"}
                    }
                },
                "CreateQuantumTwin": {
                    "type": "object",
                    "required": ["name", "twin_type"],
                    "properties": {
                        "name": {"type": "string", "example": "My Portfolio Twin"},
                        "description": {"type": "string", "example": "Twin for optimizing investment portfolio"},
                        "twin_type": {"type": "string", "enum": ["portfolio_optimization", "route_optimization", "quantum_ml"], "example": "portfolio_optimization"},
                        "parameters": {
                            "type": "object",
                            "example": {
                                "assets": ["AAPL", "GOOGL", "MSFT"],
                                "risk_tolerance": 0.1,
                                "expected_returns": [0.12, 0.15, 0.10]
                            }
                        }
                    }
                },
                "SimulationResult": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "example": 1},
                        "twin_id": {"type": "integer", "example": 1},
                        "algorithm": {"type": "string", "example": "qaoa"},
                        "status": {"type": "string", "enum": ["completed", "running", "failed"], "example": "completed"},
                        "result": {"type": "object", "description": "Algorithm-specific results"},
                        "execution_time": {"type": "number", "example": 15.7},
                        "created_at": {"type": "string", "format": "date-time"}
                    }
                }
            }
        }
    }
    
    return jsonify(spec)

@docs_bp.route('/examples')
def api_examples():
    """API usage examples page"""
    try:
        return render_template_string(API_EXAMPLES_TEMPLATE)
    except Exception as e:
        app.logger.error(f"API examples error: {e}")
        return jsonify({'error': 'API examples temporarily unavailable'}), 500

# HTML Templates
API_DOCS_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Trail API Documentation</title>
    <style>
        body { font-family: 'Monaco', 'Menlo', monospace; margin: 0; padding: 20px; background: #1e1e1e; color: #d4d4d4; line-height: 1.6; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2d3748; padding: 30px; border-radius: 8px; margin-bottom: 30px; text-align: center; }
        .header h1 { color: #63b3ed; margin: 0 0 10px 0; font-size: 2.5em; }
        .header p { color: #a0aec0; margin: 0; font-size: 1.1em; }
        .section { background: #2d3748; padding: 25px; border-radius: 8px; margin-bottom: 20px; }
        .section h2 { color: #68d391; margin-top: 0; border-bottom: 2px solid #4a5568; padding-bottom: 10px; }
        .section h3 { color: #fbb6ce; }
        .endpoint { background: #1a202c; padding: 15px; border-radius: 5px; margin: 15px 0; border-left: 4px solid #63b3ed; }
        .method { display: inline-block; padding: 4px 8px; border-radius: 3px; font-weight: bold; font-size: 0.8em; margin-right: 10px; }
        .method.get { background: #48bb78; color: white; }
        .method.post { background: #ed8936; color: white; }
        .method.put { background: #4299e1; color: white; }
        .method.delete { background: #f56565; color: white; }
        .code { background: #1a202c; padding: 15px; border-radius: 5px; overflow-x: auto; border: 1px solid #4a5568; }
        .code pre { margin: 0; color: #e2e8f0; }
        .nav-links { text-align: center; margin: 20px 0; }
        .nav-links a { display: inline-block; margin: 0 10px; padding: 10px 20px; background: #4299e1; color: white; text-decoration: none; border-radius: 5px; font-weight: bold; }
        .nav-links a:hover { background: #3182ce; }
        .parameter { background: #2d3748; padding: 10px; margin: 5px 0; border-radius: 3px; border-left: 3px solid #ed8936; }
        .response { background: #1a202c; padding: 10px; margin: 5px 0; border-radius: 3px; border-left: 3px solid #48bb78; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📚 Quantum Trail API Documentation</h1>
            <p>Complete REST API reference for the Quantum Computing Platform</p>
        </div>
        
        <div class="nav-links">
            <a href="/docs/">Documentation</a>
            <a href="/docs/openapi.json">OpenAPI Spec</a>
            <a href="/docs/examples">Examples</a>
            <a href="/graphql">GraphQL</a>
            <a href="/admin">Admin Panel</a>
            <a href="/">← Home</a>
        </div>
        
        <div class="section">
            <h2>🚀 Getting Started</h2>
            <p>The Quantum Trail API provides programmatic access to quantum computing capabilities including:</p>
            <ul>
                <li><strong>Quantum Digital Twins</strong> - Create and manage quantum simulations of real-world systems</li>
                <li><strong>Optimization Algorithms</strong> - Run QAOA and VQE for complex optimization problems</li>
                <li><strong>Machine Learning</strong> - Execute quantum machine learning algorithms</li>
                <li><strong>Real-time Monitoring</strong> - WebSocket connections for live updates</li>
            </ul>
            
            <h3>Base URL</h3>
            <div class="code">
                <pre>http://localhost:8000</pre>
            </div>
            
            <h3>Authentication</h3>
            <p>Currently in development mode - no authentication required. Production deployments will use JWT tokens.</p>
        </div>
        
        <div class="section">
            <h2>🔍 Core Endpoints</h2>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/health</strong>
                <p>Check API health status and version information.</p>
                <div class="response">
                    <strong>Response 200:</strong>
                    <div class="code">
                        <pre>{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "2.0.0"
}</pre>
                    </div>
                </div>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/api/quantum_twins</strong>
                <p>Retrieve all quantum digital twins.</p>
                <div class="response">
                    <strong>Response 200:</strong> Array of quantum twin objects
                </div>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/api/quantum_twins</strong>
                <p>Create a new quantum digital twin.</p>
                <div class="parameter">
                    <strong>Request Body:</strong>
                    <div class="code">
                        <pre>{
  "name": "Portfolio Twin",
  "description": "Twin for portfolio optimization", 
  "twin_type": "portfolio_optimization",
  "parameters": {
    "assets": ["AAPL", "GOOGL", "MSFT"],
    "risk_tolerance": 0.1,
    "expected_returns": [0.12, 0.15, 0.10]
  }
}</pre>
                    </div>
                </div>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/api/quantum_twins/{id}/simulate</strong>
                <p>Execute a quantum simulation on a digital twin.</p>
                <div class="parameter">
                    <strong>Path Parameters:</strong>
                    <ul><li><code>id</code> (integer) - Quantum twin ID</li></ul>
                </div>
                <div class="parameter">
                    <strong>Request Body:</strong>
                    <div class="code">
                        <pre>{
  "algorithm": "qaoa",
  "shots": 1024,
  "parameters": {
    "p": 2,
    "max_iterations": 100
  }
}</pre>
                    </div>
                </div>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/api/optimize</strong>
                <p>Run quantum optimization algorithms.</p>
                <div class="parameter">
                    <strong>Request Body:</strong>
                    <div class="code">
                        <pre>{
  "algorithm": "qaoa",
  "problem_type": "portfolio",
  "data": {
    "expected_returns": [0.12, 0.15, 0.10],
    "covariance_matrix": [[0.04, 0.006], [0.006, 0.09]],
    "risk_tolerance": 0.1
  }
}</pre>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>⚛️ Quantum Algorithms</h2>
            <h3>Supported Algorithms</h3>
            <ul>
                <li><strong>QAOA</strong> - Quantum Approximate Optimization Algorithm for combinatorial problems</li>
                <li><strong>VQE</strong> - Variational Quantum Eigensolver for ground state problems</li>
                <li><strong>VQC</strong> - Variational Quantum Classifier for machine learning</li>
                <li><strong>QNN</strong> - Quantum Neural Networks for deep learning applications</li>
            </ul>
            
            <h3>Problem Types</h3>
            <ul>
                <li><code>portfolio_optimization</code> - Financial portfolio optimization</li>
                <li><code>maxcut</code> - Maximum cut graph problems</li>
                <li><code>tsp</code> - Traveling salesman problem</li>
                <li><code>quantum_ml</code> - Quantum machine learning tasks</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>🔄 WebSocket API</h2>
            <p>Real-time updates via WebSocket connections:</p>
            <div class="code">
                <pre>// Connect to WebSocket
const socket = io('http://localhost:8000');

// Listen for simulation updates
socket.on('simulation_update', (data) => {
    console.log('Progress:', data.progress);
});

// Listen for results
socket.on('optimization_result', (result) => {
    console.log('Result:', result);
});</pre>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Response Formats</h2>
            <p>All responses are in JSON format. Common response structures:</p>
            
            <h3>Success Response</h3>
            <div class="code">
                <pre>{
  "status": "success",
  "data": { /* response data */ },
  "timestamp": "2024-01-15T10:30:00Z"
}</pre>
            </div>
            
            <h3>Error Response</h3>
            <div class="code">
                <pre>{
  "status": "error",
  "message": "Error description",
  "code": "ERROR_CODE",
  "timestamp": "2024-01-15T10:30:00Z"
}</pre>
            </div>
        </div>
        
        <div class="section">
            <h2>🛠️ Development Tools</h2>
            <ul>
                <li><a href="/docs/openapi.json">OpenAPI 3.0 Specification</a> - Machine-readable API spec</li>
                <li><a href="/graphql">GraphQL Playground</a> - Interactive GraphQL interface</li>
                <li><a href="/admin">Admin Dashboard</a> - System management interface</li>
                <li><a href="/docs/examples">Code Examples</a> - Usage examples in multiple languages</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>💡 Support</h2>
            <p>For API support and questions:</p>
            <ul>
                <li>📧 Email: support@quantumtrail.com</li>
                <li>🐙 GitHub: Issues and discussions</li>
                <li>📚 Full Documentation: COMPREHENSIVE_DOCUMENTATION.md</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

API_EXAMPLES_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Trail API Examples</title>
    <style>
        body { font-family: 'Monaco', 'Menlo', monospace; margin: 0; padding: 20px; background: #1e1e1e; color: #d4d4d4; line-height: 1.6; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2d3748; padding: 30px; border-radius: 8px; margin-bottom: 30px; text-align: center; }
        .header h1 { color: #63b3ed; margin: 0; }
        .section { background: #2d3748; padding: 25px; border-radius: 8px; margin-bottom: 20px; }
        .section h2 { color: #68d391; margin-top: 0; }
        .section h3 { color: #fbb6ce; }
        .code { background: #1a202c; padding: 15px; border-radius: 5px; overflow-x: auto; border: 1px solid #4a5568; margin: 10px 0; }
        .code pre { margin: 0; color: #e2e8f0; }
        .nav-links { text-align: center; margin: 20px 0; }
        .nav-links a { display: inline-block; margin: 0 10px; padding: 10px 20px; background: #4299e1; color: white; text-decoration: none; border-radius: 5px; }
        .language-tabs { display: flex; margin: 10px 0; }
        .tab { padding: 8px 16px; background: #4a5568; color: #e2e8f0; cursor: pointer; border-radius: 3px 3px 0 0; margin-right: 2px; }
        .tab.active { background: #63b3ed; color: #1a202c; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 API Examples</h1>
            <p>Complete code examples for the Quantum Trail API</p>
        </div>
        
        <div class="nav-links">
            <a href="/docs/">Documentation</a>
            <a href="/docs/examples">Examples</a>
            <a href="/graphql">GraphQL</a>
            <a href="/">← Home</a>
        </div>
        
        <div class="section">
            <h2>🐍 Python Examples</h2>
            
            <h3>Create a Quantum Twin</h3>
            <div class="code">
                <pre>import requests
import json

# Create a new quantum twin
url = "http://localhost:8000/api/quantum_twins"
data = {
    "name": "Portfolio Optimization Twin",
    "description": "Quantum twin for financial portfolio optimization",
    "twin_type": "portfolio_optimization",
    "parameters": {
        "assets": ["AAPL", "GOOGL", "MSFT", "AMZN"],
        "risk_tolerance": 0.15,
        "expected_returns": [0.12, 0.15, 0.10, 0.14],
        "covariance_matrix": [
            [0.04, 0.006, 0.004, 0.002],
            [0.006, 0.09, 0.005, 0.003],
            [0.004, 0.005, 0.025, 0.001],
            [0.002, 0.003, 0.001, 0.016]
        ]
    }
}

response = requests.post(url, json=data)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")

# Get the created twin ID
twin_id = response.json().get('id')</pre>
            </div>
            
            <h3>Run Quantum Optimization</h3>
            <div class="code">
                <pre># Run QAOA optimization on the twin
simulation_url = f"http://localhost:8000/api/quantum_twins/{twin_id}/simulate"
simulation_data = {
    "algorithm": "qaoa",
    "shots": 1024,
    "parameters": {
        "p": 2,  # QAOA depth
        "max_iterations": 100
    }
}

response = requests.post(simulation_url, json=simulation_data)
result = response.json()

print(f"Optimization Status: {result['status']}")
print(f"Optimal Weights: {result['result']['optimal_weights']}")
print(f"Expected Return: {result['result']['expected_return']}")
print(f"Portfolio Risk: {result['result']['portfolio_risk']}")</pre>
            </div>
        </div>
        
        <div class="section">
            <h2>🌐 JavaScript Examples</h2>
            
            <h3>Fetch API Usage</h3>
            <div class="code">
                <pre>// Create quantum twin using fetch
async function createQuantumTwin() {
    const response = await fetch('http://localhost:8000/api/quantum_twins', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            name: 'Risk Analysis Twin',
            twin_type: 'portfolio_optimization',
            parameters: {
                assets: ['BTC', 'ETH', 'SOL'],
                risk_tolerance: 0.2
            }
        })
    });
    
    const twin = await response.json();
    console.log('Created twin:', twin);
    return twin.id;
}

// Run simulation
async function runSimulation(twinId) {
    const response = await fetch(`http://localhost:8000/api/quantum_twins/${twinId}/simulate`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            algorithm: 'qaoa',
            shots: 512
        })
    });
    
    const result = await response.json();
    console.log('Simulation result:', result);
}</pre>
            </div>
            
            <h3>WebSocket Real-time Updates</h3>
            <div class="code">
                <pre>// Connect to WebSocket for real-time updates
const socket = io('http://localhost:8000');

// Listen for simulation progress
socket.on('simulation_update', (data) => {
    console.log(`Progress: ${data.progress}%`);
    updateProgressBar(data.progress);
});

// Listen for optimization results
socket.on('optimization_result', (result) => {
    console.log('New optimization result:', result);
    displayResult(result);
});

// Listen for system health updates
socket.on('system_health', (health) => {
    updateHealthIndicator(health);
});

// Error handling
socket.on('error', (error) => {
    console.error('WebSocket error:', error);
});

socket.on('connect', () => {
    console.log('Connected to Quantum Trail WebSocket');
});

socket.on('disconnect', () => {
    console.log('Disconnected from WebSocket');
});</pre>
            </div>
        </div>
        
        <div class="section">
            <h2>🐚 cURL Examples</h2>
            
            <h3>Health Check</h3>
            <div class="code">
                <pre>curl -X GET http://localhost:8000/health</pre>
            </div>
            
            <h3>Create Quantum Twin</h3>
            <div class="code">
                <pre>curl -X POST http://localhost:8000/api/quantum_twins \\
  -H "Content-Type: application/json" \\
  -d '{
    "name": "Test Twin",
    "twin_type": "portfolio_optimization",
    "parameters": {
      "assets": ["AAPL", "MSFT"],
      "risk_tolerance": 0.1
    }
  }'</pre>
            </div>
            
            <h3>Run Optimization</h3>
            <div class="code">
                <pre>curl -X POST http://localhost:8000/api/optimize \\
  -H "Content-Type: application/json" \\
  -d '{
    "algorithm": "qaoa",
    "problem_type": "portfolio",
    "data": {
      "expected_returns": [0.12, 0.15],
      "covariance_matrix": [[0.04, 0.006], [0.006, 0.09]],
      "risk_tolerance": 0.1
    }
  }'</pre>
            </div>
        </div>
        
        <div class="section">
            <h2>🔧 GraphQL Examples</h2>
            
            <h3>Query Quantum Twins</h3>
            <div class="code">
                <pre>query GetQuantumTwins {
  quantumTwins {
    id
    name
    twinType
    status
    parameters
    createdAt
    updatedAt
  }
}</pre>
            </div>
            
            <h3>Create and Run Simulation</h3>
            <div class="code">
                <pre>mutation CreateSimulation($input: SimulationInput!) {
  createSimulation(input: $input) {
    id
    status
    algorithm
    result
    executionTime
    createdAt
  }
}

# Variables
{
  "input": {
    "twinId": 1,
    "algorithm": "qaoa",
    "shots": 1024,
    "parameters": {
      "p": 2
    }
  }
}</pre>
            </div>
        </div>
        
        <div class="section">
            <h2>🚀 Complete Application Example</h2>
            
            <h3>Python Portfolio Optimizer</h3>
            <div class="code">
                <pre>#!/usr/bin/env python3
# Complete example: Portfolio optimization using Quantum Trail API
import requests
import time
import json

class QuantumPortfolioOptimizer:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
    
    def create_portfolio_twin(self, assets, expected_returns, covariance_matrix, risk_tolerance=0.1):
        # Create a quantum twin for portfolio optimization
        data = {
            "name": f"Portfolio Twin - {len(assets)} assets",
            "description": "Quantum portfolio optimization",
            "twin_type": "portfolio_optimization",
            "parameters": {
                "assets": assets,
                "expected_returns": expected_returns,
                "covariance_matrix": covariance_matrix,
                "risk_tolerance": risk_tolerance
            }
        }
        
        response = self.session.post(f"{self.base_url}/api/quantum_twins", json=data)
        response.raise_for_status()
        return response.json()
    
    def optimize_portfolio(self, twin_id, algorithm="qaoa", shots=1024):
        # Run quantum optimization on the portfolio twin
        data = {
            "algorithm": algorithm,
            "shots": shots,
            "parameters": {
                "p": 2 if algorithm == "qaoa" else 1,
                "max_iterations": 100
            }
        }
        
        response = self.session.post(
            f"{self.base_url}/api/quantum_twins/{twin_id}/simulate", 
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def get_health_status(self):
        # Check API health
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

# Example usage
if __name__ == "__main__":
    optimizer = QuantumPortfolioOptimizer()
    
    # Check API health
    print("Checking API health...")
    health = optimizer.get_health_status()
    print(f"API Status: {health['status']}")
    
    # Define portfolio
    assets = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    expected_returns = [0.12, 0.15, 0.10, 0.18]
    covariance_matrix = [
        [0.040, 0.006, 0.004, 0.012],
        [0.006, 0.090, 0.005, 0.020],
        [0.004, 0.005, 0.025, 0.008],
        [0.012, 0.020, 0.008, 0.160]
    ]
    
    # Create quantum twin
    print("Creating quantum twin...")
    twin = optimizer.create_portfolio_twin(assets, expected_returns, covariance_matrix)
    print(f"Created twin: {twin['name']} (ID: {twin['id']})")
    
    # Run optimization
    print("Running quantum optimization...")
    result = optimizer.optimize_portfolio(twin['id'])
    
    if result['status'] == 'completed':
        print("Optimization completed!")
        print(f"Optimal weights: {result['result']['optimal_weights']}")
        print(f"Expected return: {result['result']['expected_return']:.4f}")
        print(f"Portfolio risk: {result['result']['portfolio_risk']:.4f}")
        print(f"Sharpe ratio: {result['result']['sharpe_ratio']:.4f}")
    else:
        print(f"Optimization status: {result['status']}")</pre>
            </div>
        </div>
    </div>
</body>
</html>
"""

def create_docs_routes():
    """Factory function to create docs routes"""
    return docs_bp