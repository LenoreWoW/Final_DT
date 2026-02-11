/**
 * Interactive Quantum Circuit Designer
 * Advanced visual circuit designer with drag-and-drop functionality
 */

class QuantumCircuitDesigner {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.canvas = null;
        this.ctx = null;
        this.circuit = [];
        this.maxQubits = 12;
        this.currentQubits = 4;
        this.gridSize = 60;
        this.gateSize = 40;
        this.isDragging = false;
        this.selectedGate = null;
        this.dragOffset = { x: 0, y: 0 };
        
        // Available quantum gates
        this.availableGates = {
            'H': { name: 'Hadamard', color: '#FF6B6B', single: true },
            'X': { name: 'Pauli-X', color: '#4ECDC4', single: true },
            'Y': { name: 'Pauli-Y', color: '#45B7D1', single: true },
            'Z': { name: 'Pauli-Z', color: '#96CEB4', single: true },
            'S': { name: 'Phase', color: '#FECA57', single: true },
            'T': { name: 'T Gate', color: '#FF9FF3', single: true },
            'RX': { name: 'Rotation-X', color: '#54A0FF', single: true, parameterized: true },
            'RY': { name: 'Rotation-Y', color: '#5F27CD', single: true, parameterized: true },
            'RZ': { name: 'Rotation-Z', color: '#00D2D3', single: true, parameterized: true },
            'CNOT': { name: 'CNOT', color: '#FF7675', two_qubit: true },
            'CZ': { name: 'Controlled-Z', color: '#A29BFE', two_qubit: true },
            'SWAP': { name: 'SWAP', color: '#FD79A8', two_qubit: true },
            'MEASURE': { name: 'Measure', color: '#2D3436', measurement: true }
        };
        
        // Circuit execution results
        this.executionResults = null;
        this.isExecuting = false;
        
        this.init();
    }
    
    init() {
        this.createInterface();
        this.setupEventListeners();
        this.drawCircuit();
    }
    
    createInterface() {
        this.container.innerHTML = `
            <div class="circuit-designer">
                <div class="designer-header">
                    <h3>Quantum Circuit Designer</h3>
                    <div class="circuit-controls">
                        <button id="clearCircuit" class="btn btn-warning">Clear Circuit</button>
                        <button id="executeCircuit" class="btn btn-primary">Execute Circuit</button>
                        <button id="optimizeCircuit" class="btn btn-info">Optimize</button>
                        <button id="exportCircuit" class="btn btn-success">Export</button>
                    </div>
                </div>
                
                <div class="designer-main">
                    <div class="gate-palette">
                        <h4>Gate Palette</h4>
                        <div class="gate-categories">
                            <div class="gate-category">
                                <h5>Single Qubit Gates</h5>
                                <div class="gates-grid" id="singleQubitGates"></div>
                            </div>
                            <div class="gate-category">
                                <h5>Two Qubit Gates</h5>
                                <div class="gates-grid" id="twoQubitGates"></div>
                            </div>
                            <div class="gate-category">
                                <h5>Measurement</h5>
                                <div class="gates-grid" id="measurementGates"></div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="circuit-workspace">
                        <div class="workspace-controls">
                            <label>
                                Qubits: 
                                <input type="range" id="qubitSlider" min="2" max="12" value="4" />
                                <span id="qubitCount">4</span>
                            </label>
                        </div>
                        <canvas id="circuitCanvas" width="800" height="400"></canvas>
                        <div class="circuit-info">
                            <div class="info-panel">
                                <h5>Circuit Information</h5>
                                <div id="circuitStats">
                                    <p>Gates: <span id="gateCount">0</span></p>
                                    <p>Depth: <span id="circuitDepth">0</span></p>
                                    <p>Qubits: <span id="activeQubits">4</span></p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="execution-results" id="executionResults" style="display: none;">
                    <h4>Execution Results</h4>
                    <div class="results-content">
                        <div class="probability-histogram">
                            <canvas id="resultsCanvas" width="600" height="300"></canvas>
                        </div>
                        <div class="results-stats">
                            <div id="executionStats"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Get canvas and context
        this.canvas = document.getElementById('circuitCanvas');
        this.ctx = this.canvas.getContext('2d');
        
        // Create gate palette
        this.createGatePalette();
    }
    
    createGatePalette() {
        const singleQubitGates = document.getElementById('singleQubitGates');
        const twoQubitGates = document.getElementById('twoQubitGates');
        const measurementGates = document.getElementById('measurementGates');
        
        Object.entries(this.availableGates).forEach(([symbol, gate]) => {
            const gateElement = document.createElement('div');
            gateElement.className = 'gate-item';
            gateElement.draggable = true;
            gateElement.dataset.gateType = symbol;
            gateElement.style.backgroundColor = gate.color;
            gateElement.innerHTML = `
                <div class="gate-symbol">${symbol}</div>
                <div class="gate-name">${gate.name}</div>
            `;
            
            if (gate.single) {
                singleQubitGates.appendChild(gateElement);
            } else if (gate.two_qubit) {
                twoQubitGates.appendChild(gateElement);
            } else if (gate.measurement) {
                measurementGates.appendChild(gateElement);
            }
        });
    }
    
    setupEventListeners() {
        // Canvas events
        this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.onMouseUp(e));
        this.canvas.addEventListener('contextmenu', (e) => this.onRightClick(e));
        
        // Drag and drop from palette
        document.querySelectorAll('.gate-item').forEach(item => {
            item.addEventListener('dragstart', (e) => this.onDragStart(e));
        });
        
        this.canvas.addEventListener('dragover', (e) => e.preventDefault());
        this.canvas.addEventListener('drop', (e) => this.onDrop(e));
        
        // Controls
        document.getElementById('clearCircuit').addEventListener('click', () => this.clearCircuit());
        document.getElementById('executeCircuit').addEventListener('click', () => this.executeCircuit());
        document.getElementById('optimizeCircuit').addEventListener('click', () => this.optimizeCircuit());
        document.getElementById('exportCircuit').addEventListener('click', () => this.exportCircuit());
        
        // Qubit slider
        const slider = document.getElementById('qubitSlider');
        slider.addEventListener('input', (e) => {
            this.currentQubits = parseInt(e.target.value);
            document.getElementById('qubitCount').textContent = this.currentQubits;
            document.getElementById('activeQubits').textContent = this.currentQubits;
            this.drawCircuit();
        });
    }
    
    onDragStart(e) {
        e.dataTransfer.setData('text/plain', e.target.dataset.gateType);
    }
    
    onDrop(e) {
        e.preventDefault();
        const gateType = e.dataTransfer.getData('text/plain');
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        this.addGateAtPosition(gateType, x, y);
    }
    
    onMouseDown(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        const gate = this.findGateAtPosition(x, y);
        if (gate) {
            this.isDragging = true;
            this.selectedGate = gate;
            this.dragOffset = { x: x - gate.x, y: y - gate.y };
        }
    }
    
    onMouseMove(e) {
        if (this.isDragging && this.selectedGate) {
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            this.selectedGate.x = x - this.dragOffset.x;
            this.selectedGate.y = y - this.dragOffset.y;
            
            // Snap to grid
            this.snapToGrid(this.selectedGate);
            
            this.drawCircuit();
        }
    }
    
    onMouseUp(e) {
        this.isDragging = false;
        this.selectedGate = null;
        this.updateCircuitStats();
    }
    
    onRightClick(e) {
        e.preventDefault();
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        const gate = this.findGateAtPosition(x, y);
        if (gate) {
            this.removeGate(gate);
        }
    }
    
    addGateAtPosition(gateType, x, y) {
        const qubit = this.getQubitFromY(y);
        if (qubit < 0 || qubit >= this.currentQubits) return;
        
        const column = this.getColumnFromX(x);
        if (column < 0) return;
        
        const gate = {
            type: gateType,
            x: column * this.gridSize + 100,
            y: qubit * this.gridSize + 50,
            qubit: qubit,
            column: column,
            id: Date.now() + Math.random(),
            parameters: this.availableGates[gateType].parameterized ? { angle: Math.PI/4 } : null
        };
        
        // Handle two-qubit gates
        if (this.availableGates[gateType].two_qubit) {
            gate.controlQubit = qubit;
            gate.targetQubit = Math.min(qubit + 1, this.currentQubits - 1);
            gate.y = Math.min(gate.controlQubit, gate.targetQubit) * this.gridSize + 50;
        }
        
        this.circuit.push(gate);
        this.drawCircuit();
        this.updateCircuitStats();
    }
    
    findGateAtPosition(x, y) {
        return this.circuit.find(gate => {
            const dx = x - gate.x;
            const dy = y - gate.y;
            return Math.abs(dx) < this.gateSize/2 && Math.abs(dy) < this.gateSize/2;
        });
    }
    
    removeGate(gate) {
        this.circuit = this.circuit.filter(g => g.id !== gate.id);
        this.drawCircuit();
        this.updateCircuitStats();
    }
    
    snapToGrid(gate) {
        const column = Math.round((gate.x - 100) / this.gridSize);
        const qubit = Math.round((gate.y - 50) / this.gridSize);
        
        gate.x = Math.max(100, column * this.gridSize + 100);
        gate.y = Math.max(50, Math.min(qubit * this.gridSize + 50, (this.currentQubits - 1) * this.gridSize + 50));
        gate.column = Math.max(0, column);
        gate.qubit = Math.max(0, Math.min(qubit, this.currentQubits - 1));
    }
    
    getQubitFromY(y) {
        return Math.round((y - 50) / this.gridSize);
    }
    
    getColumnFromX(x) {
        return Math.max(0, Math.round((x - 100) / this.gridSize));
    }
    
    drawCircuit() {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw qubit lines
        this.drawQubitLines();
        
        // Draw gates
        this.circuit.forEach(gate => this.drawGate(gate));
        
        // Draw grid
        this.drawGrid();
    }
    
    drawQubitLines() {
        this.ctx.strokeStyle = '#2C3E50';
        this.ctx.lineWidth = 2;
        
        for (let q = 0; q < this.currentQubits; q++) {
            const y = q * this.gridSize + 50;
            
            this.ctx.beginPath();
            this.ctx.moveTo(50, y);
            this.ctx.lineTo(this.canvas.width - 50, y);
            this.ctx.stroke();
            
            // Qubit labels
            this.ctx.fillStyle = '#2C3E50';
            this.ctx.font = '14px Arial';
            this.ctx.fillText(`|${q}⟩`, 10, y + 5);
        }
    }
    
    drawGrid() {
        this.ctx.strokeStyle = '#ECF0F1';
        this.ctx.lineWidth = 1;
        
        // Vertical grid lines
        for (let x = 100; x < this.canvas.width - 50; x += this.gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 20);
            this.ctx.lineTo(x, this.currentQubits * this.gridSize + 20);
            this.ctx.stroke();
        }
    }
    
    drawGate(gate) {
        const gateInfo = this.availableGates[gate.type];
        
        if (gateInfo.two_qubit) {
            this.drawTwoQubitGate(gate);
        } else if (gateInfo.measurement) {
            this.drawMeasurementGate(gate);
        } else {
            this.drawSingleQubitGate(gate);
        }
    }
    
    drawSingleQubitGate(gate) {
        const gateInfo = this.availableGates[gate.type];
        
        // Gate background
        this.ctx.fillStyle = gateInfo.color;
        this.ctx.fillRect(gate.x - this.gateSize/2, gate.y - this.gateSize/2, this.gateSize, this.gateSize);
        
        // Gate border
        this.ctx.strokeStyle = '#2C3E50';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(gate.x - this.gateSize/2, gate.y - this.gateSize/2, this.gateSize, this.gateSize);
        
        // Gate symbol
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.font = 'bold 16px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(gate.type, gate.x, gate.y + 5);
        
        // Parameter display for parameterized gates
        if (gate.parameters) {
            this.ctx.fillStyle = '#2C3E50';
            this.ctx.font = '10px Arial';
            this.ctx.fillText(`θ=${gate.parameters.angle.toFixed(2)}`, gate.x, gate.y + 25);
        }
    }
    
    drawTwoQubitGate(gate) {
        const controlY = gate.controlQubit * this.gridSize + 50;
        const targetY = gate.targetQubit * this.gridSize + 50;
        
        // Control qubit (dot)
        this.ctx.fillStyle = '#2C3E50';
        this.ctx.beginPath();
        this.ctx.arc(gate.x, controlY, 8, 0, 2 * Math.PI);
        this.ctx.fill();
        
        // Connection line
        this.ctx.strokeStyle = '#2C3E50';
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        this.ctx.moveTo(gate.x, controlY);
        this.ctx.lineTo(gate.x, targetY);
        this.ctx.stroke();
        
        // Target gate
        if (gate.type === 'CNOT') {
            // X gate symbol
            this.ctx.strokeStyle = '#2C3E50';
            this.ctx.lineWidth = 3;
            this.ctx.beginPath();
            this.ctx.arc(gate.x, targetY, 15, 0, 2 * Math.PI);
            this.ctx.stroke();
            
            this.ctx.beginPath();
            this.ctx.moveTo(gate.x - 8, targetY - 8);
            this.ctx.lineTo(gate.x + 8, targetY + 8);
            this.ctx.moveTo(gate.x + 8, targetY - 8);
            this.ctx.lineTo(gate.x - 8, targetY + 8);
            this.ctx.stroke();
        } else if (gate.type === 'CZ') {
            // Z gate (dot)
            this.ctx.fillStyle = '#2C3E50';
            this.ctx.beginPath();
            this.ctx.arc(gate.x, targetY, 8, 0, 2 * Math.PI);
            this.ctx.fill();
        } else if (gate.type === 'SWAP') {
            // SWAP symbols
            this.ctx.strokeStyle = '#2C3E50';
            this.ctx.lineWidth = 3;
            
            // Control qubit X
            this.ctx.beginPath();
            this.ctx.moveTo(gate.x - 8, controlY - 8);
            this.ctx.lineTo(gate.x + 8, controlY + 8);
            this.ctx.moveTo(gate.x + 8, controlY - 8);
            this.ctx.lineTo(gate.x - 8, controlY + 8);
            this.ctx.stroke();
            
            // Target qubit X
            this.ctx.beginPath();
            this.ctx.moveTo(gate.x - 8, targetY - 8);
            this.ctx.lineTo(gate.x + 8, targetY + 8);
            this.ctx.moveTo(gate.x + 8, targetY - 8);
            this.ctx.lineTo(gate.x - 8, targetY + 8);
            this.ctx.stroke();
        }
    }
    
    drawMeasurementGate(gate) {
        // Measurement box
        this.ctx.fillStyle = '#2C3436';
        this.ctx.fillRect(gate.x - this.gateSize/2, gate.y - this.gateSize/2, this.gateSize, this.gateSize);
        
        this.ctx.strokeStyle = '#FFFFFF';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(gate.x - this.gateSize/2, gate.y - this.gateSize/2, this.gateSize, this.gateSize);
        
        // Measurement symbol (arc and arrow)
        this.ctx.strokeStyle = '#FFFFFF';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.arc(gate.x, gate.y + 5, 12, Math.PI, 0);
        this.ctx.stroke();
        
        // Arrow
        this.ctx.beginPath();
        this.ctx.moveTo(gate.x + 8, gate.y - 5);
        this.ctx.lineTo(gate.x + 12, gate.y - 8);
        this.ctx.lineTo(gate.x + 12, gate.y - 2);
        this.ctx.stroke();
    }
    
    updateCircuitStats() {
        const gateCount = this.circuit.length;
        const depth = this.calculateCircuitDepth();
        
        document.getElementById('gateCount').textContent = gateCount;
        document.getElementById('circuitDepth').textContent = depth;
    }
    
    calculateCircuitDepth() {
        if (this.circuit.length === 0) return 0;
        
        const qubitDepths = new Array(this.currentQubits).fill(0);
        
        // Sort gates by column
        const sortedGates = [...this.circuit].sort((a, b) => a.column - b.column);
        
        sortedGates.forEach(gate => {
            if (this.availableGates[gate.type].two_qubit) {
                const maxDepth = Math.max(qubitDepths[gate.controlQubit], qubitDepths[gate.targetQubit]);
                qubitDepths[gate.controlQubit] = maxDepth + 1;
                qubitDepths[gate.targetQubit] = maxDepth + 1;
            } else {
                qubitDepths[gate.qubit] += 1;
            }
        });
        
        return Math.max(...qubitDepths);
    }
    
    clearCircuit() {
        this.circuit = [];
        this.drawCircuit();
        this.updateCircuitStats();
        document.getElementById('executionResults').style.display = 'none';
    }
    
    async executeCircuit() {
        if (this.circuit.length === 0) {
            alert('Please add some gates to the circuit first!');
            return;
        }
        
        if (this.isExecuting) return;
        
        this.isExecuting = true;
        document.getElementById('executeCircuit').textContent = 'Executing...';
        
        try {
            // Convert circuit to API format
            const circuitData = this.convertCircuitToAPIFormat();
            
            // Execute circuit via API
            const response = await fetch('/api/quantum/execute-circuit', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    circuit: circuitData,
                    shots: 1024,
                    backend: 'qasm_simulator'
                })
            });
            
            if (!response.ok) {
                throw new Error('Circuit execution failed');
            }
            
            const results = await response.json();
            this.displayResults(results);
            
        } catch (error) {
            console.error('Execution error:', error);
            alert('Circuit execution failed: ' + error.message);
        } finally {
            this.isExecuting = false;
            document.getElementById('executeCircuit').textContent = 'Execute Circuit';
        }
    }
    
    convertCircuitToAPIFormat() {
        return {
            qubits: this.currentQubits,
            gates: this.circuit.map(gate => {
                const gateData = {
                    type: gate.type,
                    qubit: gate.qubit,
                    column: gate.column
                };
                
                if (this.availableGates[gate.type].two_qubit) {
                    gateData.control_qubit = gate.controlQubit;
                    gateData.target_qubit = gate.targetQubit;
                }
                
                if (gate.parameters) {
                    gateData.parameters = gate.parameters;
                }
                
                return gateData;
            })
        };
    }
    
    displayResults(results) {
        const resultsDiv = document.getElementById('executionResults');
        const statsDiv = document.getElementById('executionStats');
        
        // Show results section
        resultsDiv.style.display = 'block';
        
        // Display statistics
        statsDiv.innerHTML = `
            <h5>Execution Statistics</h5>
            <p><strong>Shots:</strong> ${results.shots}</p>
            <p><strong>Execution Time:</strong> ${results.execution_time?.toFixed(3)}s</p>
            <p><strong>Success Rate:</strong> ${((results.success_count / results.shots) * 100).toFixed(1)}%</p>
        `;
        
        // Draw probability histogram
        this.drawResultsHistogram(results.counts);
        
        this.executionResults = results;
    }
    
    drawResultsHistogram(counts) {
        const canvas = document.getElementById('resultsCanvas');
        const ctx = canvas.getContext('2d');
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Calculate probabilities
        const total = Object.values(counts).reduce((sum, count) => sum + count, 0);
        const states = Object.keys(counts).sort();
        const maxCount = Math.max(...Object.values(counts));
        
        // Draw bars
        const barWidth = Math.min(60, (canvas.width - 100) / states.length);
        const maxBarHeight = canvas.height - 80;
        
        states.forEach((state, i) => {
            const count = counts[state];
            const probability = count / total;
            const barHeight = (count / maxCount) * maxBarHeight;
            const x = 50 + i * (barWidth + 10);
            const y = canvas.height - 50 - barHeight;
            
            // Draw bar
            ctx.fillStyle = `hsl(${(i * 30) % 360}, 70%, 60%)`;
            ctx.fillRect(x, y, barWidth, barHeight);
            
            // Draw probability text
            ctx.fillStyle = '#2C3E50';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(`${(probability * 100).toFixed(1)}%`, x + barWidth/2, y - 5);
            
            // Draw state label
            ctx.fillText(`|${state}⟩`, x + barWidth/2, canvas.height - 30);
            
            // Draw count
            ctx.font = '10px Arial';
            ctx.fillText(`${count}`, x + barWidth/2, canvas.height - 15);
        });
        
        // Draw axes
        ctx.strokeStyle = '#2C3E50';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(40, 20);
        ctx.lineTo(40, canvas.height - 40);
        ctx.lineTo(canvas.width - 20, canvas.height - 40);
        ctx.stroke();
        
        // Y-axis label
        ctx.fillStyle = '#2C3E50';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.save();
        ctx.translate(15, canvas.height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Probability', 0, 0);
        ctx.restore();
        
        // X-axis label
        ctx.textAlign = 'center';
        ctx.fillText('Quantum States', canvas.width / 2, canvas.height - 5);
    }
    
    async optimizeCircuit() {
        if (this.circuit.length === 0) {
            alert('Please add some gates to the circuit first!');
            return;
        }
        
        try {
            const circuitData = this.convertCircuitToAPIFormat();
            
            const response = await fetch('/api/quantum/optimize-circuit', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    circuit: circuitData,
                    optimization_level: 3
                })
            });
            
            if (!response.ok) {
                throw new Error('Circuit optimization failed');
            }
            
            const optimizedData = await response.json();
            
            // Update circuit with optimized version
            this.loadCircuitFromAPIFormat(optimizedData.optimized_circuit);
            
            alert(`Circuit optimized!\\nGates: ${optimizedData.original_gates} → ${optimizedData.optimized_gates}\\nDepth: ${optimizedData.original_depth} → ${optimizedData.optimized_depth}`);
            
        } catch (error) {
            console.error('Optimization error:', error);
            alert('Circuit optimization failed: ' + error.message);
        }
    }
    
    loadCircuitFromAPIFormat(apiCircuit) {
        this.circuit = apiCircuit.gates.map((gate, index) => {
            const gateObj = {
                type: gate.type,
                x: gate.column * this.gridSize + 100,
                y: gate.qubit * this.gridSize + 50,
                qubit: gate.qubit,
                column: gate.column,
                id: Date.now() + index,
                parameters: gate.parameters || null
            };
            
            if (gate.control_qubit !== undefined) {
                gateObj.controlQubit = gate.control_qubit;
                gateObj.targetQubit = gate.target_qubit;
                gateObj.y = Math.min(gateObj.controlQubit, gateObj.targetQubit) * this.gridSize + 50;
            }
            
            return gateObj;
        });
        
        this.drawCircuit();
        this.updateCircuitStats();
    }
    
    exportCircuit() {
        if (this.circuit.length === 0) {
            alert('Please add some gates to the circuit first!');
            return;
        }
        
        const circuitData = {
            metadata: {
                name: 'Custom Circuit',
                created: new Date().toISOString(),
                qubits: this.currentQubits,
                gates: this.circuit.length,
                depth: this.calculateCircuitDepth()
            },
            circuit: this.convertCircuitToAPIFormat()
        };
        
        // Create download link
        const dataStr = JSON.stringify(circuitData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = 'quantum_circuit.json';
        link.click();
        
        // Also copy to clipboard
        navigator.clipboard.writeText(dataStr).then(() => {
            alert('Circuit exported to file and copied to clipboard!');
        });
    }
}

// Initialize circuit designer when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    if (document.getElementById('circuitDesigner')) {
        window.quantumCircuitDesigner = new QuantumCircuitDesigner('circuitDesigner');
    }
});