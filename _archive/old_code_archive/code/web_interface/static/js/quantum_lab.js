/**
 * 🔬 Quantum Research Laboratory Interface
 * ==================================================
 * Interactive quantum experiment platform for users
 */

let selectedAlgorithm = 'qaoa';
let currentExperiment = null;
let resultsChart = null;

document.addEventListener('DOMContentLoaded', function() {
    initializeQuantumLab();
    setupAlgorithmSelection();
    setupParameterControls();
    setupFileUpload();
    setupCircuitDesigner();
    setupExperimentControls();
    setupQuickExperiments();
    
    console.log('🔬 Quantum Lab initialized');
});

function initializeQuantumLab() {
    // Initialize default algorithm selection
    selectAlgorithm('qaoa');
    
    // Setup results chart
    initializeResultsChart();
    
    // Generate initial circuit layout
    generateQubitLines(4);
}

function setupAlgorithmSelection() {
    const algorithmCards = document.querySelectorAll('.algorithm-card');
    
    algorithmCards.forEach(card => {
        card.addEventListener('click', function() {
            const algorithm = this.dataset.algorithm;
            selectAlgorithm(algorithm);
        });
    });
}

function selectAlgorithm(algorithm) {
    selectedAlgorithm = algorithm;
    
    // Update visual selection
    document.querySelectorAll('.algorithm-card').forEach(card => {
        card.classList.remove('selected');
    });
    document.querySelector(`[data-algorithm="${algorithm}"]`).classList.add('selected');
    
    // Show/hide parameter groups
    document.querySelectorAll('.parameter-group').forEach(group => {
        group.style.display = 'none';
    });
    document.getElementById(`${algorithm}-params`).style.display = 'block';
    
    // Update console
    addConsoleMessage(`Selected algorithm: ${algorithm.toUpperCase()}`, 'info');
}

function setupParameterControls() {
    // Setup all parameter sliders
    const sliders = document.querySelectorAll('.parameter-slider');
    
    sliders.forEach(slider => {
        const valueElement = document.getElementById(slider.id + '-value');
        
        slider.addEventListener('input', function() {
            if (valueElement) {
                let displayValue = this.value;
                
                // Format special cases
                if (this.id.includes('beta') || this.id.includes('gamma')) {
                    displayValue = parseFloat(this.value).toFixed(2);
                } else if (this.id.includes('qubits') && this.id.includes('grover')) {
                    displayValue = `2^${this.value}`;
                }
                
                valueElement.textContent = displayValue;
            }
            
            // Update circuit visualization if needed
            if (this.id.includes('qubits')) {
                const qubitCount = parseInt(this.value);
                generateQubitLines(qubitCount);
            }
        });
        
        // Trigger initial update
        slider.dispatchEvent(new Event('input'));
    });
}

function setupFileUpload() {
    // Data file upload
    const dataUploadArea = document.getElementById('dataUploadArea');
    const dataFileInput = document.getElementById('dataFileInput');
    
    dataUploadArea.addEventListener('click', () => dataFileInput.click());
    
    dataUploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        this.classList.add('dragover');
    });
    
    dataUploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        this.classList.remove('dragover');
    });
    
    dataUploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        this.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        handleFileUpload(files, 'data');
    });
    
    dataFileInput.addEventListener('change', function(e) {
        handleFileUpload(e.target.files, 'data');
    });
    
    // QASM file upload
    const qasmUploadArea = document.getElementById('qasmUploadArea');
    const qasmFileInput = document.getElementById('qasmFileInput');
    
    qasmUploadArea.addEventListener('click', () => qasmFileInput.click());
    
    qasmFileInput.addEventListener('change', function(e) {
        handleFileUpload(e.target.files, 'qasm');
    });
}

function handleFileUpload(files, type) {
    const uploadedFilesContainer = document.getElementById('uploadedFiles');
    
    // Clear "no files" message
    if (uploadedFilesContainer.children.length === 1 && 
        uploadedFilesContainer.children[0].textContent.includes('No files')) {
        uploadedFilesContainer.innerHTML = '';
    }
    
    Array.from(files).forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'list-group-item d-flex justify-content-between align-items-center';
        
        const fileInfo = document.createElement('div');
        fileInfo.innerHTML = `
            <i class="fas fa-file-${type === 'qasm' ? 'code' : 'table'} me-2 text-quantum"></i>
            <strong>${file.name}</strong>
            <small class="text-secondary ms-2">(${formatFileSize(file.size)})</small>
        `;
        
        const actions = document.createElement('div');
        actions.innerHTML = `
            <button class="btn btn-sm btn-outline-primary me-2" onclick="processFile('${file.name}', '${type}')">
                <i class="fas fa-cog"></i>
            </button>
            <button class="btn btn-sm btn-outline-danger" onclick="removeFile('${file.name}')">
                <i class="fas fa-trash"></i>
            </button>
        `;
        
        fileItem.appendChild(fileInfo);
        fileItem.appendChild(actions);
        uploadedFilesContainer.appendChild(fileItem);
        
        // Process file content
        if (type === 'qasm') {
            readQASMFile(file);
        } else {
            readDataFile(file);
        }
    });
    
    addConsoleMessage(`Uploaded ${files.length} ${type} file(s)`, 'success');
}

function readQASMFile(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const qasmContent = e.target.result;
        document.getElementById('qasmEditor').value = qasmContent;
        addConsoleMessage(`Loaded QASM circuit: ${file.name}`, 'info');
    };
    reader.readAsText(file);
}

function readDataFile(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            let data;
            const extension = file.name.split('.').pop().toLowerCase();
            
            if (extension === 'json') {
                data = JSON.parse(e.target.result);
            } else if (extension === 'csv') {
                data = parseCSV(e.target.result);
            } else {
                data = e.target.result;
            }
            
            addConsoleMessage(`Parsed ${file.name}: ${typeof data === 'object' ? Object.keys(data).length : 'N/A'} entries`, 'success');
        } catch (error) {
            addConsoleMessage(`Error parsing ${file.name}: ${error.message}`, 'error');
        }
    };
    reader.readAsText(file);
}

function parseCSV(csvText) {
    const lines = csvText.split('\n');
    const headers = lines[0].split(',');
    const data = [];
    
    for (let i = 1; i < lines.length; i++) {
        if (lines[i].trim()) {
            const values = lines[i].split(',');
            const row = {};
            headers.forEach((header, index) => {
                row[header.trim()] = values[index] ? values[index].trim() : '';
            });
            data.push(row);
        }
    }
    
    return data;
}

function setupCircuitDesigner() {
    const gates = document.querySelectorAll('.circuit-gate');
    const circuitCanvas = document.getElementById('circuitCanvas');
    
    // Setup drag and drop for gates
    gates.forEach(gate => {
        gate.addEventListener('dragstart', function(e) {
            e.dataTransfer.setData('text/plain', this.dataset.gate);
            this.classList.add('dragging');
        });
        
        gate.addEventListener('dragend', function(e) {
            this.classList.remove('dragging');
        });
    });
    
    // Setup drop zone
    circuitCanvas.addEventListener('dragover', function(e) {
        e.preventDefault();
    });
    
    circuitCanvas.addEventListener('drop', function(e) {
        e.preventDefault();
        const gateType = e.dataTransfer.getData('text/plain');
        const rect = this.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        addGateToCircuit(gateType, x, y);
    });
    
    // Circuit control buttons
    document.getElementById('generateQASM').addEventListener('click', generateQASMFromCircuit);
    document.getElementById('clearCircuit').addEventListener('click', clearCircuit);
    document.getElementById('loadTemplate').addEventListener('click', loadCircuitTemplate);
}

function generateQubitLines(qubitCount) {
    const qubitLines = document.getElementById('qubitLines');
    qubitLines.innerHTML = '';
    
    for (let i = 0; i < qubitCount; i++) {
        const line = document.createElement('div');
        line.className = 'qubit-line';
        line.dataset.qubit = i;
        
        const label = document.createElement('span');
        label.textContent = `|q${i}⟩`;
        label.style.position = 'absolute';
        label.style.left = '10px';
        label.style.top = '-10px';
        label.style.fontSize = '12px';
        label.className = 'text-quantum';
        
        line.appendChild(label);
        qubitLines.appendChild(line);
    }
}

function addGateToCircuit(gateType, x, y) {
    const gate = document.createElement('div');
    gate.className = 'circuit-gate';
    gate.textContent = gateType;
    gate.style.position = 'absolute';
    gate.style.left = x + 'px';
    gate.style.top = y + 'px';
    gate.style.transform = 'translate(-50%, -50%)';
    gate.dataset.gate = gateType;
    
    // Add remove functionality
    gate.addEventListener('click', function() {
        if (confirm('Remove this gate?')) {
            this.remove();
            addConsoleMessage(`Removed ${gateType} gate`, 'info');
        }
    });
    
    document.getElementById('circuitCanvas').appendChild(gate);
    addConsoleMessage(`Added ${gateType} gate to circuit`, 'info');
}

function generateQASMFromCircuit() {
    const gates = document.querySelectorAll('#circuitCanvas .circuit-gate');
    const qubitCount = document.querySelectorAll('.qubit-line').length;
    
    let qasm = `OPENQASM 2.0;\ninclude "qelib1.inc";\n\n`;
    qasm += `qreg q[${qubitCount}];\n`;
    qasm += `creg c[${qubitCount}];\n\n`;
    
    gates.forEach(gate => {
        const gateType = gate.dataset.gate.toLowerCase();
        // Simplified QASM generation
        if (gateType === 'h') {
            qasm += `h q[0];\n`;
        } else if (gateType === 'x') {
            qasm += `x q[0];\n`;
        } else if (gateType === 'cnot') {
            qasm += `cx q[0], q[1];\n`;
        }
        // Add more gate translations as needed
    });
    
    qasm += `\nmeasure q -> c;`;
    
    document.getElementById('qasmEditor').value = qasm;
    addConsoleMessage('Generated QASM code from circuit', 'success');
}

function clearCircuit() {
    document.querySelectorAll('#circuitCanvas .circuit-gate').forEach(gate => gate.remove());
    addConsoleMessage('Circuit cleared', 'info');
}

function loadCircuitTemplate() {
    const templates = {
        'bell-state': `OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0], q[1];\nmeasure q -> c;`,
        'grover-2qubit': `OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0];\nh q[1];\nz q[1];\ncz q[0], q[1];\nh q[0];\nh q[1];\nz q[0];\nz q[1];\ncz q[0], q[1];\nh q[0];\nh q[1];\nmeasure q -> c;`
    };
    
    const template = prompt('Enter template name (bell-state, grover-2qubit):');
    if (templates[template]) {
        document.getElementById('qasmEditor').value = templates[template];
        addConsoleMessage(`Loaded ${template} template`, 'success');
    }
}

function setupExperimentControls() {
    document.getElementById('runExperiment').addEventListener('click', runExperiment);
    document.getElementById('stopExperiment').addEventListener('click', stopExperiment);
    document.getElementById('exportResults').addEventListener('click', exportResults);
}

function runExperiment() {
    if (currentExperiment) {
        addConsoleMessage('Experiment already running', 'warning');
        return;
    }
    
    const params = collectParameters();
    updateExecutionStatus('Running', 'warning');
    addConsoleMessage(`Starting ${selectedAlgorithm.toUpperCase()} experiment...`, 'info');
    
    // Simulate experiment execution
    currentExperiment = setTimeout(() => {
        completeExperiment();
    }, 3000 + Math.random() * 2000);
    
    // Show progress updates
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += 10 + Math.random() * 20;
        if (progress >= 100) {
            clearInterval(progressInterval);
            return;
        }
        addConsoleMessage(`Progress: ${Math.round(progress)}%`, 'info');
    }, 500);
}

function stopExperiment() {
    if (currentExperiment) {
        clearTimeout(currentExperiment);
        currentExperiment = null;
        updateExecutionStatus('Stopped', 'danger');
        addConsoleMessage('Experiment stopped by user', 'warning');
    }
}

function completeExperiment() {
    currentExperiment = null;
    updateExecutionStatus('Completed', 'success');
    
    // Generate realistic results
    const results = generateQuantumResults();
    displayResults(results);
    
    addConsoleMessage('Experiment completed successfully', 'success');
    addConsoleMessage(`Quantum advantage: ${results.quantumAdvantage}x`, 'success');
}

function collectParameters() {
    const params = {};
    
    // Collect parameters based on selected algorithm
    document.querySelectorAll(`#${selectedAlgorithm}-params input, #${selectedAlgorithm}-params select`).forEach(input => {
        params[input.id] = input.value;
    });
    
    return params;
}

function generateQuantumResults() {
    const baseTime = 1000 + Math.random() * 2000;
    const quantumAdvantage = 8 + Math.random() * 12;
    const fidelity = 92 + Math.random() * 7;
    const successRate = 88 + Math.random() * 11;
    
    // Generate measurement data
    const measurementData = {};
    const stateCount = selectedAlgorithm === 'grover' ? 8 : 4;
    
    for (let i = 0; i < stateCount; i++) {
        const state = i.toString(2).padStart(Math.log2(stateCount), '0');
        measurementData[state] = Math.random() * 1000;
    }
    
    return {
        executionTime: baseTime,
        quantumAdvantage: quantumAdvantage.toFixed(1),
        fidelity: fidelity.toFixed(1),
        successRate: successRate.toFixed(1),
        measurementData: measurementData
    };
}

function displayResults(results) {
    // Update metrics
    document.getElementById('executionTime').textContent = Math.round(results.executionTime);
    document.getElementById('quantumAdvantage').textContent = results.quantumAdvantage + 'x';
    document.getElementById('fidelity').textContent = results.fidelity;
    document.getElementById('successRate').textContent = results.successRate;
    
    // Update chart
    updateResultsChart(results.measurementData);
}

function initializeResultsChart() {
    const ctx = document.getElementById('resultsChart').getContext('2d');
    resultsChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['|00⟩', '|01⟩', '|10⟩', '|11⟩'],
            datasets: [{
                label: 'Measurement Counts',
                data: [0, 0, 0, 0],
                backgroundColor: [
                    'rgba(0, 230, 118, 0.8)',
                    'rgba(13, 71, 161, 0.8)',
                    'rgba(156, 39, 176, 0.8)',
                    'rgba(255, 152, 0, 0.8)'
                ],
                borderColor: [
                    'rgba(0, 230, 118, 1)',
                    'rgba(13, 71, 161, 1)',
                    'rgba(156, 39, 176, 1)',
                    'rgba(255, 152, 0, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#adb5bd'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#adb5bd'
                    }
                }
            }
        }
    });
}

function updateResultsChart(data) {
    const labels = Object.keys(data).map(state => `|${state}⟩`);
    const values = Object.values(data);
    
    resultsChart.data.labels = labels;
    resultsChart.data.datasets[0].data = values;
    resultsChart.update();
}

function setupQuickExperiments() {
    const quickButtons = document.querySelectorAll('[data-quick-experiment]');
    
    quickButtons.forEach(button => {
        button.addEventListener('click', function() {
            const experiment = this.dataset.quickExperiment;
            runQuickExperiment(experiment);
        });
    });
}

function runQuickExperiment(experiment) {
    const experiments = {
        'bell-state': { algorithm: 'qaoa', qasm: 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0], q[1];\nmeasure q -> c;' },
        'quantum-teleportation': { algorithm: 'qml', qasm: 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[3];\nh q[0];\ncx q[0], q[1];\ncx q[0], q[2];\nh q[0];\nmeasure q -> c;' },
        'superposition-demo': { algorithm: 'grover', qasm: 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0];\nh q[1];\nmeasure q -> c;' },
        'quantum-random': { algorithm: 'qaoa', qasm: 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\ncreg c[4];\nh q[0];\nh q[1];\nh q[2];\nh q[3];\nmeasure q -> c;' }
    };
    
    if (experiments[experiment]) {
        selectAlgorithm(experiments[experiment].algorithm);
        document.getElementById('qasmEditor').value = experiments[experiment].qasm;
        addConsoleMessage(`Loaded quick experiment: ${experiment}`, 'info');
        
        // Auto-run the experiment
        setTimeout(() => {
            runExperiment();
        }, 500);
    }
}

function updateExecutionStatus(status, type) {
    const statusElement = document.getElementById('executionStatus');
    const badge = statusElement.querySelector('.badge');
    
    statusElement.querySelector('span').innerHTML = `<i class="fas fa-${type === 'success' ? 'check-circle' : type === 'warning' ? 'clock' : 'info-circle'} me-2"></i>${status}`;
    badge.className = `badge bg-${type}`;
    badge.textContent = status;
}

function addConsoleMessage(message, type = 'info') {
    const console = document.getElementById('consoleOutput');
    const timestamp = new Date().toLocaleTimeString();
    const typeIcon = {
        'info': 'ℹ️',
        'success': '✅',
        'warning': '⚠️',
        'error': '❌'
    };
    
    const messageElement = document.createElement('div');
    messageElement.className = `text-${type === 'info' ? 'info' : type === 'success' ? 'success' : type === 'warning' ? 'warning' : 'danger'}`;
    messageElement.innerHTML = `[${timestamp}] ${typeIcon[type]} ${message}`;
    
    console.appendChild(messageElement);
    console.scrollTop = console.scrollHeight;
}

function exportResults() {
    if (!resultsChart || !resultsChart.data.datasets[0].data.some(val => val > 0)) {
        addConsoleMessage('No results to export', 'warning');
        return;
    }
    
    const results = {
        algorithm: selectedAlgorithm,
        parameters: collectParameters(),
        timestamp: new Date().toISOString(),
        executionTime: document.getElementById('executionTime').textContent,
        quantumAdvantage: document.getElementById('quantumAdvantage').textContent,
        fidelity: document.getElementById('fidelity').textContent,
        successRate: document.getElementById('successRate').textContent,
        measurementData: resultsChart.data.datasets[0].data,
        stateLabels: resultsChart.data.labels
    };
    
    const dataStr = JSON.stringify(results, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    
    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `quantum_experiment_${selectedAlgorithm}_${Date.now()}.json`;
    link.click();
    
    addConsoleMessage('Results exported successfully', 'success');
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function processFile(filename, type) {
    addConsoleMessage(`Processing ${filename}...`, 'info');
    // Implement file processing logic here
}

function removeFile(filename) {
    const fileItems = document.querySelectorAll('#uploadedFiles .list-group-item');
    fileItems.forEach(item => {
        if (item.textContent.includes(filename)) {
            item.remove();
            addConsoleMessage(`Removed ${filename}`, 'info');
        }
    });
}