/**
 * Quantum.js
 * Handles quantum-related functionality and UI interactions
 */

// Initialize the quantum module when the document is ready
document.addEventListener('DOMContentLoaded', function() {
    // Initialize quantum toggle
    const quantumToggle = document.getElementById('quantumToggle');
    if (quantumToggle) {
        // Check if quantum was previously enabled
        const isQuantumEnabled = localStorage.getItem('quantumEnabled') === 'true';
        quantumToggle.checked = isQuantumEnabled;
        
        // Apply initial state
        updateQuantumUIState(isQuantumEnabled);
        
        // Add event listener for toggle changes
        quantumToggle.addEventListener('change', function() {
            const isEnabled = this.checked;
            localStorage.setItem('quantumEnabled', isEnabled);
            updateQuantumUIState(isEnabled);
            
            // Show notification
            showQuantumNotification(isEnabled);
            
            // If enabled, initialize quantum components
            if (isEnabled) {
                initializeQuantumComponents();
            }
        });
    }
    
    // Initialize quantum enhancement components if they exist on the page
    initializeQuantumComponents();
});

/**
 * Updates UI to reflect the quantum state
 * @param {boolean} isEnabled - Whether quantum is enabled
 */
function updateQuantumUIState(isEnabled) {
    // Update status indicator
    const quantumStatus = document.getElementById('quantumStatus');
    if (quantumStatus) {
        quantumStatus.textContent = isEnabled ? 'Active' : 'Idle';
        quantumStatus.className = isEnabled ? 'badge bg-success' : 'badge bg-warning';
    }
    
    // Update details text
    const quantumDetails = document.getElementById('quantumDetails');
    if (quantumDetails) {
        quantumDetails.textContent = isEnabled ? 
            'Quantum enhancement active. Computing resources allocated.' : 
            'Quantum components ready for activation.';
    }
    
    // Toggle visibility of quantum-only elements
    const quantumElements = document.querySelectorAll('.quantum-component');
    quantumElements.forEach(element => {
        element.style.display = isEnabled ? 'block' : 'none';
    });
    
    // Update simulation buttons
    updateSimulationButtons(isEnabled);
}

/**
 * Updates the simulation buttons based on quantum state
 * @param {boolean} isQuantumEnabled - Whether quantum is enabled
 */
function updateSimulationButtons(isQuantumEnabled) {
    const classicalBtn = document.querySelector('.classical-simulation-btn');
    const quantumBtn = document.querySelector('.quantum-simulation-btn');
    
    if (classicalBtn) {
        classicalBtn.disabled = isQuantumEnabled;
    }
    
    if (quantumBtn) {
        quantumBtn.disabled = !isQuantumEnabled;
    }
}

/**
 * Shows a notification about quantum mode change
 * @param {boolean} isEnabled - Whether quantum is enabled
 */
function showQuantumNotification(isEnabled) {
    const alertContainer = document.getElementById('alertContainer');
    if (!alertContainer) return;
    
    const alert = document.createElement('div');
    alert.className = `alert alert-${isEnabled ? 'info' : 'warning'} alert-dismissible fade show medieval-alert`;
    alert.innerHTML = `
        <strong>${isEnabled ? 'Quantum Mode Activated' : 'Quantum Mode Deactivated'}</strong>: 
        ${isEnabled ? 
            'Enhanced quantum simulations are now available. Performance may be affected by quantum backend availability.' : 
            'Reverting to classical computation methods. All features remain available with standard precision.'}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    alertContainer.appendChild(alert);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alert.classList.remove('show');
        setTimeout(() => alert.remove(), 300);
    }, 5000);
}

/**
 * Initializes quantum components on the page
 */
function initializeQuantumComponents() {
    const isQuantumEnabled = localStorage.getItem('quantumEnabled') === 'true';
    if (!isQuantumEnabled) return;
    
    // Check for Monte Carlo simulation component
    initializeQuantumMonteCarlo();
    
    // Check for Quantum Circuit Visualization
    initializeQuantumCircuitVis();
    
    // Check for Quantum Machine Learning component
    initializeQuantumML();
}

/**
 * Initializes Quantum Monte Carlo simulation if element exists
 */
function initializeQuantumMonteCarlo() {
    const qmcContainer = document.getElementById('quantumMonteCarloContainer');
    if (!qmcContainer) return;
    
    // Show loading state
    qmcContainer.innerHTML = '<div class="text-center my-4"><i class="fas fa-atom fa-spin fa-2x"></i><p class="mt-2">Initializing Quantum Monte Carlo Simulation...</p></div>';
    
    // Simulate initialization
    simulateQuantumProgress('quantumProgress', 100, 2000, () => {
        // Mock data for demonstration
        const qmcData = generateMockQMCData();
        renderQMCVisualization(qmcContainer, qmcData);
    });
}

/**
 * Initializes Quantum Circuit Visualization if element exists
 */
function initializeQuantumCircuitVis() {
    const circuitContainer = document.getElementById('quantumCircuitContainer');
    if (!circuitContainer) return;
    
    // Simulate loading
    simulateQuantumProgress('quantumProgress', 60, 1500, () => {
        // For demo, just show a placeholder
        circuitContainer.innerHTML = `
            <div class="quantum-component p-3">
                <h5 class="mb-3"><i class="fas fa-project-diagram me-2"></i>Quantum Circuit</h5>
                <div class="text-center">
                    <img src="https://qiskit.org/documentation/stable/0.19/_images/qiskit-circuits.png" 
                         alt="Quantum Circuit Visualization" 
                         class="img-fluid border rounded"
                         style="max-height: 300px;">
                    <p class="text-muted mt-2">Example quantum circuit for environmental simulation</p>
                </div>
                <div class="mt-3">
                    <button class="btn btn-sm btn-outline-info medieval-btn">Change Circuit</button>
                    <button class="btn btn-sm btn-outline-success medieval-btn">Run Simulation</button>
                </div>
            </div>
        `;
    });
}

/**
 * Initializes Quantum Machine Learning component if element exists
 */
function initializeQuantumML() {
    const qmlContainer = document.getElementById('quantumMLContainer');
    if (!qmlContainer) return;
    
    // Simulate loading
    simulateQuantumProgress('quantumProgress', 80, 2500, () => {
        qmlContainer.innerHTML = `
            <div class="quantum-component p-3">
                <h5 class="mb-3"><i class="fas fa-brain me-2"></i>Quantum ML Predictions</h5>
                <div class="chart-container">
                    <canvas id="qmlPredictionChart"></canvas>
                </div>
                <div class="text-muted mt-2">
                    <small>Quantum-enhanced predictions show 27% improvement in accuracy over classical methods</small>
                </div>
            </div>
        `;
        
        // Initialize chart if Chart.js is available
        if (window.Chart) {
            renderQMLChart();
        }
    });
}

/**
 * Simulates a progress animation for quantum operations
 * @param {string} elementId - ID of the progress bar
 * @param {number} targetValue - Target percentage value
 * @param {number} duration - Duration of the animation in ms
 * @param {Function} callback - Callback function when complete
 */
function simulateQuantumProgress(elementId, targetValue, duration, callback) {
    const progressBar = document.getElementById(elementId);
    if (!progressBar) {
        if (callback) callback();
        return;
    }
    
    let currentValue = parseInt(progressBar.getAttribute('aria-valuenow') || '0');
    const startTime = Date.now();
    
    function updateProgress() {
        const elapsedTime = Date.now() - startTime;
        const progress = Math.min(elapsedTime / duration, 1);
        currentValue = Math.floor(progress * targetValue);
        
        progressBar.style.width = `${currentValue}%`;
        progressBar.setAttribute('aria-valuenow', currentValue);
        progressBar.textContent = `${currentValue}%`;
        
        if (progress < 1) {
            requestAnimationFrame(updateProgress);
        } else {
            if (callback) callback();
        }
    }
    
    updateProgress();
}

/**
 * Generates mock data for Quantum Monte Carlo visualization
 * @returns {Object} Mock data for visualization
 */
function generateMockQMCData() {
    // Generate classical vs quantum data points
    const numPoints = 50;
    const classicalData = [];
    const quantumData = [];
    const labels = [];
    
    for (let i = 0; i < numPoints; i++) {
        const x = i / (numPoints - 1);
        labels.push(i);
        
        // Classical has more noise and variance
        const classicalNoise = Math.random() * 0.3 - 0.15;
        const classicalValue = Math.sin(x * Math.PI * 2) * 0.5 + 0.5 + classicalNoise;
        classicalData.push(classicalValue);
        
        // Quantum has less noise and better convergence
        const quantumNoise = Math.random() * 0.1 - 0.05;
        const quantumValue = Math.sin(x * Math.PI * 2) * 0.5 + 0.5 + quantumNoise;
        quantumData.push(quantumValue);
    }
    
    return {
        labels,
        classicalData,
        quantumData
    };
}

/**
 * Renders Quantum Monte Carlo visualization
 * @param {HTMLElement} container - Container element
 * @param {Object} data - Data for visualization
 */
function renderQMCVisualization(container, data) {
    container.innerHTML = `
        <div class="quantum-component p-3">
            <h5 class="mb-3"><i class="fas fa-random me-2"></i>Quantum Monte Carlo Simulation</h5>
            <div class="chart-container">
                <canvas id="qmcComparisonChart"></canvas>
            </div>
            <div class="text-muted mt-2">
                <small>Comparison of classical and quantum Monte Carlo methods for environment simulation</small>
            </div>
            <div class="mt-3">
                <button class="btn btn-sm btn-outline-primary medieval-btn" id="regenerateQMC">
                    Regenerate Simulation
                </button>
                <button class="btn btn-sm btn-outline-info medieval-btn" id="toggleQMCDetail">
                    Show Details
                </button>
            </div>
        </div>
    `;
    
    // Initialize chart if Chart.js is available
    if (window.Chart) {
        const ctx = document.getElementById('qmcComparisonChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: [
                    {
                        label: 'Classical Monte Carlo',
                        data: data.classicalData,
                        borderColor: '#f8f9fa',
                        backgroundColor: 'rgba(248, 249, 250, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.4
                    },
                    {
                        label: 'Quantum Monte Carlo',
                        data: data.quantumData,
                        borderColor: '#6f42c1',
                        backgroundColor: 'rgba(111, 66, 193, 0.2)',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#d4af37'
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Iterations',
                            color: '#adb5bd'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#adb5bd'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Convergence',
                            color: '#adb5bd'
                        },
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
    
    // Add event handlers
    const regenerateBtn = document.getElementById('regenerateQMC');
    if (regenerateBtn) {
        regenerateBtn.addEventListener('click', function() {
            initializeQuantumMonteCarlo();
        });
    }
}

/**
 * Renders Quantum ML Chart
 */
function renderQMLChart() {
    const ctx = document.getElementById('qmlPredictionChart').getContext('2d');
    
    // Generate labels and data
    const labels = ['Endurance', 'Speed', 'Strength', 'Fatigue', 'Recovery'];
    const classicalPredictions = [68, 73, 59, 81, 62];
    const quantumPredictions = [72, 76, 65, 84, 69];
    const actualValues = [75, 78, 67, 82, 71];
    
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Actual Values',
                    data: actualValues,
                    borderColor: '#20c997',
                    backgroundColor: 'rgba(32, 201, 151, 0.1)',
                    borderWidth: 2,
                    pointBackgroundColor: '#20c997'
                },
                {
                    label: 'Quantum Predictions',
                    data: quantumPredictions,
                    borderColor: '#6f42c1',
                    backgroundColor: 'rgba(111, 66, 193, 0.1)',
                    borderWidth: 2,
                    pointBackgroundColor: '#6f42c1'
                },
                {
                    label: 'Classical Predictions',
                    data: classicalPredictions,
                    borderColor: '#f8f9fa',
                    backgroundColor: 'rgba(248, 249, 250, 0.1)',
                    borderWidth: 2,
                    pointBackgroundColor: '#f8f9fa'
                }
            ]
        },
        options: {
            elements: {
                line: {
                    tension: 0.2
                }
            },
            scales: {
                r: {
                    angleLines: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    pointLabels: {
                        color: '#d4af37'
                    },
                    ticks: {
                        color: '#adb5bd',
                        backdropColor: 'transparent'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#d4af37'
                    }
                }
            }
        }
    });
} 