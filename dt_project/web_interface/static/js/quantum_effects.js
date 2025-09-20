/**
 * 🌌 Modern Quantum Visual Effects
 * ==================================================
 * Enhanced visual effects for the quantum digital twin platform
 */

document.addEventListener('DOMContentLoaded', function() {
    
    // Initialize quantum animations
    initializeQuantumEffects();
    
    // Setup interactive elements
    setupInteractiveElements();
    
    // Start quantum status updates
    startQuantumStatusUpdates();
    
    console.log('🌌 Quantum effects initialized');
});

function initializeQuantumEffects() {
    // Add quantum glow effects to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        // Staggered animation
        setTimeout(() => {
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 100);
        
        // Add hover effects
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px) scale(1.02)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });
    
    // Animate quantum indicators
    const indicators = document.querySelectorAll('.quantum-indicator');
    indicators.forEach(indicator => {
        if (indicator.classList.contains('active')) {
            // Add pulsing animation
            indicator.style.animation = 'pulse 2s infinite';
        }
    });
    
    // Add scanning effect to quantum circuits
    const circuits = document.querySelectorAll('.quantum-circuit');
    circuits.forEach(circuit => {
        circuit.addEventListener('mouseenter', function() {
            this.classList.add('scanning');
        });
    });
}

function setupInteractiveElements() {
    // Enhanced quantum toggle
    const quantumToggle = document.getElementById('quantumToggle');
    if (quantumToggle) {
        quantumToggle.addEventListener('change', function() {
            const isEnabled = this.checked;
            document.body.classList.toggle('quantum-enhanced', isEnabled);
            
            // Update all quantum indicators
            const indicators = document.querySelectorAll('.quantum-indicator');
            indicators.forEach(indicator => {
                if (isEnabled) {
                    indicator.classList.add('active');
                    indicator.classList.remove('inactive');
                } else {
                    indicator.classList.add('inactive');
                    indicator.classList.remove('active');
                }
            });
            
            // Show/hide quantum sections
            const quantumSections = document.querySelectorAll('[id*="quantum"]');
            quantumSections.forEach(section => {
                if (section.style.display === 'none' || section.style.display === '') {
                    section.style.display = isEnabled ? 'block' : 'none';
                }
            });
            
            // Update progress bars
            updateQuantumProgress(isEnabled ? 87 : 0);
            
            console.log(`🔬 Quantum mode ${isEnabled ? 'enabled' : 'disabled'}`);
        });
    }
    
    // Interactive buttons with quantum effects
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            // Create ripple effect
            createRippleEffect(this, e);
        });
    });
    
    // Metric card interactions
    const metricCards = document.querySelectorAll('.metric-card');
    metricCards.forEach(card => {
        card.addEventListener('click', function() {
            // Animate metric value
            const value = this.querySelector('.metric-value');
            if (value) {
                value.style.transform = 'scale(1.2)';
                value.style.transition = 'transform 0.3s ease';
                setTimeout(() => {
                    value.style.transform = 'scale(1)';
                }, 300);
            }
        });
    });
}

function startQuantumStatusUpdates() {
    // Simulate real-time quantum system updates
    setInterval(() => {
        updateQuantumMetrics();
    }, 5000);
    
    // Update progress bars periodically
    setInterval(() => {
        const quantumToggle = document.getElementById('quantumToggle');
        if (quantumToggle && quantumToggle.checked) {
            const progress = 85 + Math.random() * 10; // 85-95%
            updateQuantumProgress(progress);
        }
    }, 3000);
}

function updateQuantumMetrics() {
    // Update quantum indicators with slight variations
    const indicators = document.querySelectorAll('.quantum-indicator.active');
    indicators.forEach(indicator => {
        // Simulate quantum fluctuations
        const opacity = 0.7 + Math.random() * 0.3;
        indicator.style.opacity = opacity;
        
        setTimeout(() => {
            indicator.style.opacity = '1';
        }, 500);
    });
    
    // Update metric values with slight variations
    const metricValues = document.querySelectorAll('.metric-value');
    metricValues.forEach((value, index) => {
        const text = value.textContent;
        
        // Only update if it contains numbers
        if (text.match(/\d/)) {
            // Add subtle glow effect
            value.style.textShadow = '0 0 10px rgba(0, 230, 118, 0.5)';
            setTimeout(() => {
                value.style.textShadow = 'none';
            }, 1000);
        }
    });
}

function updateQuantumProgress(percentage) {
    const progressBar = document.getElementById('quantumProgress');
    if (progressBar) {
        progressBar.style.width = percentage + '%';
        progressBar.textContent = Math.round(percentage) + '%';
        progressBar.setAttribute('aria-valuenow', percentage);
        
        // Update status text
        const statusElement = document.getElementById('quantumStatus');
        if (statusElement) {
            if (percentage > 80) {
                statusElement.textContent = 'Optimal';
                statusElement.className = 'badge bg-success';
            } else if (percentage > 50) {
                statusElement.textContent = 'Active';
                statusElement.className = 'badge bg-warning';
            } else {
                statusElement.textContent = 'Idle';
                statusElement.className = 'badge bg-secondary';
            }
        }
    }
}

function createRippleEffect(element, event) {
    const rect = element.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    const x = event.clientX - rect.left - size / 2;
    const y = event.clientY - rect.top - size / 2;
    
    const ripple = document.createElement('div');
    ripple.style.cssText = `
        position: absolute;
        width: ${size}px;
        height: ${size}px;
        left: ${x}px;
        top: ${y}px;
        background: rgba(0, 230, 118, 0.3);
        border-radius: 50%;
        transform: scale(0);
        animation: ripple 0.6s ease-out;
        pointer-events: none;
    `;
    
    // Ensure button has relative positioning
    if (getComputedStyle(element).position === 'static') {
        element.style.position = 'relative';
    }
    
    element.appendChild(ripple);
    
    // Remove ripple after animation
    setTimeout(() => {
        if (ripple.parentNode) {
            ripple.parentNode.removeChild(ripple);
        }
    }, 600);
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes ripple {
        from {
            transform: scale(0);
            opacity: 1;
        }
        to {
            transform: scale(2);
            opacity: 0;
        }
    }
    
    .scanning::before {
        animation: scan 2s linear infinite;
    }
    
    .quantum-enhanced {
        --quantum-glow-intensity: 1.5;
    }
    
    .quantum-enhanced .quantum-indicator.active {
        box-shadow: 0 0 20px var(--quantum-accent);
    }
    
    .quantum-enhanced .card:hover {
        box-shadow: 
            0 12px 40px rgba(0, 0, 0, 0.4),
            0 0 30px rgba(0, 230, 118, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .quantum-enhanced .progress-bar {
        box-shadow: 0 0 15px rgba(0, 230, 118, 0.5);
    }
    
    .card {
        transition: all 0.3s ease;
        opacity: 0;
        transform: translateY(20px);
    }
    
    .btn {
        overflow: hidden;
    }
    
    .metric-card {
        cursor: pointer;
        user-select: none;
    }
`;

document.head.appendChild(style);

// Export for use in other scripts
window.QuantumEffects = {
    updateQuantumProgress,
    createRippleEffect,
    updateQuantumMetrics
};