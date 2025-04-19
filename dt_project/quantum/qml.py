"""
Quantum Machine Learning Module
Implements quantum-enhanced machine learning for performance prediction.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import time
import json
import os
import warnings

# Global variables to track availability
PENNYLANE_AVAILABLE = False
SKLEARN_AVAILABLE = False

try:
    import pennylane as qml
    from pennylane import numpy as qnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    warnings.warn("PennyLane not available. Quantum ML features will be limited.")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. ML features will be limited.")
    # Define placeholders for required classes
    class StandardScaler:
        """Placeholder for sklearn's StandardScaler when not available."""
        def fit_transform(self, X):
            return X
        def transform(self, X):
            return X
        def inverse_transform(self, X):
            return X
    
    def train_test_split(*args, **kwargs):
        """Placeholder for train_test_split when sklearn is not available."""
        X = args[0]
        y = args[1]
        test_size = kwargs.get('test_size', 0.2)
        # Simple split - no randomization
        n = len(X)
        split_idx = int(n * (1 - test_size))
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

from dt_project.config import ConfigManager

logger = logging.getLogger(__name__)

class DataEncoding:
    """Data encoding methods for quantum machine learning."""
    
    @staticmethod
    def angle_encoding(x, wires):
        """
        Angle encoding: encode features as rotation angles.
        
        Args:
            x: Input features
            wires: Quantum wires to apply encoding to
        """
        if not PENNYLANE_AVAILABLE:
            logger.warning("PennyLane not available. Angle encoding will have no effect.")
            return
            
        # Ensure x is a numpy array and normalized to [0, 2π]
        x_normalized = np.array(x) * np.pi
        
        for i, wire in enumerate(wires):
            if i < len(x_normalized):
                qml.RX(x_normalized[i], wires=wire)
                qml.RY(x_normalized[i], wires=wire)
    
    @staticmethod
    def amplitude_encoding(x, wires):
        """
        Amplitude encoding: encode features in the amplitudes of a quantum state.
        
        Args:
            x: Input features
            wires: Quantum wires to apply encoding to
        """
        if not PENNYLANE_AVAILABLE:
            logger.warning("PennyLane not available. Amplitude encoding will have no effect.")
            return
            
        # Normalize input vector
        x_normalized = np.array(x) / np.linalg.norm(x)
        qml.AmplitudeEmbedding(features=x_normalized, wires=wires, normalize=True)
    
    @staticmethod
    def basis_encoding(x, wires):
        """
        Basis encoding: encode binary features in the computational basis.
        
        Args:
            x: Input features (binary or discretized)
            wires: Quantum wires to apply encoding to
        """
        if not PENNYLANE_AVAILABLE:
            logger.warning("PennyLane not available. Basis encoding will have no effect.")
            return
            
        # Convert features to binary (0 or 1)
        x_binary = np.array(x) > 0.5
        
        # Apply X gates for 1s
        for i, wire in enumerate(wires):
            if i < len(x_binary) and x_binary[i]:
                qml.PauliX(wire)
    
    @staticmethod
    def zz_encoding(x, wires):
        """
        ZZ encoding: angle encoding followed by entanglement with ZZ gates.
        
        Args:
            x: Input features
            wires: Quantum wires to apply encoding to
        """
        if not PENNYLANE_AVAILABLE:
            logger.warning("PennyLane not available. ZZ encoding will have no effect.")
            return
            
        # First apply angle encoding
        DataEncoding.angle_encoding(x, wires)
        
        # Then apply ZZ entangling gates between adjacent qubits
        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i+1]])
            qml.RZ(np.pi/4, wires=wires[i+1])
            qml.CNOT(wires=[wires[i], wires[i+1]])

class QuantumML:
    """
    Quantum Machine Learning for performance prediction.
    Implements hybrid quantum-classical models for improved prediction accuracy.
    """
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize the Quantum Machine Learning module.
        
        Args:
            config: Configuration manager. If None, creates a new one.
        """
        self.config = config or ConfigManager()
        self._load_config()
        
        # Check dependencies
        self.qml_available = PENNYLANE_AVAILABLE and SKLEARN_AVAILABLE
        if not PENNYLANE_AVAILABLE:
            logger.warning("PennyLane is not available. Quantum ML features will be disabled.")
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn is not available. Quantum ML features will be disabled.")
        
        # Initialize backend
        self.device = None
        if self.qml_available and self.enabled:
            self._initialize_device()
            
        # Cache for models
        self.cache_dir = os.path.join(self.config.get("cache_dir", "data/cache"), "quantum_ml")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # For tracking training history
        self.training_history = []
        
        # Initialize encoding methods
        self.encoding_methods = {
            "angle": DataEncoding.angle_encoding,
            "amplitude": DataEncoding.amplitude_encoding,
            "basis": DataEncoding.basis_encoding,
            "zz": DataEncoding.zz_encoding
        }
        
        # Encoding comparison metrics
        self.encoding_comparison = {}
    
    def _load_config(self) -> None:
        """Load configuration parameters."""
        quantum_config = self.config.get("quantum", {})
        self.enabled = quantum_config.get("enabled", False)
        self.backend_name = quantum_config.get("backend", "simulator")
        self.n_qubits = quantum_config.get("n_qubits", 5)
        self.shots = quantum_config.get("shots", 1024)
        
        # ML specific parameters
        ml_config = quantum_config.get("ml", {})
        self.learning_rate = ml_config.get("learning_rate", 0.01)
        self.n_layers = ml_config.get("n_layers", 2)
        self.max_iterations = ml_config.get("max_iterations", 100)
        self.batch_size = ml_config.get("batch_size", 10)
        self.early_stopping = ml_config.get("early_stopping", True)
        self.feature_map = ml_config.get("feature_map", "zz")
        
        # Advanced configuration
        self.ansatz_type = ml_config.get("ansatz_type", "strongly_entangling")
        self.error_mitigation = ml_config.get("error_mitigation", False)
        self.auto_differentiation = ml_config.get("auto_differentiation", True)
    
    def _initialize_device(self) -> None:
        """Initialize the quantum device."""
        if not self.qml_available:
            return
        
        try:
            if self.backend_name == "simulator":
                self.device = qml.device("default.qubit", wires=self.n_qubits, shots=self.shots)
                logger.info(f"Using PennyLane default.qubit simulator with {self.n_qubits} qubits")
            elif self.backend_name.startswith("ibmq"):
                # Try to use IBM quantum device via PennyLane's qiskit plugin
                token = os.getenv("IBMQ_TOKEN")
                if token and PENNYLANE_AVAILABLE:
                    try:
                        self.device = qml.device("qiskit.ibmq", wires=self.n_qubits, 
                                               ibmqx_token=token, backend=self.backend_name,
                                               shots=self.shots)
                        logger.info(f"Using IBM Q device {self.backend_name} with {self.n_qubits} qubits")
                    except Exception as e:
                        logger.error(f"Error initializing IBM Q device: {str(e)}")
                        self.device = qml.device("default.qubit", wires=self.n_qubits, shots=self.shots)
                        logger.info("Falling back to default.qubit simulator")
                else:
                    logger.warning("IBMQ token not provided. Falling back to simulator.")
                    self.device = qml.device("default.qubit", wires=self.n_qubits, shots=self.shots)
            else:
                logger.warning(f"Unknown backend {self.backend_name}. Falling back to default.qubit.")
                self.device = qml.device("default.qubit", wires=self.n_qubits, shots=self.shots)
        except Exception as e:
            logger.error(f"Error initializing PennyLane device: {str(e)}")
            self.device = None
    
    def is_available(self) -> bool:
        """
        Check if quantum ML is available.
        
        Returns:
            True if quantum ML is available, False otherwise
        """
        return self.qml_available and self.enabled and self.device is not None
    
    def compare_encoding_strategies(self, X: np.ndarray, y: np.ndarray, 
                                   test_size: float = 0.2, 
                                   iterations: int = 20) -> Dict[str, Any]:
        """
        Compare different encoding strategies for the given dataset.
        
        Args:
            X: Features (numpy array)
            y: Target values (numpy array)
            test_size: Proportion of data to use for testing
            iterations: Number of training iterations for each encoding
            
        Returns:
            Dictionary with comparison results
        """
        if not self.is_available():
            logger.warning("Quantum ML not available. Cannot compare encoding strategies.")
            return {"success": False, "error": "Quantum ML not available"}
        
        # Prepare data
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        X_scaler = StandardScaler()
        X_scaled = X_scaler.fit_transform(X)
        
        y_scaler = StandardScaler()
        y_scaled = y_scaler.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_size)
        
        # Test each encoding method
        results = {}
        for encoding_name, encoding_function in self.encoding_methods.items():
            logger.info(f"Testing {encoding_name} encoding strategy...")
            try:
                # Create and train model with this encoding
                start_time = time.time()
                
                # Build circuit with specific encoding
                circuit, weight_shapes = self._build_quantum_circuit_with_encoding(
                    X.shape[1], encoding_function)
                
                # Initialize weights
                np.random.seed(42)
                weights = np.random.uniform(
                    low=-np.pi, high=np.pi, 
                    size=qml.templates.StronglyEntanglingLayers.shape(
                        n_layers=self.n_layers, n_wires=self.n_qubits)
                )
                
                # Define optimizer
                opt = qml.AdamOptimizer(stepsize=self.learning_rate)
                
                # Simple training loop for comparison
                batch_size = min(self.batch_size, len(X_train))
                
                # Define cost function
                def cost(weights, features, targets):
                    predictions = [circuit(x, weights) for x in features]
                    predictions = np.array(predictions)
                    return np.mean((predictions - targets) ** 2)
                
                # Train for fixed number of iterations
                train_costs = []
                test_costs = []
                
                for i in range(iterations):
                    # Select batch
                    batch_indices = np.random.choice(len(X_train), size=batch_size, replace=False)
                    X_batch = X_train[batch_indices]
                    y_batch = y_train[batch_indices]
                    
                    # Update weights
                    weights, train_cost = opt.step_and_cost(
                        lambda w: cost(w, X_batch, y_batch), weights)
                    
                    # Evaluate test cost
                    test_cost = cost(weights, X_test, y_test)
                    
                    train_costs.append(float(train_cost))
                    test_costs.append(float(test_cost))
                
                # Make predictions with final model
                predictions = np.array([circuit(x, weights) for x in X_test])
                
                # Calculate metrics
                mse = np.mean((predictions - y_test) ** 2)
                training_time = time.time() - start_time
                
                # Store results
                results[encoding_name] = {
                    "mse": float(mse),
                    "training_time": training_time,
                    "train_costs": train_costs,
                    "test_costs": test_costs,
                    "final_train_cost": train_costs[-1],
                    "final_test_cost": test_costs[-1]
                }
                
                logger.info(f"{encoding_name} encoding: MSE = {mse:.4f}, Time = {training_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error testing {encoding_name} encoding: {str(e)}")
                results[encoding_name] = {"error": str(e)}
        
        # Determine best encoding
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        if valid_results:
            best_encoding = min(valid_results.items(), key=lambda x: x[1]["mse"])[0]
            best_mse = valid_results[best_encoding]["mse"]
            logger.info(f"Best encoding strategy: {best_encoding} (MSE: {best_mse:.4f})")
            
            comparison_summary = {
                "best_encoding": best_encoding,
                "best_mse": best_mse,
                "details": results
            }
        else:
            comparison_summary = {
                "error": "All encoding strategies failed",
                "details": results
            }
        
        self.encoding_comparison = comparison_summary
        return comparison_summary
    
    def _build_quantum_circuit_with_encoding(self, n_features, encoding_function):
        """
        Build a quantum circuit with the specified encoding function.
        
        Args:
            n_features: Number of input features
            encoding_function: Function to use for data encoding
            
        Returns:
            quantum circuit, weight shapes
        """
        n_qubits = min(self.n_qubits, max(3, int(np.ceil(np.log2(n_features)))))
        wires = list(range(n_qubits))
        
        # Parameter shapes for the variational circuit
        weight_shapes = {"weights": (self.n_layers, n_qubits, 3)}
        
        # Define quantum circuit with the given encoding
        @qml.qnode(self.device)
        def circuit(inputs, weights):
            # Apply the specified encoding function
            encoding_function(inputs, wires)
            
            # Apply variational circuit
            self._variational_circuit(weights, wires)
            
            # Return expectation values
            return [qml.expval(qml.PauliZ(w)) for w in wires]
        
        return circuit, weight_shapes
    
    def _zz_feature_map(self, x, wires):
        """ZZ feature map for data encoding."""
        # Normalize input features
        x_norm = x / np.pi
        
        # First layer of Hadamards
        for i in wires:
            qml.Hadamard(wires=i)
        
        # First rotation layer
        for i, wi in enumerate(wires):
            qml.RZ(x_norm[i], wires=wi)
        
        # Second rotation with entanglement
        for i, wi in enumerate(wires):
            for j, wj in enumerate(wires[i+1:], i+1):
                qml.CNOT(wires=[wi, wj])
                qml.RZ(x_norm[i] * x_norm[j], wires=wj)
                qml.CNOT(wires=[wi, wj])
    
    def _amplitude_feature_map(self, x, wires):
        """Amplitude encoding feature map."""
        # Normalize to unit vector
        x_norm = x / np.linalg.norm(x)
        
        # Prepare initial state
        qml.MottonenStatePreparation(x_norm, wires=wires)
    
    def _angle_feature_map(self, x, wires):
        """Angle encoding feature map."""
        # Scale features to be in [0, 2π]
        x_scaled = x * 2 * np.pi
        
        # Encode using rotation gates
        for i, wire in enumerate(wires):
            if i < len(x):
                qml.RX(x_scaled[i], wires=wire)
                
        # Add RY rotations for any remaining features (cyclic)
        for i, wire in enumerate(wires):
            j = i + len(wires)
            if j < len(x):
                qml.RY(x_scaled[j], wires=wire)
    
    def _variational_circuit(self, weights, wires):
        """
        Defines a variational quantum circuit based on the configured ansatz type.
        
        Args:
            weights: Trainable weights for the circuit
            wires: Qubit wires to use
        """
        if self.ansatz_type == "strongly_entangling":
            qml.StronglyEntanglingLayers(weights, wires=wires)
        elif self.ansatz_type == "basic":
            # Custom basic ansatz with rotation gates and entanglement
            for layer in range(self.n_layers):
                # Rotation gates with parameters
                for i, wire in enumerate(wires):
                    qml.RX(weights[layer][i][0], wires=wire)
                    qml.RY(weights[layer][i][1], wires=wire)
                    qml.RZ(weights[layer][i][2], wires=wire)
                
                # Entangling gates
                for i in range(len(wires) - 1):
                    qml.CNOT(wires=[wires[i], wires[i + 1]])
                
                # Optional extra entanglement
                if len(wires) > 2:
                    qml.CNOT(wires=[wires[-1], wires[0]])
        elif self.ansatz_type == "hardware_efficient":
            # Hardware-efficient ansatz
            for layer in range(self.n_layers):
                # Single-qubit rotations
                for i, wire in enumerate(wires):
                    qml.RY(weights[layer][i][0], wires=wire)
                    qml.RZ(weights[layer][i][1], wires=wire)
                
                # Entangling layer with CZ gates (more hardware-friendly)
                for i in range(len(wires) - 1):
                    qml.CZ(wires=[wires[i], wires[i + 1]])
        elif self.ansatz_type == "real_amplitudes":
            # Real amplitudes ansatz
            for layer in range(self.n_layers):
                # Rotation Y gates
                for i, wire in enumerate(wires):
                    qml.RY(weights[layer][i][0], wires=wire)
                
                # Entanglement
                for i in range(len(wires) - 1):
                    qml.CNOT(wires=[wires[i], wires[i + 1]])
                
                if layer < self.n_layers - 1:
                    # Additional rotations between entanglements for all but last layer
                    for i, wire in enumerate(wires):
                        qml.RY(weights[layer][i][1], wires=wire)
        else:
            # Default to strongly entangling if unknown ansatz type
            qml.StronglyEntanglingLayers(weights, wires=wires)
    
    def _apply_error_mitigation(self, circuit, backend=None):
        """
        Apply error mitigation to quantum circuit.
        
        Args:
            circuit: Quantum circuit function
            backend: Optional specific backend to use
            
        Returns:
            Mitigated circuit function
        """
        if not self.error_mitigation or not self.qml_available:
            return circuit
            
        # If not using PennyLane, return original circuit
        if not PENNYLANE_AVAILABLE:
            return circuit
            
        try:
            # Create a wrapped circuit function that applies error mitigation
            def mitigated_circuit(*args, **kwargs):
                # Original result
                result = circuit(*args, **kwargs)
                
                # Apply basic readout error mitigation
                # This is simplified for demonstration purposes
                # In a real implementation, this would use PennyLane's error mitigation
                # techniques or custom implementations
                
                # For now, just return the original result with a note that mitigation was attempted
                return result
                
            return mitigated_circuit
            
        except Exception as e:
            logger.error(f"Error applying error mitigation: {str(e)}")
            return circuit
    
    def _build_quantum_circuit(self, n_features):
        """
        Build a quantum circuit for classification or regression.
        
        Args:
            n_features: Number of input features
            
        Returns:
            Quantum circuit function
        """
        # Determine number of qubits needed
        n_qubits = min(self.n_qubits, max(3, int(np.ceil(np.log2(n_features)))))
        wires = list(range(n_qubits))
        
        # Parameter shapes for the variational circuit based on ansatz type
        if self.ansatz_type == "strongly_entangling":
            weight_shapes = {"weights": qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=n_qubits)}
        elif self.ansatz_type == "hardware_efficient":
            weight_shapes = {"weights": (self.n_layers, n_qubits, 2)} 
        elif self.ansatz_type == "real_amplitudes":
            weight_shapes = {"weights": (self.n_layers, n_qubits, 2)}
        else:  # basic or unknown
            weight_shapes = {"weights": (self.n_layers, n_qubits, 3)}
        
        # Define the quantum neural network
        @qml.qnode(self.device, diff_method="parameter-shift" if self.auto_differentiation else "finite-diff")
        def quantum_circuit(inputs, weights):
            # Apply feature map based on configuration
            encoding_fn = self.encoding_methods.get(self.feature_map, self.encoding_methods["zz"])
            encoding_fn(inputs, wires)
            
            # Apply variational circuit
            self._variational_circuit(weights, wires)
            
            # Return expectation value
            return [qml.expval(qml.PauliZ(w)) for w in wires]
        
        # Apply error mitigation if enabled
        if self.error_mitigation:
            return self._apply_error_mitigation(quantum_circuit), weight_shapes
            
        return quantum_circuit, weight_shapes
    
    def train_model(self, X, y, test_size=0.2, verbose=True, save_model=True) -> Dict[str, Any]:
        """
        Train a quantum machine learning model.
        
        Args:
            X: Features (numpy array)
            y: Target values (numpy array)
            test_size: Proportion of data to use for testing
            verbose: Whether to print progress
            save_model: Whether to save the trained model to cache
            
        Returns:
            Dictionary with training results
        """
        if not self.is_available():
            logger.warning("Quantum ML not available. Cannot train model.")
            return {"success": False, "error": "Quantum ML not available"}
        
        start_time = time.time()
        
        # Prepare data
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
        # Handle 1D y
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        # Normalize data
        X_scaler = StandardScaler()
        X_scaled = X_scaler.fit_transform(X)
        
        y_scaler = StandardScaler()
        y_scaled = y_scaler.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_size)
        
        # Build model
        n_features = X.shape[1]
        qnn, weight_shapes = self._build_quantum_circuit(n_features)
        
        # Initialize weights
        np.random.seed(42)
        
        if self.ansatz_type == "strongly_entangling":
            weights_init = np.random.uniform(
                low=-np.pi, high=np.pi, 
                size=qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=self.n_qubits)
            )
        else:
            weights_init = np.random.uniform(
                low=-np.pi, high=np.pi, 
                size=weight_shapes["weights"]
            )
        
        # Define cost function
        def mean_squared_error(weights, features, targets, batch_size=None):
            if batch_size is None:
                batch_size = len(features)
            
            # Randomly select a batch
            indices = np.random.choice(len(features), size=batch_size, replace=False)
            feats = features[indices]
            targs = targets[indices]
            
            # Get predictions for all samples
            predictions = []
            for i in range(batch_size):
                pred = qnn(feats[i], weights)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            # Reshape targets for compatibility if needed
            if len(targs.shape) == 1:
                targs = targs.reshape(-1, 1)
                
            # Ensure predictions match target shape
            if targs.shape[1] == 1 and predictions.shape[1] > 1:
                predictions = np.mean(predictions, axis=1).reshape(-1, 1)
            
            # Calculate MSE
            loss = np.mean((predictions - targs) ** 2)
            return loss
        
        # Define optimizer
        opt = qml.AdamOptimizer(stepsize=self.learning_rate)
        
        # Training loop
        weights = weights_init
        history = []
        best_weights = weights
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for i in range(self.max_iterations):
            # Training step
            weights, loss = opt.step_and_cost(
                lambda w: mean_squared_error(w, X_train, y_train, self.batch_size), 
                weights
            )
            
            # Evaluate on test set
            test_loss = mean_squared_error(weights, X_test, y_test)
            
            # Store history
            history.append({
                "iteration": i,
                "train_loss": float(loss),
                "test_loss": float(test_loss)
            })
            
            # Print progress
            if verbose and i % 10 == 0:
                logger.info(f"Iteration {i}: Train loss = {loss:.4f}, Test loss = {test_loss:.4f}")
            
            # Check for early stopping
            if test_loss < best_loss:
                best_loss = test_loss
                best_weights = weights.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if self.early_stopping and patience_counter >= patience:
                    if verbose:
                        logger.info(f"Early stopping at iteration {i}")
                    break
        
        training_time = time.time() - start_time
        
        # Create the model object
        model = {
            "weights": best_weights,
            "circuit": qnn,
            "X_scaler": X_scaler,
            "y_scaler": y_scaler,
            "n_features": n_features,
            "n_qubits": self.n_qubits,
            "training_time": training_time,
            "n_layers": self.n_layers,
            "feature_map": self.feature_map,
            "ansatz_type": self.ansatz_type,
            "backend": str(self.device),
            "history": history,
            "best_loss": best_loss
        }
        
        self.training_history = history
        self.current_model = model
        
        if save_model:
            model_name = f"qml_model_{int(time.time())}"
            self._save_model(model, model_name)
        
        # Return results
        return {
            "success": True,
            "training_time": training_time,
            "final_train_loss": float(loss),
            "final_test_loss": float(test_loss),
            "best_test_loss": float(best_loss),
            "n_iterations": i + 1,
            "early_stopped": patience_counter >= patience if self.early_stopping else False,
            "model_name": model_name if save_model else None,
            "ansatz_type": self.ansatz_type,
            "feature_map": self.feature_map
        }
    
    def predict(self, X, model=None):
        """
        Make predictions using the trained quantum model.
        
        Args:
            X: Features to predict (numpy array)
            model: Model to use (if None, uses the most recent model)
            
        Returns:
            Predictions
        """
        if model is None:
            if not hasattr(self, 'current_model'):
                raise ValueError("No model available. Train a model first.")
            model = self.current_model
        
        # Prepare data
        X = np.array(X, dtype=np.float64)
        
        # Scale input
        X_scaled = model["X_scaler"].transform(X)
        
        # Make predictions
        predictions = []
        qnn = model["circuit"]
        weights = model["weights"]
        
        for i in range(len(X_scaled)):
            pred = qnn(X_scaled[i], weights)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Ensure predictions have the right shape for inverse transform
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        elif predictions.shape[1] > 1:
            # If output dimension is wrong, average across qubits
            predictions = np.mean(predictions, axis=1).reshape(-1, 1)
        
        # Inverse transform to get original scale
        y_pred = model["y_scaler"].inverse_transform(predictions)
        
        return y_pred
    
    def _save_model(self, model, model_name):
        """
        Save a trained model to disk.
        
        Args:
            model: Model to save
            model_name: Name to save the model under
        """
        model_file = os.path.join(self.cache_dir, f"{model_name}.pkl")
        
        try:
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model saved to {model_file}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def load_model(self, model_name):
        """
        Load a trained model from disk.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model
        """
        model_file = os.path.join(self.cache_dir, f"{model_name}.pkl")
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model {model_name} not found")
        
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {model_file}")
            self.current_model = model
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def get_training_history(self):
        """
        Get the training history of the most recent model.
        
        Returns:
            List of dictionaries with training metrics
        """
        return self.training_history
    
    def compare_with_classical(self, X, y, classical_model, test_size=0.2, verbose=True):
        """
        Compare quantum model with a classical ML model.
        
        Args:
            X: Features (numpy array)
            y: Target values (numpy array)
            classical_model: Scikit-learn compatible model
            test_size: Proportion of data to use for testing
            verbose: Whether to print progress
            
        Returns:
            Dictionary with comparison results
        """
        # Train quantum model
        quantum_results = self.train_model(X, y, test_size=test_size, verbose=verbose)
        
        # Prepare data for classical model
        X = np.array(X)
        y = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train classical model
        classical_start_time = time.time()
        classical_model.fit(X_train, y_train)
        classical_training_time = time.time() - classical_start_time
        
        # Evaluate classical model
        y_pred = classical_model.predict(X_test)
        classical_mse = np.mean((y_pred - y_test) ** 2)
        
        # Get quantum model predictions
        y_pred_quantum = self.predict(X_test)
        quantum_mse = np.mean((y_pred_quantum - y_test) ** 2)
        
        # Prepare comparison results
        comparison = {
            "quantum_mse": float(quantum_mse),
            "classical_mse": float(classical_mse),
            "quantum_training_time": quantum_results["training_time"],
            "classical_training_time": float(classical_training_time),
            "relative_improvement": 1 - quantum_mse / max(classical_mse, 1e-10),
            "speedup_factor": classical_training_time / max(quantum_results["training_time"], 1e-10),
            "quantum_iterations": quantum_results["n_iterations"]
        }
        
        if verbose:
            logger.info(f"Quantum MSE: {quantum_mse:.4f}, Classical MSE: {classical_mse:.4f}")
            logger.info(f"Quantum training time: {quantum_results['training_time']:.2f}s, Classical training time: {classical_training_time:.2f}s")
            logger.info(f"Relative improvement: {comparison['relative_improvement']*100:.1f}%")
        
        return comparison 