"""
Quantum Machine Learning Pipeline for Quantum Trail.
Integrates quantum computing with machine learning workflows.
"""

import asyncio
import time
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import structlog

from dt_project.quantum.async_quantum_backend import AsyncQuantumProcessor
from dt_project.performance.cache import ml_cache
from dt_project.performance.profiler import profile
from dt_project.monitoring.metrics import metrics

logger = structlog.get_logger(__name__)

class QuantumMLAlgorithm(Enum):
    """Types of quantum ML algorithms."""
    QNN = "quantum_neural_network"
    QSVM = "quantum_support_vector_machine"
    QKMeans = "quantum_k_means"
    QPCA = "quantum_principal_component_analysis"
    VQC = "variational_quantum_classifier"
    QAOA_ML = "qaoa_for_ml"
    QBM = "quantum_boltzmann_machine"
    HYBRID = "hybrid_quantum_classical"

class DataEncodingType(Enum):
    """Types of quantum data encoding."""
    AMPLITUDE = "amplitude_encoding"
    ANGLE = "angle_encoding"
    BASIS = "basis_encoding"
    ARBITRARY = "arbitrary_encoding"

@dataclass
class QuantumMLConfig:
    """Configuration for quantum ML pipeline."""
    algorithm: QuantumMLAlgorithm
    n_qubits: int
    encoding_type: DataEncodingType
    max_iterations: int = 100
    learning_rate: float = 0.01
    batch_size: int = 32
    convergence_threshold: float = 1e-6
    use_quantum_advantage: bool = True
    hybrid_classical_layers: List[int] = field(default_factory=list)
    regularization: float = 0.0
    noise_mitigation: bool = True
    backend_preference: str = "simulator"

@dataclass
class QuantumMLResult:
    """Result of quantum ML training or inference."""
    algorithm: QuantumMLAlgorithm
    accuracy: float
    loss: float
    quantum_advantage: float
    training_time: float
    convergence_iterations: int
    model_parameters: np.ndarray
    quantum_fidelity: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class QuantumDataEncoder:
    """Encodes classical data into quantum states."""
    
    def __init__(self, encoding_type: DataEncodingType, n_qubits: int):
        self.encoding_type = encoding_type
        self.n_qubits = n_qubits
        self.max_features = self._calculate_max_features()
        
    def _calculate_max_features(self) -> int:
        """Calculate maximum number of features that can be encoded."""
        if self.encoding_type == DataEncodingType.AMPLITUDE:
            return 2 ** self.n_qubits
        elif self.encoding_type == DataEncodingType.ANGLE:
            return self.n_qubits
        elif self.encoding_type == DataEncodingType.BASIS:
            return self.n_qubits
        else:  # ARBITRARY
            return self.n_qubits * 2
    
    def encode(self, data: np.ndarray) -> Dict[str, Any]:
        """Encode classical data into quantum circuit parameters."""
        if data.shape[-1] > self.max_features:
            raise ValueError(f"Data has {data.shape[-1]} features, but encoding supports max {self.max_features}")
        
        if self.encoding_type == DataEncodingType.AMPLITUDE:
            return self._amplitude_encoding(data)
        elif self.encoding_type == DataEncodingType.ANGLE:
            return self._angle_encoding(data)
        elif self.encoding_type == DataEncodingType.BASIS:
            return self._basis_encoding(data)
        else:  # ARBITRARY
            return self._arbitrary_encoding(data)
    
    def _amplitude_encoding(self, data: np.ndarray) -> Dict[str, Any]:
        """Encode data as quantum amplitudes."""
        # Normalize data to valid probability amplitudes
        normalized_data = data / np.linalg.norm(data, axis=-1, keepdims=True)
        
        # Pad to 2^n_qubits if needed
        target_size = 2 ** self.n_qubits
        if normalized_data.shape[-1] < target_size:
            padding = np.zeros((*normalized_data.shape[:-1], target_size - normalized_data.shape[-1]))
            normalized_data = np.concatenate([normalized_data, padding], axis=-1)
        
        return {
            'encoding_type': 'amplitude',
            'amplitudes': normalized_data.tolist(),
            'n_qubits': self.n_qubits
        }
    
    def _angle_encoding(self, data: np.ndarray) -> Dict[str, Any]:
        """Encode data as rotation angles."""
        # Normalize data to [0, 2À] range
        normalized_data = (data - data.min()) / (data.max() - data.min()) * 2 * np.pi
        
        return {
            'encoding_type': 'angle',
            'angles': normalized_data.tolist(),
            'n_qubits': self.n_qubits
        }
    
    def _basis_encoding(self, data: np.ndarray) -> Dict[str, Any]:
        """Encode data as computational basis states."""
        # Convert to binary representation
        binary_data = (data > np.median(data)).astype(int)
        
        return {
            'encoding_type': 'basis',
            'binary_values': binary_data.tolist(),
            'n_qubits': self.n_qubits
        }
    
    def _arbitrary_encoding(self, data: np.ndarray) -> Dict[str, Any]:
        """Arbitrary encoding using multiple rotation gates."""
        # Use both RY and RZ rotations
        angles_y = (data - data.min()) / (data.max() - data.min()) * np.pi
        angles_z = (data - data.mean()) / data.std() * np.pi
        
        return {
            'encoding_type': 'arbitrary',
            'angles_y': angles_y.tolist(),
            'angles_z': angles_z.tolist(),
            'n_qubits': self.n_qubits
        }

class QuantumMLModel(ABC):
    """Abstract base class for quantum ML models."""
    
    def __init__(self, config: QuantumMLConfig):
        self.config = config
        self.encoder = QuantumDataEncoder(config.encoding_type, config.n_qubits)
        self.parameters: Optional[np.ndarray] = None
        self.is_trained = False
        self.training_history = []
        
    @abstractmethod
    async def train(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: Optional[np.ndarray] = None, 
                   y_val: Optional[np.ndarray] = None) -> QuantumMLResult:
        """Train the quantum ML model."""
        pass
    
    @abstractmethod
    async def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> bool:
        """Save the trained model."""
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> bool:
        """Load a pre-trained model."""
        pass

class VariationalQuantumClassifier(QuantumMLModel):
    """Variational Quantum Classifier implementation."""
    
    def __init__(self, config: QuantumMLConfig):
        super().__init__(config)
        self.circuit_depth = 3  # Default ansatz depth
        self.param_count = self.config.n_qubits * self.circuit_depth * 3  # 3 parameters per qubit per layer
    
    async def train(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: Optional[np.ndarray] = None,
                   y_val: Optional[np.ndarray] = None) -> QuantumMLResult:
        """Train the VQC using variational optimization."""
        
        start_time = time.time()
        
        # Initialize parameters
        self.parameters = np.random.uniform(-np.pi, np.pi, self.param_count)
        
        # Training loop
        best_loss = float('inf')
        convergence_count = 0
        
        for iteration in range(self.config.max_iterations):
            # Compute gradients and update parameters
            gradients = await self._compute_gradients(X_train, y_train)
            self.parameters -= self.config.learning_rate * gradients
            
            # Compute current loss
            current_loss = await self._compute_loss(X_train, y_train)
            self.training_history.append(current_loss)
            
            # Check convergence
            if abs(best_loss - current_loss) < self.config.convergence_threshold:
                convergence_count += 1
                if convergence_count >= 5:
                    logger.info(f"VQC converged after {iteration + 1} iterations")
                    break
            else:
                convergence_count = 0
            
            best_loss = min(best_loss, current_loss)
            
            if iteration % 10 == 0:
                logger.debug(f"VQC iteration {iteration}: loss = {current_loss:.6f}")
        
        # Compute final accuracy
        train_predictions = await self.predict(X_train)
        train_accuracy = self._compute_accuracy(train_predictions, y_train)
        
        val_accuracy = train_accuracy
        if X_val is not None and y_val is not None:
            val_predictions = await self.predict(X_val)
            val_accuracy = self._compute_accuracy(val_predictions, y_val)
        
        training_time = time.time() - start_time
        self.is_trained = True
        
        return QuantumMLResult(
            algorithm=QuantumMLAlgorithm.VQC,
            accuracy=val_accuracy,
            loss=best_loss,
            quantum_advantage=self._estimate_quantum_advantage(training_time),
            training_time=training_time,
            convergence_iterations=len(self.training_history),
            model_parameters=self.parameters.copy(),
            quantum_fidelity=0.95,  # Mock fidelity
            metadata={
                'train_accuracy': train_accuracy,
                'circuit_depth': self.circuit_depth,
                'param_count': self.param_count,
                'encoding_type': self.config.encoding_type.value
            }
        )
    
    async def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute parameter gradients using parameter shift rule."""
        gradients = np.zeros_like(self.parameters)
        
        # Parameter shift rule for quantum gradients
        shift = np.pi / 2
        
        for i in range(len(self.parameters)):
            # Forward shift
            params_plus = self.parameters.copy()
            params_plus[i] += shift
            loss_plus = await self._compute_loss_with_params(X, y, params_plus)
            
            # Backward shift
            params_minus = self.parameters.copy()
            params_minus[i] -= shift
            loss_minus = await self._compute_loss_with_params(X, y, params_minus)
            
            # Gradient
            gradients[i] = (loss_plus - loss_minus) / 2
        
        return gradients
    
    async def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute loss with current parameters."""
        return await self._compute_loss_with_params(X, y, self.parameters)
    
    async def _compute_loss_with_params(self, X: np.ndarray, y: np.ndarray, params: np.ndarray) -> float:
        """Compute loss with given parameters."""
        # Simulate quantum circuit evaluation
        predictions = await self._predict_with_params(X, params)
        
        # Cross-entropy loss
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        
        # Add regularization
        if self.config.regularization > 0:
            loss += self.config.regularization * np.sum(params ** 2)
        
        return loss
    
    async def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        return await self._predict_with_params(X, self.parameters)
    
    async def _predict_with_params(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Make predictions with given parameters."""
        # Simulate quantum circuit execution
        await asyncio.sleep(0.01 * len(X))  # Simulate quantum computation time
        
        # Mock quantum circuit evaluation
        # In reality, this would construct and execute quantum circuits
        encoded_data = self.encoder.encode(X)
        
        # Simulate variational circuit output
        n_samples = X.shape[0]
        
        # Create mock predictions based on data and parameters
        feature_interactions = np.sum(X * params[:X.shape[1]], axis=1)
        predictions = 1 / (1 + np.exp(-feature_interactions))  # Sigmoid activation
        
        # Add quantum noise
        noise_level = 0.05 if not self.config.noise_mitigation else 0.02
        predictions += np.random.normal(0, noise_level, predictions.shape)
        predictions = np.clip(predictions, 0, 1)
        
        return predictions
    
    def _compute_accuracy(self, predictions: np.ndarray, y_true: np.ndarray) -> float:
        """Compute classification accuracy."""
        predicted_labels = (predictions > 0.5).astype(int)
        return np.mean(predicted_labels == y_true)
    
    def _estimate_quantum_advantage(self, training_time: float) -> float:
        """Estimate quantum advantage factor."""
        # Mock quantum advantage calculation
        classical_time_estimate = training_time * 2.5  # Assume classical would take 2.5x longer
        return classical_time_estimate / training_time
    
    def save_model(self, path: str) -> bool:
        """Save the trained model."""
        try:
            model_data = {
                'parameters': self.parameters.tolist() if self.parameters is not None else None,
                'config': {
                    'algorithm': self.config.algorithm.value,
                    'n_qubits': self.config.n_qubits,
                    'encoding_type': self.config.encoding_type.value,
                    'max_iterations': self.config.max_iterations,
                    'learning_rate': self.config.learning_rate
                },
                'is_trained': self.is_trained,
                'training_history': self.training_history,
                'circuit_depth': self.circuit_depth,
                'param_count': self.param_count,
                'saved_at': datetime.utcnow().isoformat()
            }
            
            with open(path, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save VQC model: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Load a pre-trained model."""
        try:
            with open(path, 'r') as f:
                model_data = json.load(f)
            
            if model_data['parameters']:
                self.parameters = np.array(model_data['parameters'])
            
            self.is_trained = model_data['is_trained']
            self.training_history = model_data['training_history']
            self.circuit_depth = model_data['circuit_depth']
            self.param_count = model_data['param_count']
            
            return True
        except Exception as e:
            logger.error(f"Failed to load VQC model: {e}")
            return False

class QuantumNeuralNetwork(QuantumMLModel):
    """Quantum Neural Network implementation."""
    
    def __init__(self, config: QuantumMLConfig):
        super().__init__(config)
        self.layer_count = 4  # Default number of quantum layers
        self.param_count = self.config.n_qubits * self.layer_count * 6  # More parameters for QNN
    
    async def train(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: Optional[np.ndarray] = None,
                   y_val: Optional[np.ndarray] = None) -> QuantumMLResult:
        """Train the QNN."""
        
        start_time = time.time()
        
        # Initialize parameters
        self.parameters = np.random.uniform(-np.pi, np.pi, self.param_count)
        
        # Training with mini-batches
        n_samples = X_train.shape[0]
        n_batches = max(1, n_samples // self.config.batch_size)
        
        best_loss = float('inf')
        
        for iteration in range(self.config.max_iterations):
            epoch_loss = 0.0
            
            # Mini-batch training
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.config.batch_size
                end_idx = min(start_idx + self.config.batch_size, n_samples)
                
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                
                # Compute gradients for batch
                gradients = await self._compute_qnn_gradients(X_batch, y_batch)
                
                # Update parameters
                self.parameters -= self.config.learning_rate * gradients
                
                # Compute batch loss
                batch_loss = await self._compute_qnn_loss(X_batch, y_batch)
                epoch_loss += batch_loss
            
            epoch_loss /= n_batches
            self.training_history.append(epoch_loss)
            
            # Check convergence
            if abs(best_loss - epoch_loss) < self.config.convergence_threshold:
                logger.info(f"QNN converged after {iteration + 1} iterations")
                break
            
            best_loss = min(best_loss, epoch_loss)
            
            if iteration % 10 == 0:
                logger.debug(f"QNN iteration {iteration}: loss = {epoch_loss:.6f}")
        
        # Compute final accuracy
        train_predictions = await self.predict(X_train)
        train_accuracy = self._compute_accuracy(train_predictions, y_train)
        
        val_accuracy = train_accuracy
        if X_val is not None and y_val is not None:
            val_predictions = await self.predict(X_val)
            val_accuracy = self._compute_accuracy(val_predictions, y_val)
        
        training_time = time.time() - start_time
        self.is_trained = True
        
        return QuantumMLResult(
            algorithm=QuantumMLAlgorithm.QNN,
            accuracy=val_accuracy,
            loss=best_loss,
            quantum_advantage=self._estimate_quantum_advantage(training_time),
            training_time=training_time,
            convergence_iterations=len(self.training_history),
            model_parameters=self.parameters.copy(),
            quantum_fidelity=0.93,  # Mock fidelity
            metadata={
                'train_accuracy': train_accuracy,
                'layer_count': self.layer_count,
                'param_count': self.param_count,
                'batch_size': self.config.batch_size
            }
        )
    
    async def _compute_qnn_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute QNN gradients."""
        # Simplified gradient computation for QNN
        gradients = np.zeros_like(self.parameters)
        
        # Mock gradient computation
        predictions = await self._predict_qnn(X)
        error = predictions - y
        
        # Approximate gradients
        for i in range(len(self.parameters)):
            gradients[i] = np.mean(error * np.sin(self.parameters[i] + X.sum(axis=1)))
        
        return gradients
    
    async def _compute_qnn_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute QNN loss."""
        predictions = await self._predict_qnn(X)
        mse_loss = np.mean((predictions - y) ** 2)
        
        # Add regularization
        if self.config.regularization > 0:
            mse_loss += self.config.regularization * np.sum(self.parameters ** 2)
        
        return mse_loss
    
    async def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained QNN."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        return await self._predict_qnn(X)
    
    async def _predict_qnn(self, X: np.ndarray) -> np.ndarray:
        """QNN prediction implementation."""
        # Simulate quantum neural network computation
        await asyncio.sleep(0.02 * len(X))  # Simulate computation time
        
        n_samples = X.shape[0]
        
        # Mock QNN forward pass
        layer_outputs = X.copy()
        
        for layer in range(self.layer_count):
            start_idx = layer * self.config.n_qubits * 6
            end_idx = start_idx + self.config.n_qubits * 6
            layer_params = self.parameters[start_idx:end_idx]
            
            # Apply quantum layer transformation
            layer_outputs = np.tanh(
                layer_outputs @ layer_params[:X.shape[1]].reshape(-1, 1) + 
                layer_params[X.shape[1]:][:n_samples].reshape(-1, 1)
            ).flatten()
            
            if len(layer_outputs) != n_samples:
                layer_outputs = layer_outputs[:n_samples]
        
        # Add quantum noise
        noise_level = 0.03 if not self.config.noise_mitigation else 0.01
        layer_outputs += np.random.normal(0, noise_level, layer_outputs.shape)
        
        return np.clip(layer_outputs, 0, 1)
    
    def _compute_accuracy(self, predictions: np.ndarray, y_true: np.ndarray) -> float:
        """Compute regression/classification accuracy."""
        if len(np.unique(y_true)) <= 2:  # Binary classification
            predicted_labels = (predictions > 0.5).astype(int)
            return np.mean(predicted_labels == y_true)
        else:  # Regression - use R²
            ss_res = np.sum((y_true - predictions) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / (ss_tot + 1e-15))
    
    def _estimate_quantum_advantage(self, training_time: float) -> float:
        """Estimate quantum advantage for QNN."""
        classical_time_estimate = training_time * 3.2  # QNN might have higher advantage
        return classical_time_estimate / training_time
    
    def save_model(self, path: str) -> bool:
        """Save the trained QNN model."""
        try:
            model_data = {
                'parameters': self.parameters.tolist() if self.parameters is not None else None,
                'config': {
                    'algorithm': self.config.algorithm.value,
                    'n_qubits': self.config.n_qubits,
                    'encoding_type': self.config.encoding_type.value,
                    'batch_size': self.config.batch_size,
                    'learning_rate': self.config.learning_rate
                },
                'is_trained': self.is_trained,
                'training_history': self.training_history,
                'layer_count': self.layer_count,
                'param_count': self.param_count,
                'saved_at': datetime.utcnow().isoformat()
            }
            
            with open(path, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save QNN model: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Load a pre-trained QNN model."""
        try:
            with open(path, 'r') as f:
                model_data = json.load(f)
            
            if model_data['parameters']:
                self.parameters = np.array(model_data['parameters'])
            
            self.is_trained = model_data['is_trained']
            self.training_history = model_data['training_history']
            self.layer_count = model_data['layer_count']
            self.param_count = model_data['param_count']
            
            return True
        except Exception as e:
            logger.error(f"Failed to load QNN model: {e}")
            return False

class QuantumMLPipeline:
    """Main quantum ML pipeline that orchestrates training and inference."""
    
    def __init__(self, quantum_processor: Optional[AsyncQuantumProcessor] = None):
        self.quantum_processor = quantum_processor
        self.models: Dict[str, QuantumMLModel] = {}
        self.pipelines: Dict[str, Dict[str, Any]] = {}
        
    @profile
    async def create_model(self, model_id: str, config: QuantumMLConfig) -> QuantumMLModel:
        """Create a new quantum ML model."""
        if config.algorithm == QuantumMLAlgorithm.VQC:
            model = VariationalQuantumClassifier(config)
        elif config.algorithm == QuantumMLAlgorithm.QNN:
            model = QuantumNeuralNetwork(config)
        else:
            raise ValueError(f"Unsupported algorithm: {config.algorithm}")
        
        self.models[model_id] = model
        logger.info(f"Created quantum ML model {model_id} with algorithm {config.algorithm.value}")
        
        return model
    
    @profile
    async def train_model(self, model_id: str, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: Optional[np.ndarray] = None, 
                         y_val: Optional[np.ndarray] = None) -> QuantumMLResult:
        """Train a quantum ML model."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        
        # Check cache for similar training job
        cache_key = f"train_{model_id}_{hash(X_train.tobytes())}_{hash(y_train.tobytes())}"
        cached_result = ml_cache.get(cache_key)
        
        if cached_result:
            logger.info(f"Using cached training result for model {model_id}")
            return cached_result
        
        logger.info(f"Starting training for quantum ML model {model_id}")
        
        # Train model
        result = await model.train(X_train, y_train, X_val, y_val)
        
        # Cache result
        ml_cache.put(cache_key, result, ttl_seconds=3600)
        
        # Record metrics
        if metrics:
            metrics.ml_models_trained_total.inc()
            metrics.ml_training_time.observe(result.training_time)
        
        logger.info(f"Training completed for model {model_id}: accuracy={result.accuracy:.3f}, loss={result.loss:.6f}")
        
        return result
    
    @profile
    async def predict(self, model_id: str, X: np.ndarray) -> np.ndarray:
        """Make predictions using a trained model."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        
        # Check cache for similar prediction
        cache_key = f"predict_{model_id}_{hash(X.tobytes())}"
        cached_result = ml_cache.get(cache_key)
        
        if cached_result is not None:
            logger.debug(f"Using cached prediction for model {model_id}")
            return cached_result
        
        # Make prediction
        predictions = await model.predict(X)
        
        # Cache result
        ml_cache.put(cache_key, predictions, ttl_seconds=1800)
        
        # Record metrics
        if metrics:
            metrics.ml_predictions_total.inc()
        
        return predictions
    
    def save_model(self, model_id: str, path: str) -> bool:
        """Save a trained model."""
        if model_id not in self.models:
            logger.error(f"Model {model_id} not found")
            return False
        
        return self.models[model_id].save_model(path)
    
    def load_model(self, model_id: str, path: str, config: QuantumMLConfig) -> bool:
        """Load a pre-trained model."""
        try:
            # Create model instance
            model = await self.create_model(model_id, config)
            
            # Load parameters
            success = model.load_model(path)
            
            if success:
                logger.info(f"Loaded quantum ML model {model_id} from {path}")
            
            return success
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return False
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a model."""
        if model_id not in self.models:
            return None
        
        model = self.models[model_id]
        
        return {
            'model_id': model_id,
            'algorithm': model.config.algorithm.value,
            'n_qubits': model.config.n_qubits,
            'encoding_type': model.config.encoding_type.value,
            'is_trained': model.is_trained,
            'parameter_count': len(model.parameters) if model.parameters is not None else 0,
            'training_iterations': len(model.training_history)
        }
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all models in the pipeline."""
        return [self.get_model_info(model_id) for model_id in self.models.keys()]
    
    def remove_model(self, model_id: str) -> bool:
        """Remove a model from the pipeline."""
        if model_id in self.models:
            del self.models[model_id]
            logger.info(f"Removed quantum ML model {model_id}")
            return True
        return False
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        total_models = len(self.models)
        trained_models = sum(1 for model in self.models.values() if model.is_trained)
        
        algorithm_counts = {}
        for model in self.models.values():
            algo = model.config.algorithm.value
            algorithm_counts[algo] = algorithm_counts.get(algo, 0) + 1
        
        return {
            'total_models': total_models,
            'trained_models': trained_models,
            'untrained_models': total_models - trained_models,
            'algorithm_distribution': algorithm_counts,
            'cache_stats': ml_cache.get_stats()
        }

# Global quantum ML pipeline instance
quantum_ml_pipeline = QuantumMLPipeline()

# Convenience functions
async def create_vqc_model(model_id: str, n_qubits: int = 4, 
                          encoding_type: DataEncodingType = DataEncodingType.ANGLE) -> VariationalQuantumClassifier:
    """Create a Variational Quantum Classifier."""
    config = QuantumMLConfig(
        algorithm=QuantumMLAlgorithm.VQC,
        n_qubits=n_qubits,
        encoding_type=encoding_type
    )
    return await quantum_ml_pipeline.create_model(model_id, config)

async def create_qnn_model(model_id: str, n_qubits: int = 4,
                          encoding_type: DataEncodingType = DataEncodingType.AMPLITUDE) -> QuantumNeuralNetwork:
    """Create a Quantum Neural Network."""
    config = QuantumMLConfig(
        algorithm=QuantumMLAlgorithm.QNN,
        n_qubits=n_qubits,
        encoding_type=encoding_type,
        batch_size=16
    )
    return await quantum_ml_pipeline.create_model(model_id, config)

def get_ml_pipeline_status() -> Dict[str, Any]:
    """Get comprehensive ML pipeline status."""
    return quantum_ml_pipeline.get_pipeline_stats()