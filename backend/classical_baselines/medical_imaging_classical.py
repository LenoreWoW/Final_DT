"""
Medical Imaging Classical Baseline

CNN-based tumor detection in medical images.
Uses a simplified ResNet-like architecture for classification.
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ConvLayer:
    """Convolutional layer"""
    filters: np.ndarray  # (out_channels, in_channels, kernel_h, kernel_w)
    bias: np.ndarray  # (out_channels,)
    stride: int = 1
    padding: int = 0


@dataclass
class FCLayer:
    """Fully connected layer"""
    weights: np.ndarray  # (out_features, in_features)
    bias: np.ndarray  # (out_features,)


class ClassicalCNN:
    """Classical Convolutional Neural Network for medical imaging"""

    def __init__(self, input_shape: Tuple[int, int, int] = (64, 64, 1)):
        """
        Initialize CNN

        Args:
            input_shape: (height, width, channels)
        """
        self.input_shape = input_shape
        self.layers = []
        self.trained = False

        # Initialize network architecture (simplified ResNet-like)
        self._build_network()

    def _build_network(self):
        """Build CNN architecture"""
        rng = np.random.RandomState(42)

        # Conv1: 1 -> 32 channels
        self.conv1 = ConvLayer(
            filters=rng.randn(32, self.input_shape[2], 3, 3) * 0.01,
            bias=np.zeros(32),
            stride=1,
            padding=1
        )

        # Conv2: 32 -> 64 channels
        self.conv2 = ConvLayer(
            filters=rng.randn(64, 32, 3, 3) * 0.01,
            bias=np.zeros(64),
            stride=1,
            padding=1
        )

        # Conv3: 64 -> 128 channels
        self.conv3 = ConvLayer(
            filters=rng.randn(128, 64, 3, 3) * 0.01,
            bias=np.zeros(128),
            stride=1,
            padding=1
        )

        # Fully connected layers
        # After 3 max pools (2x2), spatial dim reduces by 8x
        h_out = self.input_shape[0] // 8
        w_out = self.input_shape[1] // 8
        fc_input_size = 128 * h_out * w_out

        self.fc1 = FCLayer(
            weights=rng.randn(256, fc_input_size) * 0.01,
            bias=np.zeros(256)
        )

        self.fc2 = FCLayer(
            weights=rng.randn(2, 256) * 0.01,  # Binary classification
            bias=np.zeros(2)
        )

    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation"""
        return np.maximum(0, x)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def conv2d(self, input_data: np.ndarray, layer: ConvLayer) -> np.ndarray:
        """2D convolution operation"""
        batch_size, in_channels, in_h, in_w = input_data.shape
        out_channels, _, kernel_h, kernel_w = layer.filters.shape
        out_h = (in_h + 2 * layer.padding - kernel_h) // layer.stride + 1
        out_w = (in_w + 2 * layer.padding - kernel_w) // layer.stride + 1
        if layer.padding > 0:
            padded = np.pad(input_data, ((0, 0), (0, 0), (layer.padding, layer.padding), (layer.padding, layer.padding)), mode='constant')
        else:
            padded = input_data
        output = np.zeros((batch_size, out_channels, out_h, out_w))
        for b in range(batch_size):
            for oc in range(out_channels):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * layer.stride
                        w_start = w * layer.stride
                        receptive_field = padded[b, :, h_start:h_start + kernel_h, w_start:w_start + kernel_w]
                        output[b, oc, h, w] = np.sum(receptive_field * layer.filters[oc]) + layer.bias[oc]
        return output

    def max_pool2d(self, input_data: np.ndarray, pool_size: int = 2, stride: int = 2) -> np.ndarray:
        """Max pooling operation"""
        batch_size, channels, in_h, in_w = input_data.shape
        out_h = (in_h - pool_size) // stride + 1
        out_w = (in_w - pool_size) // stride + 1
        output = np.zeros((batch_size, channels, out_h, out_w))
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * stride
                        w_start = w * stride
                        pool_region = input_data[b, c, h_start:h_start + pool_size, w_start:w_start + pool_size]
                        output[b, c, h, w] = np.max(pool_region)
        return output

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""
        if not self.trained:
            raise RuntimeError(
                "CNN has not been trained. Call _simulate_training() or "
                "train the network before running predictions."
            )
        x = np.transpose(x, (0, 3, 1, 2))
        x = self.conv2d(x, self.conv1)
        x = self.relu(x)
        x = self.max_pool2d(x, pool_size=2, stride=2)
        x = self.conv2d(x, self.conv2)
        x = self.relu(x)
        x = self.max_pool2d(x, pool_size=2, stride=2)
        x = self.conv2d(x, self.conv3)
        x = self.relu(x)
        x = self.max_pool2d(x, pool_size=2, stride=2)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = np.dot(x, self.fc1.weights.T) + self.fc1.bias
        x = self.relu(x)
        x = np.dot(x, self.fc2.weights.T) + self.fc2.bias
        probs = np.array([self.softmax(x[i]) for i in range(batch_size)])
        return probs


class MedicalImagingClassical:
    """Classical medical imaging tumor detection"""

    def __init__(self, image_size: Tuple[int, int] = (64, 64), confidence_threshold: float = 0.7):
        """Initialize classical medical imaging system"""
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.cnn = ClassicalCNN(input_shape=(image_size[0], image_size[1], 1))
        self._simulate_training()

    def _simulate_training(self):
        """He-initialize all layers, then train fc1/fc2 on synthetic data."""
        rng = np.random.RandomState(123)

        # He initialization for all layers
        fan_in_1 = self.cnn.conv1.filters.shape[1] * self.cnn.conv1.filters.shape[2] * self.cnn.conv1.filters.shape[3]
        self.cnn.conv1.filters = rng.randn(*self.cnn.conv1.filters.shape) * np.sqrt(2.0 / fan_in_1)
        fan_in_2 = self.cnn.conv2.filters.shape[1] * self.cnn.conv2.filters.shape[2] * self.cnn.conv2.filters.shape[3]
        self.cnn.conv2.filters = rng.randn(*self.cnn.conv2.filters.shape) * np.sqrt(2.0 / fan_in_2)
        fan_in_3 = self.cnn.conv3.filters.shape[1] * self.cnn.conv3.filters.shape[2] * self.cnn.conv3.filters.shape[3]
        self.cnn.conv3.filters = rng.randn(*self.cnn.conv3.filters.shape) * np.sqrt(2.0 / fan_in_3)
        fan_in_fc1 = self.cnn.fc1.weights.shape[1]
        self.cnn.fc1.weights = rng.randn(*self.cnn.fc1.weights.shape) * np.sqrt(2.0 / fan_in_fc1)
        fan_in_fc2 = self.cnn.fc2.weights.shape[1]
        self.cnn.fc2.weights = rng.randn(*self.cnn.fc2.weights.shape) * np.sqrt(2.0 / fan_in_fc2)
        self.cnn.trained = True

        # Train fc1 and fc2 on synthetic labeled images (conv layers frozen)
        n_train = 50
        lr = 0.01
        n_epochs = 10

        # Generate labeled training data
        train_images = []
        train_labels = []
        for idx in range(n_train):
            has_tumor = (idx % 2 == 0)
            img = generate_synthetic_medical_image(
                has_tumor=has_tumor, size=self.image_size, seed=1000 + idx
            )
            preprocessed = self.preprocess_image(img)
            train_images.append(preprocessed)
            train_labels.append(1 if has_tumor else 0)

        # Forward through frozen conv layers to get features
        features = []
        for img_batch in train_images:
            x = np.transpose(img_batch, (0, 3, 1, 2))
            x = self.cnn.conv2d(x, self.cnn.conv1)
            x = self.cnn.relu(x)
            x = self.cnn.max_pool2d(x, pool_size=2, stride=2)
            x = self.cnn.conv2d(x, self.cnn.conv2)
            x = self.cnn.relu(x)
            x = self.cnn.max_pool2d(x, pool_size=2, stride=2)
            x = self.cnn.conv2d(x, self.cnn.conv3)
            x = self.cnn.relu(x)
            x = self.cnn.max_pool2d(x, pool_size=2, stride=2)
            x = x.reshape(1, -1)
            features.append(x[0])
        features = np.array(features)  # (n_train, fc_input_size)

        # SGD training loop on fc1 and fc2 only
        for epoch in range(n_epochs):
            for i in range(n_train):
                feat = features[i]  # (fc_input_size,)
                label = train_labels[i]

                # Forward: fc1
                z1 = feat @ self.cnn.fc1.weights.T + self.cnn.fc1.bias  # (256,)
                a1 = np.maximum(0, z1)  # ReLU

                # Forward: fc2
                z2 = a1 @ self.cnn.fc2.weights.T + self.cnn.fc2.bias  # (2,)
                exp_z = np.exp(z2 - np.max(z2))
                probs = exp_z / np.sum(exp_z)  # softmax

                # Cross-entropy gradient: dL/dz2 = probs - one_hot
                one_hot = np.zeros(2)
                one_hot[label] = 1.0
                dz2 = probs - one_hot  # (2,)

                # Gradients for fc2
                dW2 = np.outer(dz2, a1)  # (2, 256)
                db2 = dz2  # (2,)

                # Backprop through ReLU to fc1
                da1 = self.cnn.fc2.weights.T @ dz2  # (256,)
                dz1 = da1 * (z1 > 0).astype(float)  # (256,)
                dW1 = np.outer(dz1, feat)  # (256, fc_input_size)
                db1 = dz1  # (256,)

                # SGD update
                self.cnn.fc2.weights -= lr * dW2
                self.cnn.fc2.bias -= lr * db2
                self.cnn.fc1.weights -= lr * dW1
                self.cnn.fc1.bias -= lr * db1

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess medical image"""
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)
        h, w = image.shape
        target_h, target_w = self.image_size
        if h != target_h or w != target_w:
            if h > target_h:
                start_h = (h - target_h) // 2
                image = image[start_h:start_h + target_h, :]
            elif h < target_h:
                pad_h = (target_h - h) // 2
                image = np.pad(image, ((pad_h, target_h - h - pad_h), (0, 0)))
            if w > target_w:
                start_w = (w - target_w) // 2
                image = image[:, start_w:start_w + target_w]
            elif w < target_w:
                pad_w = (target_w - w) // 2
                image = np.pad(image, ((0, 0), (pad_w, target_w - w - pad_w)))
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        image = image.reshape(1, target_h, target_w, 1)
        return image

    def detect_tumor(self, image: np.ndarray) -> Dict:
        """Detect tumor in medical image"""
        preprocessed = self.preprocess_image(image)
        probs = self.cnn.forward(preprocessed)[0]
        tumor_confidence = probs[1]
        detected = tumor_confidence >= self.confidence_threshold
        return {
            'tumor_detected': bool(detected),
            'confidence': float(tumor_confidence),
            'normal_confidence': float(probs[0]),
            'tumor_confidence': float(probs[1])
        }

    def batch_detect(self, images: List[np.ndarray]) -> List[Dict]:
        """Detect tumors in batch of images"""
        results = []
        for image in images:
            result = self.detect_tumor(image)
            results.append(result)
        return results


def generate_synthetic_medical_image(has_tumor: bool = False, size: Tuple[int, int] = (64, 64), seed: Optional[int] = None) -> np.ndarray:
    """Generate synthetic medical image for testing"""
    rng = np.random.RandomState(seed)
    h, w = size
    image = rng.randn(h, w) * 0.1 + 0.5
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)
    organ = np.exp(-(xx**2 + yy**2) / 0.5)
    image += organ * 0.3
    if has_tumor:
        tumor_x = rng.uniform(-0.5, 0.5)
        tumor_y = rng.uniform(-0.5, 0.5)
        tumor_size = rng.uniform(0.1, 0.3)
        tumor = np.exp(-((xx - tumor_x)**2 + (yy - tumor_y)**2) / tumor_size)
        image += tumor * 0.5
        noise = rng.randn(h, w) * 0.1
        image += noise * tumor
    image = np.clip(image, 0, 1)
    return image


def run_medical_imaging_classical(num_images: int = 100) -> Dict:
    """Run classical medical imaging benchmark"""
    imaging = MedicalImagingClassical(image_size=(64, 64), confidence_threshold=0.7)
    start_time = time.time()
    images = []
    ground_truth = []
    for i in range(num_images):
        has_tumor = (i % 2 == 0)
        image = generate_synthetic_medical_image(has_tumor=has_tumor, size=(64, 64), seed=i)
        images.append(image)
        ground_truth.append(has_tumor)
    results = imaging.batch_detect(images)
    elapsed_time = time.time() - start_time
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    for i, result in enumerate(results):
        predicted = result['tumor_detected']
        actual = ground_truth[i]
        if predicted and actual:
            true_positives += 1
        elif predicted and not actual:
            false_positives += 1
        elif not predicted and not actual:
            true_negatives += 1
        else:
            false_negatives += 1
    accuracy = (true_positives + true_negatives) / num_images
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    sensitivity = recall
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    return {
        'num_images': num_images,
        'processing_time': elapsed_time,
        'throughput': num_images / elapsed_time,
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1_score * 100,
        'sensitivity': sensitivity * 100,
        'specificity': specificity * 100,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        'method': 'Classical CNN (ResNet-like)'
    }


if __name__ == '__main__':
    print("Testing Classical Medical Imaging...")
    print("=" * 60)
    results = run_medical_imaging_classical(num_images=100)
    print(f"\nImages Processed: {results['num_images']}")
    print(f"Processing Time: {results['processing_time']:.2f} seconds")
    print(f"Throughput: {results['throughput']:.1f} images/second")
    print(f"\nAccuracy: {results['accuracy']:.1f}%")
    print(f"Precision: {results['precision']:.1f}%")
    print(f"Recall: {results['recall']:.1f}%")
    print(f"F1 Score: {results['f1_score']:.1f}%")
    print(f"Sensitivity: {results['sensitivity']:.1f}%")
    print(f"Specificity: {results['specificity']:.1f}%")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {results['true_positives']}, FP: {results['false_positives']}")
    print(f"  FN: {results['false_negatives']}, TN: {results['true_negatives']}")
