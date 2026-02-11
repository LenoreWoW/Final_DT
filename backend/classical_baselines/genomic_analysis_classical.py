"""
Genomic Analysis Classical Baseline

PCA + Random Forest for gene expression analysis and disease prediction.
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter


@dataclass
class DecisionTree:
    """Simple decision tree for random forest"""
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['DecisionTree'] = None
    right: Optional['DecisionTree'] = None
    value: Optional[int] = None  # For leaf nodes


class PCA:
    """Principal Component Analysis"""

    def __init__(self, n_components: int):
        """
        Initialize PCA

        Args:
            n_components: Number of principal components to keep
        """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, X: np.ndarray):
        """
        Fit PCA to data

        Args:
            X: Data matrix (n_samples, n_features)
        """
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort by eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Keep top n_components
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to principal component space

        Args:
            X: Data matrix (n_samples, n_features)

        Returns:
            Transformed data (n_samples, n_components)
        """
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(X)
        return self.transform(X)


class RandomForest:
    """Random Forest Classifier"""

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 10,
                 min_samples_split: int = 2,
                 random_state: Optional[int] = None):
        """
        Initialize Random Forest

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split node
            random_state: Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []

        if random_state is not None:
            np.random.seed(random_state)

    def _bootstrap_sample(self,
                         X: np.ndarray,
                         y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create bootstrap sample"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def _build_tree(self,
                   X: np.ndarray,
                   y: np.ndarray,
                   depth: int = 0) -> DecisionTree:
        """
        Build a single decision tree recursively

        Args:
            X: Features
            y: Labels
            depth: Current depth

        Returns:
            DecisionTree node
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stopping criteria
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_classes == 1):
            # Create leaf node
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionTree(value=leaf_value)

        # Find best split
        best_gain = -1
        best_feature = None
        best_threshold = None

        # Random feature subset
        n_features_to_try = max(1, int(np.sqrt(n_features)))
        feature_indices = np.random.choice(
            n_features,
            size=n_features_to_try,
            replace=False
        )

        for feature_idx in feature_indices:
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                # Split data
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # Calculate information gain
                gain = self._information_gain(y, y[left_mask], y[right_mask])

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        # If no good split found, create leaf
        if best_feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionTree(value=leaf_value)

        # Create split
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Recursively build subtrees
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return DecisionTree(
            feature_idx=best_feature,
            threshold=best_threshold,
            left=left_tree,
            right=right_tree
        )

    def _information_gain(self,
                         parent: np.ndarray,
                         left: np.ndarray,
                         right: np.ndarray) -> float:
        """Calculate information gain of a split"""
        def entropy(y):
            counts = Counter(y)
            probs = np.array(list(counts.values())) / len(y)
            return -np.sum(probs * np.log2(probs + 1e-10))

        parent_entropy = entropy(parent)
        n = len(parent)
        n_left, n_right = len(left), len(right)

        if n_left == 0 or n_right == 0:
            return 0

        child_entropy = (n_left / n) * entropy(left) + (n_right / n) * entropy(right)
        return parent_entropy - child_entropy

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit random forest

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        """
        self.trees = []

        for _ in range(self.n_estimators):
            # Bootstrap sample
            X_sample, y_sample = self._bootstrap_sample(X, y)

            # Build tree
            tree = self._build_tree(X_sample, y_sample)
            self.trees.append(tree)

    def _predict_tree(self, tree: DecisionTree, x: np.ndarray) -> int:
        """Predict using single tree"""
        if tree.value is not None:
            return tree.value

        if x[tree.feature_idx] <= tree.threshold:
            return self._predict_tree(tree.left, x)
        else:
            return self._predict_tree(tree.right, x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels

        Args:
            X: Test features (n_samples, n_features)

        Returns:
            Predicted labels (n_samples,)
        """
        predictions = []

        for x in X:
            # Get prediction from each tree
            tree_predictions = [
                self._predict_tree(tree, x)
                for tree in self.trees
            ]

            # Majority vote
            prediction = Counter(tree_predictions).most_common(1)[0][0]
            predictions.append(prediction)

        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities

        Args:
            X: Test features (n_samples, n_features)

        Returns:
            Class probabilities (n_samples, n_classes)
        """
        n_samples = X.shape[0]
        n_classes = 2  # Binary classification

        probabilities = np.zeros((n_samples, n_classes))

        for i, x in enumerate(X):
            # Get predictions from all trees
            tree_predictions = [
                self._predict_tree(tree, x)
                for tree in self.trees
            ]

            # Calculate probabilities
            counts = Counter(tree_predictions)
            for class_label, count in counts.items():
                probabilities[i, class_label] = count / self.n_estimators

        return probabilities


class GenomicAnalysisClassical:
    """Classical genomic analysis pipeline"""

    def __init__(self,
                 n_components: int = 50,
                 n_estimators: int = 100):
        """
        Initialize genomic analysis system

        Args:
            n_components: Number of PCA components
            n_estimators: Number of trees in random forest
        """
        self.n_components = n_components
        self.n_estimators = n_estimators

        self.pca = PCA(n_components=n_components)
        self.rf = RandomForest(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=2
        )

    def analyze_gene_expression(self,
                                X_train: np.ndarray,
                                y_train: np.ndarray,
                                X_test: np.ndarray,
                                y_test: np.ndarray) -> Dict:
        """
        Analyze gene expression data

        Args:
            X_train: Training gene expression data (n_samples, n_genes)
            y_train: Training labels (0=healthy, 1=disease)
            X_test: Test gene expression data
            y_test: Test labels

        Returns:
            Analysis results
        """
        start_time = time.time()

        # PCA for dimensionality reduction
        X_train_pca = self.pca.fit_transform(X_train)
        X_test_pca = self.pca.transform(X_test)

        # Train random forest
        self.rf.fit(X_train_pca, y_train)

        # Predict
        y_pred = self.rf.predict(X_test_pca)
        y_proba = self.rf.predict_proba(X_test_pca)

        elapsed_time = time.time() - start_time

        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)

        true_positives = np.sum((y_pred == 1) & (y_test == 1))
        false_positives = np.sum((y_pred == 1) & (y_test == 0))
        true_negatives = np.sum((y_pred == 0) & (y_test == 0))
        false_negatives = np.sum((y_pred == 0) & (y_test == 1))

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Feature importance (simplified - based on PCA variance)
        explained_variance_ratio = self.pca.explained_variance / np.sum(self.pca.explained_variance)

        return {
            'training_time': elapsed_time,
            'n_genes_original': X_train.shape[1],
            'n_components_used': self.n_components,
            'explained_variance_ratio': float(np.sum(explained_variance_ratio)),
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1_score * 100,
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'true_negatives': int(true_negatives),
            'false_negatives': int(false_negatives),
            'method': 'PCA + Random Forest'
        }


def generate_synthetic_gene_expression(n_samples: int = 200,
                                      n_genes: int = 1000,
                                      n_informative: int = 100,
                                      seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic gene expression data

    Args:
        n_samples: Number of samples
        n_genes: Number of genes
        n_informative: Number of informative genes
        seed: Random seed

    Returns:
        X, y where X is gene expression and y is labels
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate labels (balanced)
    y = np.array([0, 1] * (n_samples // 2))
    if n_samples % 2 == 1:
        y = np.append(y, 0)

    # Generate gene expression
    X = np.random.randn(n_samples, n_genes)

    # Make some genes informative
    for i in range(n_informative):
        # These genes are correlated with disease status
        signal_strength = np.random.uniform(1.0, 3.0)
        X[:, i] += y * signal_strength + np.random.randn(n_samples) * 0.5

    # Normalize
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

    return X, y


def run_genomic_analysis_classical(n_genes: int = 1000,
                                  n_samples: int = 200) -> Dict:
    """
    Run classical genomic analysis benchmark

    Args:
        n_genes: Number of genes to analyze
        n_samples: Number of samples

    Returns:
        Benchmark results
    """
    # Generate synthetic data
    X, y = generate_synthetic_gene_expression(
        n_samples=n_samples,
        n_genes=n_genes,
        n_informative=min(100, n_genes // 10),
        seed=42
    )

    # Split into train/test
    split_idx = int(0.7 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Run analysis
    analyzer = GenomicAnalysisClassical(
        n_components=min(50, n_genes // 20),
        n_estimators=100
    )

    results = analyzer.analyze_gene_expression(
        X_train, y_train,
        X_test, y_test
    )

    return results


if __name__ == '__main__':
    # Test the classical genomic analysis
    print("Testing Classical Genomic Analysis...")
    print("=" * 60)

    results = run_genomic_analysis_classical(n_genes=1000, n_samples=200)

    print(f"\nGenes Analyzed: {results['n_genes_original']}")
    print(f"PCA Components: {results['n_components_used']}")
    print(f"Explained Variance: {results['explained_variance_ratio']:.1%}")
    print(f"Training Time: {results['training_time']:.2f} seconds")

    print(f"\nAccuracy: {results['accuracy']:.1f}%")
    print(f"Precision: {results['precision']:.1f}%")
    print(f"Recall: {results['recall']:.1f}%")
    print(f"F1 Score: {results['f1_score']:.1f}%")

    print(f"\nConfusion Matrix:")
    print(f"  TP: {results['true_positives']}, FP: {results['false_positives']}")
    print(f"  FN: {results['false_negatives']}, TN: {results['true_negatives']}")
