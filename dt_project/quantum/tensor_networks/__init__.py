"""
Tensor Network Module for Quantum Digital Twins

This module implements tensor network representations for quantum digital twins,
targeting CERN-level fidelity benchmarks (99.9%) through sophisticated
mathematical frameworks.

Key Components:
- Matrix Product Operators (MPO) for quantum system representation
- Tree Tensor Networks (TTN) for hierarchical representations
- Tensor network simulation with high fidelity
- Fidelity optimization algorithms
- Integration with existing quantum digital twin architecture
"""

from .matrix_product_operator import MatrixProductOperator

try:
    from .tree_tensor_network import TreeTensorNetwork
except (ImportError, AttributeError) as e:
    import logging
    logging.debug(f"Tree tensor network not available: {e}")
    TreeTensorNetwork = None

__all__ = [
    'MatrixProductOperator',
    'TreeTensorNetwork',
]
