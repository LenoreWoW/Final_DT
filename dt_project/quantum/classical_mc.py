"""
Classical Monte Carlo Module for comparison with quantum methods.
"""

import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Callable, Union, Any

from dt_project.config import ConfigManager

logger = logging.getLogger(__name__)

class ClassicalMonteCarlo:
    """
    Classical Monte Carlo simulation for comparison with quantum methods.
    """
    
    def __init__(self, config=None):
        """
        Initialize classical Monte Carlo simulator.
        
        Args:
            config: Configuration manager (optional)
        """
        self.config = config or ConfigManager()
    
    def integrate_2d(self, target_function, x_min, x_max, y_min, y_max, 
                    iterations=1000, distribution="uniform"):
        """
        Perform classical Monte Carlo integration in 2D.
        
        Args:
            target_function: Function to integrate
            x_min, x_max: Bounds for x
            y_min, y_max: Bounds for y
            iterations: Number of random samples to use
            distribution: Type of distribution to use
            
        Returns:
            Dictionary with results including mean and standard error
        """
        start_time = time.time()
        
        # Generate random points according to distribution
        if distribution == "uniform":
            x_samples = np.random.uniform(x_min, x_max, iterations)
            y_samples = np.random.uniform(y_min, y_max, iterations)
        elif distribution == "normal":
            # For normal distribution, we need to scale appropriately
            x_center = (x_max + x_min) / 2
            y_center = (y_max + y_min) / 2
            x_scale = (x_max - x_min) / 6  # 3 sigma on each side
            y_scale = (y_max - y_min) / 6
            
            x_samples = np.random.normal(x_center, x_scale, iterations)
            y_samples = np.random.normal(y_center, y_scale, iterations)
            
            # Clip to bounds
            x_samples = np.clip(x_samples, x_min, x_max)
            y_samples = np.clip(y_samples, y_min, y_max)
        else:
            # Default to uniform
            x_samples = np.random.uniform(x_min, x_max, iterations)
            y_samples = np.random.uniform(y_min, y_max, iterations)
        
        # Evaluate function at each point
        values = np.array([target_function(x, y) for x, y in zip(x_samples, y_samples)])
        
        # Calculate the area of the integration region
        area = (x_max - x_min) * (y_max - y_min)
        
        # Scale the results by the area
        scaled_values = values * area
        
        # Calculate statistics
        mean = np.mean(scaled_values)
        std = np.std(scaled_values)
        std_error = std / np.sqrt(iterations)
        
        execution_time = time.time() - start_time
        
        return {
            'mean': mean,
            'std': std,
            'std_error': std_error,
            'min': np.min(scaled_values),
            'max': np.max(scaled_values),
            'execution_time': execution_time
        }
    
    def run_classical_monte_carlo(self, param_ranges, target_function, iterations=1000, distribution="uniform"):
        """
        Run classical Monte Carlo simulation for parameter sampling.
        Compatible with the QuantumMonteCarlo interface.
        
        Args:
            param_ranges: Dictionary mapping parameter names to (min, max) tuples
            target_function: Function to evaluate at each sampled point
            iterations: Number of iterations to run
            distribution: Type of distribution to use
            
        Returns:
            Dictionary with results and statistics
        """
        start_time = time.time()
        
        param_names = list(param_ranges.keys())
        param_samples = {name: [] for name in param_names}
        values = []
        
        for _ in range(iterations):
            params = {}
            for param_name in param_names:
                min_val, max_val = param_ranges[param_name]
                if distribution == "uniform":
                    param_value = np.random.uniform(min_val, max_val)
                elif distribution == "normal":
                    center = (min_val + max_val) / 2
                    scale = (max_val - min_val) / 6
                    param_value = np.random.normal(center, scale)
                    param_value = np.clip(param_value, min_val, max_val)
                else:
                    # Default to uniform
                    param_value = np.random.uniform(min_val, max_val)
                    
                params[param_name] = param_value
                param_samples[param_name].append(param_value)
            
            # Evaluate target function if provided
            if target_function:
                result_value = target_function(**params)
                values.append(result_value)
        
        results = {
            'param_samples': param_samples,
            'execution_time': time.time() - start_time,
            'backend': 'classical',
            'quantum': False
        }
        
        if target_function:
            results['values'] = values
            results['mean'] = np.mean(values)
            results['std'] = np.std(values)
            results['min'] = np.min(values)
            results['max'] = np.max(values)
        
        return results 