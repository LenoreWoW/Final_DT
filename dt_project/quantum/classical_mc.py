"""
Classical Monte Carlo simulation module.

This module provides classical Monte Carlo simulation capabilities for comparison
with quantum Monte Carlo methods.
"""

import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

class ClassicalMonteCarlo:
    """Classical Monte Carlo simulation for integration and optimization tasks."""
    
    def __init__(self, config=None):
        """Initialize the classical Monte Carlo simulator.
        
        Args:
            config: Optional configuration manager instance
        """
        self.config = config
        self.iterations = 1000
        self.dimensions = 2
        self.distribution = "uniform"
        self.available_distributions = {
            "uniform": self._uniform_distribution,
            "normal": self._normal_distribution,
            "exponential": self._exponential_distribution,
            "beta": self._beta_distribution
        }
        logger.info("Classical Monte Carlo simulator initialized")
    
    def is_available(self):
        """Check if classical Monte Carlo is available.
        
        Returns:
            bool: Always True since classical methods are always available
        """
        return True
    
    def _uniform_distribution(self, param_ranges, samples):
        """Generate uniform random samples within parameter ranges.
        
        Args:
            param_ranges (dict): Dictionary mapping parameter names to (min, max) tuples
            samples (int): Number of samples to generate
            
        Returns:
            dict: Dictionary mapping parameter names to arrays of random values
        """
        result = {}
        for param, (min_val, max_val) in param_ranges.items():
            result[param] = np.random.uniform(min_val, max_val, samples)
        return result
    
    def _normal_distribution(self, param_ranges, samples):
        """Generate normally distributed random samples within parameter ranges.
        
        Args:
            param_ranges (dict): Dictionary mapping parameter names to (mean, std) tuples
            samples (int): Number of samples to generate
            
        Returns:
            dict: Dictionary mapping parameter names to arrays of random values
        """
        result = {}
        for param, (mean, std) in param_ranges.items():
            result[param] = np.random.normal(mean, std, samples)
        return result
    
    def _exponential_distribution(self, param_ranges, samples):
        """Generate exponentially distributed random samples.
        
        Args:
            param_ranges (dict): Dictionary mapping parameter names to (scale) tuples
            samples (int): Number of samples to generate
            
        Returns:
            dict: Dictionary mapping parameter names to arrays of random values
        """
        result = {}
        for param, (scale,) in param_ranges.items():
            result[param] = np.random.exponential(scale, samples)
        return result
    
    def _beta_distribution(self, param_ranges, samples):
        """Generate beta distributed random samples.
        
        Args:
            param_ranges (dict): Dictionary mapping parameter names to (alpha, beta) tuples
            samples (int): Number of samples to generate
            
        Returns:
            dict: Dictionary mapping parameter names to arrays of random values
        """
        result = {}
        for param, (alpha, beta) in param_ranges.items():
            result[param] = np.random.beta(alpha, beta, samples)
        return result
    
    def integrate(self, target_function, param_ranges, distribution="uniform", iterations=None):
        """Perform Monte Carlo integration on the target function.
        
        Args:
            target_function (callable): Function to integrate
            param_ranges (dict): Dictionary mapping parameter names to ranges
            distribution (str): Distribution to use for sampling
            iterations (int, optional): Number of iterations to use
            
        Returns:
            dict: Dictionary containing mean, standard error, and other statistics
        """
        if iterations is None:
            iterations = self.iterations
        
        if distribution not in self.available_distributions:
            raise ValueError(f"Distribution '{distribution}' not supported. Available: {list(self.available_distributions.keys())}")
        
        # Start time measurement
        start_time = time.time()
        
        # Generate random samples
        samples = self.available_distributions[distribution](param_ranges, iterations)
        
        # Compute function values
        param_names = list(samples.keys())
        function_values = np.zeros(iterations)
        
        for i in range(iterations):
            args = [samples[param][i] for param in param_names]
            
            try:
                if len(args) == 1:
                    function_values[i] = target_function(args[0])
                else:
                    function_values[i] = target_function(*args)
            except Exception as e:
                logger.error(f"Error evaluating function at {args}: {str(e)}")
                function_values[i] = np.nan
        
        # Filter out NaN values
        valid_indices = ~np.isnan(function_values)
        valid_function_values = function_values[valid_indices]
        
        # If all values are NaN, return NaN results
        if len(valid_function_values) == 0:
            return {
                "mean": np.nan,
                "std": np.nan,
                "sem": np.nan,
                "confidence_interval": (np.nan, np.nan),
                "valid_samples": 0,
                "total_samples": iterations,
                "execution_time": time.time() - start_time
            }
        
        # Compute statistics
        mean = np.mean(valid_function_values)
        std = np.std(valid_function_values)
        sem = std / np.sqrt(len(valid_function_values))
        
        # Compute confidence interval (95%)
        confidence_interval = (mean - 1.96 * sem, mean + 1.96 * sem)
        
        # End time measurement
        execution_time = time.time() - start_time
        
        return {
            "mean": mean,
            "std": std,
            "sem": sem,
            "confidence_interval": confidence_interval,
            "valid_samples": len(valid_function_values),
            "total_samples": iterations,
            "execution_time": execution_time
        }
    
    def run_classical_monte_carlo(self, param_ranges, iterations, target_function, distribution="uniform"):
        """Run a classical Monte Carlo simulation.
        
        Args:
            param_ranges (dict): Dictionary mapping parameter names to ranges
            iterations (int): Number of iterations to use
            target_function (callable): Function to evaluate
            distribution (str): Distribution to use for sampling
            
        Returns:
            dict: Dictionary containing mean, standard error, and other statistics
        """
        return self.integrate(target_function, param_ranges, distribution, iterations) 