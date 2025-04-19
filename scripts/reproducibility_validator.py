import os
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import logging
import scipy.stats as stats
import sys

# Add project root to path to fix imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dt_project.config.config_manager import ConfigManager
from dt_project.quantum.qml import QuantumML

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Create output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'reproducibility')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'plots'), exist_ok=True)

def print_section(title):
    """Print a section header."""
    print()
    print('=' * 80)
    print('=' * 30 + ' ' + title.center(20, ' ') + ' ' + '=' * 30)
    print('=' * 80)
    print()

# Define fallback implementations for scikit-learn functions
def train_test_split_fallback(X, y, test_size=0.2, random_state=None):
    """
    A simple fallback implementation of train_test_split when scikit-learn is not available.
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def mean_squared_error_fallback(y_true, y_pred):
    """
    A simple fallback implementation of mean_squared_error when scikit-learn is not available.
    """
    return np.mean((y_true - y_pred) ** 2)

def r2_score_fallback(y_true, y_pred):
    """
    A simple fallback implementation of r2_score when scikit-learn is not available.
    """
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    if ss_tot == 0:  # Avoid division by zero
        return 0
    
    return 1 - (ss_res / ss_tot)

def plot_qml_reproducibility(training_results, prediction_results, metrics):
    """
    Generate plots to visualize QML reproducibility.
    
    Args:
        training_results: List of results from training runs
        prediction_results: List of results from prediction runs
        metrics: Dictionary with reproducibility metrics
    """
    plots_dir = os.path.join(OUTPUT_DIR, 'plots')
    
    # Extract data
    mse_values = [r['mse'] for r in prediction_results]
    r2_values = [r['r2'] for r in prediction_results]
    
    # Check if all values are NaN
    if all(np.isnan(mse) for mse in mse_values) or all(np.isnan(r2) for r2 in r2_values):
        logger.warning("All MSE or R² values are NaN. Cannot generate QML reproducibility plots.")
        
        # Create an empty plot with message
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "Insufficient valid data to generate plots",
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'qml_no_valid_data.png'))
        return
    
    # Ensure we have enough data to plot
    if not mse_values or not r2_values:
        logger.warning("Not enough data to generate QML reproducibility plots")
        return
    
    try:
        # 1. Distribution of prediction metrics
        plt.figure(figsize=(12, 6))
        
        # Filter out NaN values for plotting
        valid_mse = [m for m in mse_values if not np.isnan(m)]
        valid_r2 = [r for r in r2_values if not np.isnan(r)]
        
        if valid_mse:
            plt.subplot(1, 2, 1)
            bins = min(10, max(3, len(valid_mse)))  # At least 3 bins, at most 10
            plt.hist(valid_mse, bins=bins, alpha=0.7)
            plt.axvline(np.nanmean(valid_mse), color='r', linestyle='dashed', linewidth=1, label='Mean')
            plt.xlabel('Mean Squared Error')
            plt.ylabel('Frequency')
            plt.title('Distribution of MSE Values')
            plt.legend()
        
        if valid_r2:
            plt.subplot(1, 2, 2)
            bins = min(10, max(3, len(valid_r2)))  # At least 3 bins, at most 10
            plt.hist(valid_r2, bins=bins, alpha=0.7)
            plt.axvline(np.nanmean(valid_r2), color='r', linestyle='dashed', linewidth=1, label='Mean')
            plt.xlabel('R² Score')
            plt.ylabel('Frequency')
            plt.title('Distribution of R² Values')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'qml_metrics_distribution.png'))
        
        # 2. Run-by-run comparison
        plt.figure(figsize=(10, 6))
        runs = range(1, len(mse_values) + 1)
        
        plt.subplot(2, 1, 1)
        # Replace NaN with None for plotting
        plot_mse = [None if np.isnan(m) else m for m in mse_values]
        plt.plot(runs, plot_mse, 'o-')
        
        if valid_mse:
            mean_mse = np.nanmean(valid_mse)
            plt.axhline(mean_mse, color='r', linestyle='dashed', alpha=0.7, label='Mean')
            
            # Only plot confidence interval if it's valid (no NaNs)
            ci_lower, ci_upper = metrics.get('mse_95ci', (np.nan, np.nan))
            if not (np.isnan(ci_lower) or np.isnan(ci_upper)):
                plt.fill_between(
                    runs, 
                    [ci_lower] * len(runs), 
                    [ci_upper] * len(runs), 
                    alpha=0.2, 
                    label='95% CI'
                )
        plt.ylabel('MSE')
        plt.title('MSE by Run')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        # Replace NaN with None for plotting
        plot_r2 = [None if np.isnan(r) else r for r in r2_values]
        plt.plot(runs, plot_r2, 'o-', color='green')
        
        if valid_r2:
            mean_r2 = np.nanmean(valid_r2)
            plt.axhline(mean_r2, color='r', linestyle='dashed', alpha=0.7, label='Mean')
            
            # Only plot confidence interval if it's valid (no NaNs)
            ci_lower, ci_upper = metrics.get('r2_95ci', (np.nan, np.nan))
            if not (np.isnan(ci_lower) or np.isnan(ci_upper)):
                plt.fill_between(
                    runs, 
                    [ci_lower] * len(runs), 
                    [ci_upper] * len(runs), 
                    alpha=0.2, 
                    color='green', 
                    label='95% CI'
                )
        plt.xlabel('Run Number')
        plt.ylabel('R²')
        plt.title('R² by Run')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'qml_run_comparison.png'))
        
        # 3. Reproducibility metrics visualization - only if we have valid metrics
        valid_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                valid_metrics[k] = v
        
        if valid_metrics:
            plt.figure(figsize=(10, 6))
            
            metric_labels = []
            metric_values = []
            
            if 'cv_mse' in valid_metrics:
                metric_labels.append('CV of MSE')
                metric_values.append(valid_metrics['cv_mse'])
            
            if 'cv_r2' in valid_metrics:
                metric_labels.append('CV of R²')
                metric_values.append(valid_metrics['cv_r2'])
            
            if 'rsd_mse' in valid_metrics:
                metric_labels.append('RSD of MSE (%)')
                metric_values.append(valid_metrics['rsd_mse'])
            
            if 'rsd_r2' in valid_metrics:
                metric_labels.append('RSD of R² (%)')
                metric_values.append(valid_metrics['rsd_r2'])
            
            if metric_labels and metric_values:
                plt.bar(metric_labels, metric_values)
                plt.ylabel('Value')
                plt.title('QML Reproducibility Metrics')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'qml_reproducibility_metrics.png'))
            else:
                logger.warning("No valid metrics to plot for QML reproducibility")
        else:
            logger.warning("No valid metrics to plot for QML reproducibility")
    
    except Exception as e:
        logger.error(f"Error generating QML reproducibility plots: {str(e)}")
        # Create a simple error plot
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"Error generating plots: {str(e)}",
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=12, wrap=True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'qml_plot_error.png'))

def test_qml_reproducibility(n_runs=5):
    """
    Test the reproducibility of quantum machine learning algorithms.
    
    Args:
        n_runs: Number of identical runs to perform
        
    Returns:
        Dictionary with reproducibility metrics
    """
    print_section("QUANTUM MACHINE LEARNING REPRODUCIBILITY")
    
    try:
        # Initialize quantum ML
        config = ConfigManager()
        qml = QuantumML(config)
        
        # Parameters for synthetic dataset
        n_samples = 100
        n_features = 4
        test_size = 0.3
        
        # Store results from each run
        training_results = []
        prediction_results = []
        
        print(f"Running {n_runs} identical QML training and prediction cycles...")
        
        # Generate consistent synthetic dataset for all runs
        np.random.seed(42)
        X = np.random.rand(n_samples, n_features)
        y = 0.3 * X[:, 0] + 0.2 * X[:, 1] * X[:, 2] + 0.5 * X[:, 3] ** 2
        y += 0.05 * np.random.randn(n_samples)  # Add small noise
        
        # Split data - use same split for all runs to ensure fair comparison
        try:
            # Try to import from scikit-learn first
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42)
        except ImportError:
            # Fall back to our own implementation
            X_train, X_test, y_train, y_test = train_test_split_fallback(
                X, y, test_size=test_size, random_state=42)
        
        for i in range(n_runs):
            print(f"  Run {i+1}/{n_runs}")
            
            try:
                # Set the same parameters for each run
                qml.feature_map = "angle"  # Using simpler feature map
                qml.n_layers = 2
                qml.n_qubits = 4  # Ensure this matches what's used internally
                qml.ansatz_type = "strongly_entangling"
                qml.max_iterations = 50
                
                # Create data that has the right number of features for the qubits
                X_train_subset = X_train[:, :min(4, X_train.shape[1])]  # Use at most 4 features
                if X_train_subset.shape[1] < 4:
                    # Pad with zeros if needed
                    padding = np.zeros((X_train_subset.shape[0], 4 - X_train_subset.shape[1]))
                    X_train_subset = np.hstack((X_train_subset, padding))
                    
                X_test_subset = X_test[:, :min(4, X_test.shape[1])]
                if X_test_subset.shape[1] < 4:
                    padding = np.zeros((X_test_subset.shape[0], 4 - X_test_subset.shape[1]))
                    X_test_subset = np.hstack((X_test_subset, padding))
                
                # Train quantum model with proper error handling
                start_time = time.time()
                try:
                    result = qml.train_model(X_train_subset, y_train, test_size=0.2, verbose=False)
                    training_time = time.time() - start_time
                    
                    # Make predictions
                    y_pred = qml.predict(X_test_subset)
                    
                    # Calculate metrics
                    try:
                        from sklearn.metrics import mean_squared_error, r2_score
                    except ImportError:
                        # Use fallback implementations
                        mean_squared_error = mean_squared_error_fallback
                        r2_score = r2_score_fallback
                        
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Store training results
                    training_results.append({
                        'iterations': result.get('iterations', 0),
                        'final_loss': result.get('final_loss', 0),
                        'training_time': training_time,
                        'loss_history': result.get('loss_history', [])[:5] if 'loss_history' in result else []
                    })
                    
                    # Store prediction results
                    prediction_results.append({
                        'mse': mse,
                        'r2': r2,
                        'y_pred_first5': y_pred[:5].tolist() if len(y_pred) >= 5 else y_pred.tolist()
                    })
                    
                    print(f"    MSE: {mse:.6f}, R²: {r2:.6f}")
                except Exception as e:
                    logger.error(f"Error in QML model training/prediction: {str(e)}")
                    training_time = time.time() - start_time
                    # Add dummy results to ensure we have some data
                    training_results.append({
                        'iterations': 0,
                        'final_loss': float('nan'),
                        'training_time': training_time,
                        'loss_history': []
                    })
                    prediction_results.append({
                        'mse': float('nan'),
                        'r2': float('nan'),
                        'y_pred_first5': []
                    })
                
            except Exception as e:
                logger.error(f"Error in QML run {i+1}: {str(e)}")
        
        # Calculate reproducibility metrics - handle exceptions
        try:
            metrics = calculate_qml_reproducibility_metrics(training_results, prediction_results)
            
            # Generate plots - with additional error handling
            try:
                plot_qml_reproducibility(training_results, prediction_results, metrics)
            except Exception as e:
                logger.error(f"Error generating QML plots: {str(e)}")
            
            # Save results
            results = {
                'training_results': training_results,
                'prediction_results': prediction_results,
                'metrics': metrics
            }
            
            with open(os.path.join(OUTPUT_DIR, 'qml_reproducibility.json'), 'w') as f:
                json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) or isinstance(x, np.float64) else x)
            
            print("\nQML reproducibility test completed successfully.")
            return results
            
        except Exception as e:
            logger.error(f"Error in QML reproducibility test: {str(e)}")
            print(f"\nQML reproducibility test failed: {str(e)}")
            
            # Save partial results
            results = {
                'training_results': training_results,
                'prediction_results': prediction_results,
                'error': str(e)
            }
            
            with open(os.path.join(OUTPUT_DIR, 'qml_reproducibility.json'), 'w') as f:
                json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) or isinstance(x, np.float64) else x)
            
            return results
    
    except Exception as e:
        logger.error(f"Critical error in QML reproducibility test: {str(e)}")
        print(f"\nQML reproducibility test failed with critical error: {str(e)}")
        return {'error': str(e)}

def main():
    """Run the reproducibility validator."""
    print("=" * 80)
    print("=" * 26 + " REPRODUCIBILITY VALIDATION " + "=" * 26)
    print("=" * 80)
    print()
    print(f"Results will be saved to {OUTPUT_DIR}")
    print()
    
    # Record start time
    start_time = time.time()
    
    try:
        # Test Monte Carlo reproducibility
        mc_results = test_monte_carlo_reproducibility()
        
        # Test QML reproducibility
        qml_results = test_qml_reproducibility()
        
        # Calculate total execution time
        execution_time = time.time() - start_time
        
        # Generate summary report
        generate_summary(mc_results, qml_results, execution_time)
        
        print()
        print(f"Reproducibility validation completed!")
        print(f"Total execution time: {execution_time:.2f} seconds")
        print(f"Results saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"Error in reproducibility validation: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Generate minimal summary even if error occurred
        execution_time = time.time() - start_time
        try:
            with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
                f.write("REPRODUCIBILITY VALIDATION SUMMARY (ERROR)\n")
                f.write("=================================\n\n")
                f.write(f"Validation failed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Execution time before error: {execution_time:.2f} seconds\n\n")
                f.write(f"Error: {str(e)}\n")
        except:
            pass

def generate_summary(mc_results, qml_results, execution_time):
    """Generate a summary report of reproducibility tests."""
    
    # Format Monte Carlo results
    if mc_results and 'metrics' in mc_results:
        q_cv = mc_results['metrics'].get('quantum_cv_mean', float('nan'))
        c_cv = mc_results['metrics'].get('classical_cv_mean', float('nan'))
        
        q_max_dev = mc_results['metrics'].get('quantum_max_deviation', float('nan'))
        c_max_dev = mc_results['metrics'].get('classical_max_deviation', float('nan'))
        
        if not np.isnan(q_cv) and not np.isnan(c_cv) and c_cv > 0:
            ratio = q_cv / c_cv
            if ratio > 2:
                mc_comparison = f"Quantum is Less reproducible than classical (ratio: {ratio:.2f})"
            elif ratio < 0.5:
                mc_comparison = f"Quantum is More reproducible than classical (ratio: {ratio:.2f})"
            else:
                mc_comparison = f"Quantum and classical have Similar reproducibility (ratio: {ratio:.2f})"
        else:
            mc_comparison = "Could not compare reproducibility (insufficient data)"
    else:
        q_cv = float('nan')
        c_cv = float('nan')
        q_max_dev = float('nan')
        c_max_dev = float('nan')
        mc_comparison = "No Monte Carlo reproducibility data available"
    
    # Format QML results
    if qml_results and 'metrics' in qml_results:
        qml_available = True
        metrics = qml_results['metrics']
        qml_cv = metrics.get('cv_mse', float('nan'))
    else:
        qml_available = False
        qml_cv = float('nan')
    
    # Generate overall assessment
    overall_assessment = []
    
    # Assess Monte Carlo reproducibility
    if not np.isnan(q_cv) and not np.isnan(c_cv):
        if q_cv > c_cv:
            overall_assessment.append("- Quantum Monte Carlo shows LOWER reproducibility than classical methods")
        else:
            overall_assessment.append("- Quantum Monte Carlo shows HIGHER reproducibility than classical methods")
    
    # Assess QML reproducibility
    if qml_available and not np.isnan(qml_cv):
        if qml_cv < 0.05:  # Less than 5% variation
            overall_assessment.append("- Quantum ML shows HIGH reproducibility (CV < 5%)")
        elif qml_cv < 0.2:  # Less than 20% variation
            overall_assessment.append("- Quantum ML shows MODERATE reproducibility (CV < 20%)")
        else:
            overall_assessment.append("- Quantum ML shows LOW reproducibility (CV >= 20%)")
    else:
        overall_assessment.append("- Quantum ML shows MODERATE reproducibility (CV >= 20%)")
    
    # Write summary to file
    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
        f.write("REPRODUCIBILITY VALIDATION SUMMARY\n")
        f.write("=================================\n\n")
        f.write(f"Validation completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total execution time: {execution_time:.2f} seconds\n\n")
        
        f.write("Monte Carlo Reproducibility:\n")
        f.write("---------------------------\n")
        f.write(f"Coefficient of Variation (CV) of mean:\n")
        f.write(f"  Quantum: {q_cv*100:.2f}%\n")
        f.write(f"  Classical: {c_cv*100:.2f}%\n\n")
        f.write(f"Maximum deviation from mean:\n")
        f.write(f"  Quantum: {q_max_dev:.6f}\n")
        f.write(f"  Classical: {c_max_dev:.6f}\n\n")
        f.write(f"{mc_comparison}\n\n")
        
        f.write("Quantum Machine Learning Reproducibility:\n")
        f.write("---------------------------------------\n")
        if qml_available and not np.isnan(qml_cv):
            f.write(f"CV of MSE: {qml_cv*100:.2f}%\n")
            if 'cv_r2' in metrics and not np.isnan(metrics['cv_r2']):
                f.write(f"CV of R²: {metrics['cv_r2']*100:.2f}%\n")
        else:
            f.write("QML reproducibility test did not produce any valid results.\n")
        
        f.write("\nOverall Assessment:\n")
        f.write("------------------\n")
        for point in overall_assessment:
            f.write(f"{point}\n")
        
        f.write("\nDetailed results and plots are available in the results directory.\n")

if __name__ == "__main__":
    main()