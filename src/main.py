import os
from .utils.csv_loader import CSVDataManager
from .config.compressor_config import CompressorConfig
from .config.optimization_config import OptimizationConfig
from .optimization.optimizer import CompressorOptimizer
from .utils.logging_config import setup_logging
import pprint
import logging

def main():
    """
    Main execution function to run the screw compressor optimization.
    """
    # --- Setup ---
    # Set up logging to capture detailed debug information
    setup_logging(level=logging.DEBUG)

    # Construct the path to the CSV file relative to this script's location
    # This makes the script runnable from any directory
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, '..', '..', 'Master_updated.csv')
    
    # Define the model to be optimized - QC approved target for Sprint 3A
    model_to_optimize = "Frick NGC 100 A"

    print("--- Screw Compressor Geometry Optimizer ---")
    print(f"Loading data from: {os.path.abspath(csv_path)}")
    print(f"Target model for optimization: {model_to_optimize}\n")

    # --- Data Loading ---
    try:
        data_manager = CSVDataManager(csv_path)
        compressor_config = data_manager.get_configuration(model_to_optimize)
        print("Compressor Configuration:")
        pprint.pprint(compressor_config)
        print("-" * 30)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"Error during data loading: {e}")
        return

    # --- Optimization ---
    opt_config = OptimizationConfig()
    optimizer = CompressorOptimizer(compressor_config, opt_config)
    
    print("Starting optimization...")
    results = optimizer.optimize()
    print("Optimization complete.\n")

    # --- Results ---
    print("--- Optimization Results ---")
    pprint.pprint(results)

    if results['success']:
        print("\n✅ Optimization successful!")
        print("Found optimal parameters (in meters):")
        for param, value in results['final_params'].items():
            print(f"  - {param}: {value:.6f}")
    else:
        print("\n❌ Optimization failed or did not converge to a satisfactory solution.")
        print(f"  - Reason: {results.get('message', 'No message provided.')}")

if __name__ == "__main__":
    main() 