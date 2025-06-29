from scipy.optimize import differential_evolution
import numpy as np
from ..config.compressor_config import CompressorConfig
from ..config.optimization_config import OptimizationConfig
from .objective_function import ObjectiveFunction

class CompressorOptimizer:
    """
    Main optimizer class that uses differential evolution to find the optimal
    geometric parameters for a screw compressor.
    """

    def __init__(self, compressor_config: CompressorConfig, opt_config: OptimizationConfig):
        """
        Initializes the optimizer.

        Args:
            compressor_config: The configuration for the compressor.
            opt_config: The configuration for the optimization algorithm.
        """
        self.compressor_config = compressor_config
        self.opt_config = opt_config
        self.objective_function = ObjectiveFunction(compressor_config, opt_config)

    def _get_optimization_bounds(self) -> list:
        """
        Calculates optimization bounds for the new parameter set:
        [r1, r0/r1 ratio, r2/r1 ratio, r3/r1 ratio].
        This version uses the user's literature-based guesses to center the search
        in a promising region of the parameter space.
        """
        D1e_mm = self.compressor_config.target_D1e
        
        # Use the user's provided values as the center of our search space.
        r0_guess = 0.00648 * D1e_mm
        r1_guess = 0.1507 * D1e_mm
        r2_guess = 0.03515 * D1e_mm
        r3_guess = 0.0406 * D1e_mm
        
        # Convert guesses to meters for the optimizer
        r1_guess_m = r1_guess / 1000.0
        
        bounds = [
            # r1 bounds: Search in a +/- 50% range around the excellent guess.
            (r1_guess_m * 0.5, r1_guess_m * 1.5),
            
            # r0/r1 ratio bounds: Centered around the guess ratio.
            ( (r0_guess / r1_guess) * 0.5, (r0_guess / r1_guess) * 1.5),
            
            # r2/r1 ratio bounds: Centered around the guess ratio.
            ( (r2_guess / r1_guess) * 0.5, (r2_guess / r1_guess) * 1.5),

            # r3/r1 ratio bounds: Centered around the guess ratio.
            ( (r3_guess / r1_guess) * 0.5, (r3_guess / r1_guess) * 1.5)
        ]
        return bounds

    def optimize(self) -> dict:
        """
        Runs the differential evolution optimization.

        Returns:
            A dictionary containing the results of the optimization.
        """
        bounds = self._get_optimization_bounds()
        
        result = differential_evolution(
            self.objective_function.evaluate,
            bounds,
            strategy=self.opt_config.strategy,
            maxiter=self.opt_config.maxiter,
            popsize=self.opt_config.popsize,
            tol=self.opt_config.tolerance,
            seed=42  # for reproducibility
        )
        
        return self._process_results(result)

    def _process_results(self, result) -> dict:
        """
        Processes the raw result from scipy.optimize into a clean dictionary.
        """
        success = result.success and result.fun < 1e-3

        final_params = {}
        if success:
            r1, r0_r1, r2_r1, r3_r1 = result.x
            final_params = {
                'r1': r1,
                'r0': r1 * r0_r1,
                'r2': r1 * r2_r1,
                'r3': r1 * r3_r1
            }
            
        return {
            'success': success,
            'message': result.message,
            'optimized_parameters': final_params,
            'final_error': result.fun,
            'iterations': result.nit,
            'function_evaluations': result.nfev,
        } 