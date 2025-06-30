from scipy.optimize import minimize, OptimizeResult
import numpy as np
import time
from ..config.compressor_config import CompressorConfig
from ..config.optimization_config import OptimizationConfig
from .objective_function import ObjectiveFunction
from ..utils.logging_config import get_logger
from typing import List, Tuple

logger = get_logger(__name__)

class CompressorOptimizer:
    """
    QC-approved optimizer for Sprint 3A using L-BFGS-B algorithm.
    Implements 4-parameter optimization: [r1, r0_ratio, r2_ratio, r3_ratio] with r4=r3.
    """

    def __init__(self, compressor_config: CompressorConfig, opt_config: OptimizationConfig):
        """
        Initializes the optimizer with QC-approved settings.

        Args:
            compressor_config: The configuration for the compressor.
            opt_config: The configuration for the optimization algorithm.
        """
        self.compressor_config = compressor_config
        self.opt_config = opt_config
        self.objective_function = ObjectiveFunction(compressor_config, opt_config)

    def _get_initial_guess(self) -> np.ndarray:
        """
        QC-approved initial guess for 4/6 N-profile rotors.
        These ratios give geometry that passes circle residual check.
        """
        # QC recipe: r1 = target_D1e / 2 * 0.92 (slightly inside outer diameter)
        target_D1e_m = self.compressor_config.target_D1e / 1000.0  # Convert mm to m
        r1_guess = (target_D1e_m / 2) * 0.92
        
        # QC-approved ratios for City-type "N" rotors (4/6)
        r0_ratio_guess = 0.04   # r0/r1: 0.035-0.06 range
        r2_ratio_guess = 0.18   # r2/r1: 0.15-0.25 range  
        r3_ratio_guess = 0.12   # r3/r1: 0.09-0.18 range
        # r4 = r3 (constraint)
        
        initial_guess = np.array([r1_guess, r0_ratio_guess, r2_ratio_guess, r3_ratio_guess])
        
        logger.info(f"QC Initial guess: r1={r1_guess:.6f}m ({target_D1e_m/2*0.92*1000:.1f}mm)")
        logger.info(f"  Ratios: r0={r0_ratio_guess:.3f}, r2={r2_ratio_guess:.3f}, r3={r3_ratio_guess:.3f}")
        logger.info(f"  Absolute: r0={r1_guess*r0_ratio_guess*1000:.2f}mm, r2={r1_guess*r2_ratio_guess*1000:.2f}mm, r3={r1_guess*r3_ratio_guess*1000:.2f}mm")
        
        return initial_guess

    def _get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """
        QC-approved parameter bounds using City-type N-profile ratio ranges.
        Much tighter than previous 50% windows.
        
        CRITICAL FIX: Constraint check uses r2/r1w, but bounds here are relative to r1.
        Need to convert r1w-relative limits to r1-relative bounds.
        """
        # Estimate reasonable r1 range from target diameter
        target_D1e_m = self.compressor_config.target_D1e / 1000.0
        r1_center = (target_D1e_m / 2) * 0.92
        r1_min = r1_center * 0.85  # Allow some variation around center
        r1_max = r1_center * 1.15
        
        # QC ratio ranges for City-type "N" rotors - RESTORED SIMPLE BOUNDS per QC feedback
        r0_ratio_min, r0_ratio_max = 0.035, 0.060   # inner fillet
        r2_ratio_min, r2_ratio_max = 0.150, 0.350   # r2/r1 - WIDENED per QC
        r3_ratio_min, r3_ratio_max = 0.090, 0.200   # gate root (C-D arc)
        
        bounds = [
            (0.8*r1_center, 1.3*r1_center),       # r1 - QC requested format
            (0.035, 0.060),                        # r0/r1
            (0.150, 0.350),                        # r2/r1 ← widened per QC
            (0.090, 0.200)                         # r3/r1
        ]
        
        logger.info(f"QC Parameter bounds (SIMPLE - per QC feedback):")
        logger.info(f"  r1: [{0.8*r1_center*1000:.1f}, {1.3*r1_center*1000:.1f}] mm")
        logger.info(f"  r0/r1: [0.035, 0.060]")
        logger.info(f"  r2/r1: [0.150, 0.350] WIDENED")
        logger.info(f"  r3/r1: [0.090, 0.200]")
        logger.info(f"  Constraint r2/r1w <= 0.305 enforced in objective function")
        
        return bounds

    def _scale_variables(self, params: np.ndarray) -> np.ndarray:
        """
        QC-approved gradient scaling: normalize variables by r1 for L-BFGS-B.
        """
        if not self.opt_config.gradient_scaling:
            return params
            
        r1 = params[0]
        scaled_params = params.copy()
        scaled_params[0] = 1.0  # r1 normalized to 1
        # Ratios remain unchanged as they're already dimensionless
        return scaled_params

    def _unscale_variables(self, scaled_params: np.ndarray, r1_scale: float) -> np.ndarray:
        """
        Convert scaled variables back to physical units.
        """
        if not self.opt_config.gradient_scaling:
            return scaled_params
            
        unscaled_params = scaled_params.copy()
        unscaled_params[0] = scaled_params[0] * r1_scale  # Restore r1 scale
        return unscaled_params

    def optimize(self) -> dict:
        """
        Run QC-approved L-BFGS-B optimization for Sprint 3A.

        Returns:
            A dictionary containing the results of the optimization.
        """
        logger.info("Starting Sprint 3A optimization with L-BFGS-B")
        start_time = time.time()
        
        # Set up bounds  
        bounds = self._get_parameter_bounds()
        initial_guess = self._get_initial_guess()
        
        # Calculate r1_center for logging (same as in _get_parameter_bounds)
        target_D1e_m = self.compressor_config.target_D1e / 1000.0
        r1_center = (target_D1e_m / 2) * 0.92
        
        logger.info(f"QC Parameter bounds (SIMPLE - per QC feedback):")
        logger.info(f"  r1: [{0.8*r1_center*1000:.1f}, {1.3*r1_center*1000:.1f}] mm")
        logger.info(f"  r0/r1: [0.035, 0.060]")
        logger.info(f"  r2/r1: [0.150, 0.350] WIDENED")
        logger.info(f"  r3/r1: [0.090, 0.200]")
        logger.info(f"  Constraint r2/r1w <= 0.305 enforced in objective function")
        
        # QC-approved L-BFGS-B optimization with proper settings
        logger.info("Starting L-BFGS-B optimization with QC settings...")
        
        # QC: Instrumentation for progress tracking
        self.iteration_count = 0
        self.last_objective = float('inf')
        
        def iteration_callback(xk):
            """QC-required iteration progress instrumentation - every 10 iterations."""
            self.iteration_count += 1
            
            # Print progress every 10 iterations as requested by QC
            if self.iteration_count % 10 == 0:
                try:
                    r_params = self._params_to_dict(xk)
                    
                    # Use QC analytic diameter formulas  
                    r1w_m = self.compressor_config.r1w / 1000.0
                    r2w_m = self.compressor_config.r2w / 1000.0
                    
                    D1e_calc = 2 * (r1w_m + r_params['r2']) * 1000
                    D2e_calc = 2 * (r2w_m + r_params['r4']) * 1000
                    
                    # Calculate errors (using QC analytic formulas)
                    D1e_error_mm = D1e_calc - self.compressor_config.target_D1e
                    D2e_error_mm = D2e_calc - self.compressor_config.target_D2e
                    
                    # Volume will be calculated by the actual objective function
                    current_obj = self.objective_function(xk)
                    
                    print(f"\nQC ITERATION {self.iteration_count}: Obj={current_obj:.0f}, D1={D1e_error_mm:+.1f}mm, D2={D2e_error_mm:+.1f}mm")
                    
                    # QC: Check if within acceptance criteria
                    if abs(D1e_error_mm) <= 0.30 and abs(D2e_error_mm) <= 0.30:
                        print("QC DIAMETER CRITERIA MET! D1e and D2e errors <= 0.30mm")
                    
                except Exception as e:
                    print(f"Iter {self.iteration_count}: Progress calculation error: {e}")
        
        # QC: Updated settings per QC specifications with fallback algorithms
        algorithms_to_try = [
            ('L-BFGS-B', {
                'maxiter': 200,     # QC: Raised to 200 for full iterations
                'maxfun': 400,      # QC: Limit function evaluations
                'factr': 1e7,       # QC: Use factr instead of pgtol for older SciPy
                'ftol': self.opt_config.tolerance,
                'gtol': 1e-8,
                'disp': True
            }),
            ('SLSQP', {
                'maxiter': 200,
                'ftol': self.opt_config.tolerance,
                'disp': True
            }),
            ('TNC', {
                'maxiter': 200,
                'maxfun': 400,
                'ftol': self.opt_config.tolerance,
                'disp': True
            })
        ]
        
        result = None
        for algorithm, options in algorithms_to_try:
            logger.info(f"Trying optimization algorithm: {algorithm}")
            try:
                result = minimize(
                    fun=self.objective_function,
                    x0=initial_guess,
                    method=algorithm,
                    bounds=bounds,
                    callback=iteration_callback if algorithm == 'L-BFGS-B' else None,
                    options=options
                )
                
                # If successful, use this result
                if result.success:
                    logger.info(f"SUCCESS: {algorithm} converged successfully!")
                    break
                else:
                    logger.warning(f"ERROR: {algorithm} failed: {result.message}")
                    
            except Exception as e:
                logger.warning(f"ERROR: {algorithm} crashed: {e}")
                continue
        
        # Handle case where all algorithms failed
        if result is None:
            logger.error("ERROR: ALL optimization algorithms failed!")
            from scipy.optimize import OptimizeResult
            result = OptimizeResult({
                'success': False,
                'message': 'All optimization algorithms failed',
                'fun': np.inf,
                'x': initial_guess,
                'nit': 0,
                'nfev': 0
            })
        
        end_time = time.time()
        optimization_time = end_time - start_time
        
        # Process results with QC validation
        if result.success:
            final_params = self._params_to_dict(result.x)
            final_error = result.fun
            
            # QC validation: ensure geometry is actually feasible
            try:
                from ..geometry.rack_profile import NRackProfile
                test_profile = NRackProfile(final_params, self.compressor_config)
                logger.info("SUCCESS: QC validation: Final geometry validation complete")
            except Exception as e:
                logger.warning(f"Final geometry validation failed: {e}")
                final_error = np.inf
            
            logger.info(f"SUCCESS: L-BFGS-B converged successfully!")
            logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
            logger.info(f"Final objective value: {result.fun:.6f}")
            logger.info(f"Iterations: {result.nit}, Function evaluations: {result.nfev}")
            
        else:
            final_params = None
            final_error = np.inf
            logger.warning(f"ERROR: L-BFGS-B failed to converge: {result.message}")
            logger.warning(f"Final objective value: {result.fun:.6f}")
            logger.warning(f"Iterations: {result.nit}, Function evaluations: {result.nfev}")
        
        return {
            'success': result.success and final_error < np.inf,
            'final_params': final_params,
            'final_error': final_error,
            'iterations': result.nit,
            'function_evaluations': result.nfev,
            'time': optimization_time,
            'optimizer_result': result
        }

    def _process_results(self, result, optimization_time: float) -> dict:
        """
        Process optimization results and check QC success criteria.
        """
        # Extract final parameters
        r1, r0_ratio, r2_ratio, r3_ratio = result.x
        r0 = r1 * r0_ratio
        r2 = r1 * r2_ratio
        r3 = r1 * r3_ratio
        r4 = r3  # QC constraint
        
        final_params = {
            'r1': r1,
            'r0': r0, 
            'r2': r2,
            'r3': r3,
            'r4': r4,
            'r_tip': r1 * 0.15  # Default for compatibility
        }
        
        # Calculate final errors for QC validation
        try:
            final_error_value = self.objective_function.evaluate(result.x)
            
            # Extract individual error components (re-calculate for clarity)
            opt_params = {'r0': r0, 'r1': r1, 'r2': r2, 'r3': r3, 'r4': r4, 'r_tip': r1 * 0.15}
            
            # Quick geometry check
            from ..geometry.rack_profile import NRackProfile
            from ..geometry.rotor_generation import RotorGenerator
            from ..geometry.volume_calculator import SweptVolumeCalculator
            
            rack = NRackProfile(opt_params, self.compressor_config)
            rotor_gen = RotorGenerator(rack.get_full_profile(), self.compressor_config)
            rotor_gen.generate_rotors()
            
            volume_calc = SweptVolumeCalculator(rotor_gen.main_rotor_profile, self.compressor_config)
            achieved_volume = volume_calc.calculate_volume_per_revolution()
            
            main_radii = np.linalg.norm(rotor_gen.main_rotor_profile, axis=1)
            achieved_D1e = 2 * np.max(main_radii) * 1000
            
            gate_radii = np.linalg.norm(rotor_gen.gate_rotor_profile, axis=1) 
            achieved_D2e = 2 * np.max(gate_radii) * 1000
            
            # Calculate QC success criteria
            volume_error_percent = abs(achieved_volume - self.compressor_config.target_swv_per_rev) / self.compressor_config.target_swv_per_rev
            d1e_error_mm = abs(achieved_D1e - self.compressor_config.target_D1e)
            d2e_error_mm = abs(achieved_D2e - self.compressor_config.target_D2e)
            
            # QC success check
            meets_volume_criteria = volume_error_percent <= self.opt_config.target_volume_error
            meets_d1e_criteria = d1e_error_mm <= self.opt_config.target_diameter_error  
            meets_d2e_criteria = d2e_error_mm <= self.opt_config.target_diameter_error
            meets_time_criteria = optimization_time <= 300  # 5 minutes
            
            qc_success = meets_volume_criteria and meets_d1e_criteria and meets_d2e_criteria and meets_time_criteria
            
        except Exception as e:
            logger.error(f"Error in final validation: {e}")
            qc_success = False
            volume_error_percent = 1.0
            d1e_error_mm = 999.0
            d2e_error_mm = 999.0
        
        # Get optimization statistics
        stats = self.objective_function.get_statistics()
        
        return {
            'success': result.success and qc_success,
            'qc_success': qc_success,
            'message': result.message,
            'optimized_parameters': final_params,
            'final_error': result.fun,
            'iterations': result.nit,
            'function_evaluations': result.nfev,
            'optimization_time': optimization_time,
            'volume_error_percent': volume_error_percent * 100,
            'D1e_error_mm': d1e_error_mm,
            'D2e_error_mm': d2e_error_mm,
            'meets_qc_criteria': {
                'volume': meets_volume_criteria,
                'D1e': meets_d1e_criteria, 
                'D2e': meets_d2e_criteria,
                'time': meets_time_criteria
            },
            'statistics': stats
        }

    def _create_failure_result(self, error_message: str, optimization_time: float) -> dict:
        """Create a failure result dictionary."""
        return {
            'success': False,
            'qc_success': False,
            'message': f"Optimization failed: {error_message}",
            'optimized_parameters': {},
            'final_error': float('inf'),
            'iterations': 0,
            'function_evaluations': 0,
            'optimization_time': optimization_time,
            'volume_error_percent': 100.0,
            'D1e_error_mm': 999.0,
            'D2e_error_mm': 999.0,
            'meets_qc_criteria': {
                'volume': False,
                'D1e': False,
                'D2e': False,
                'time': False
            },
            'statistics': self.objective_function.get_statistics()
        }

    def _params_to_dict(self, params: np.ndarray) -> dict:
        """
        Convert normalized parameters [r1, r0_ratio, r2_ratio, r3_ratio] to r_params dict.
        
        QC constraint: r4 = r3 (hard equality).
        """
        r1, r0_ratio, r2_ratio, r3_ratio = params
        
        return {
            'r0': r1 * r0_ratio,
            'r1': r1,
            'r2': r1 * r2_ratio,
            'r3': r1 * r3_ratio,
            'r4': r1 * r3_ratio,  # QC constraint: r4 = r3
            'r_tip': r1 * 0.18   # Default r_tip ratio
        } 