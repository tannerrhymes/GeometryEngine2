import numpy as np
from typing import Dict
from ..config.compressor_config import CompressorConfig
from ..config.optimization_config import OptimizationConfig
from ..geometry.rack_profile import NRackProfile
from ..geometry.rotor_generation import RotorGenerator
from ..geometry.volume_calculator import SweptVolumeCalculator
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

class ObjectiveFunction:
    """
    The objective function for screw compressor optimization.
    It evaluates a set of geometric parameters and returns a single error value
    representing how well the resulting geometry meets the specified targets.
    """
    
    def __init__(self, compressor_config: CompressorConfig, opt_config: OptimizationConfig):
        """
        Initializes the objective function.

        Args:
            compressor_config (CompressorConfig): The configuration for the compressor.
            opt_config (OptimizationConfig): The configuration for the optimization.
        """
        self.compressor_config = compressor_config
        self.opt_config = opt_config
        
        # Store target values for quick access
        self.target_volume = self.compressor_config.target_swv_per_rev
        self.target_D1e = self.compressor_config.target_D1e
        self.target_D2e = self.compressor_config.target_D2e
        
        # Statistics to track performance
        self.evaluation_count = 0
        self.best_error = float('inf')

    def evaluate(self, params: np.ndarray) -> float:
        """
        Evaluates the objective function for a given set of parameters.

        Args:
            params (np.ndarray): An array containing the optimization variables
                                 [r1, r0_r1_ratio, r2_r1_ratio, r3_r1_ratio].

        Returns:
            float: The combined, weighted error for the given parameters.
        """
        self.evaluation_count += 1
        
        try:
            r1, r0_r1, r2_r1, r3_r1 = params
            # Calculate absolute values from ratios
            r0 = r1 * r0_r1
            r2 = r1 * r2_r1
            r3 = r1 * r3_r1
            opt_params = {'r0': r0, 'r1': r1, 'r2': r2, 'r3': r3}
            
            # --- Enforce Geometric Constraint as a Soft Penalty ---
            r1w_m = self.compressor_config.r1w / 1000.0
            r1e_m = self.compressor_config.target_D1e / 2000.0
            r1i_m = r1w_m - r0
            
            constraint_violation = 0
            if r1i_m <= 0:
                constraint_violation = 1e6 # Penalize invalid inner radius
            else:
                psi_1 = np.pi / self.compressor_config.z1
                psi_2 = 2 * np.pi / self.compressor_config.z1 - psi_1
                A = r1i_m + r3
                B = r1e_m - r1
                C = r1 + r3
                if (2 * A * B) != 0:
                    cos_val = (A**2 + B**2 - C**2) / (2 * A * B)
                    if not -1 <= cos_val <= 1:
                        constraint_violation = 1e6 # Penalize invalid triangle
                    else:
                        lhs = 2 * A * B * np.cos(psi_2)
                        rhs = A**2 + B**2 - C**2
                        constraint_violation = (lhs - rhs)**2
                else:
                    constraint_violation = 1e6
            
            # --- Full Geometry Pipeline ---
            rack = NRackProfile(opt_params, self.compressor_config)
            
            rotor_gen = RotorGenerator(rack.get_full_profile(), self.compressor_config)
            rotor_gen.generate_rotors()
            
            volume_calc = SweptVolumeCalculator(rotor_gen.main_rotor_profile, self.compressor_config)
            achieved_volume = volume_calc.calculate_volume_per_revolution()

            main_radii = np.linalg.norm(rotor_gen.main_rotor_profile, axis=1)
            achieved_D1e = 2 * np.max(main_radii) * 1000
            
            gate_radii = np.linalg.norm(rotor_gen.gate_rotor_profile, axis=1)
            achieved_D2e = 2 * np.max(gate_radii) * 1000

            # --- Calculate Error Components ---
            volume_error = (achieved_volume - self.target_volume)**2
            d1e_error = (achieved_D1e - self.target_D1e)**2
            d2e_error = (achieved_D2e - self.target_D2e)**2
            
            # --- Apply Weights ---
            weighted_error = (
                self.opt_config.weight_volume * volume_error +
                self.opt_config.weight_D1e * d1e_error +
                self.opt_config.weight_D2e * d2e_error +
                constraint_violation * 1e7 # High penalty for constraint violation
            )

            if self.evaluation_count % 500 == 0 or weighted_error < self.best_error:
                logger.debug(f"--- Eval #{self.evaluation_count} ---")
                logger.debug(f"  Params: r1={r1:.6f}, r0={r0:.6f}, r2={r2:.6f}, r3={r3:.6f}")
                logger.debug(f"  Vol:    Achieved={achieved_volume:.6f}, Target={self.target_volume:.6f}, Err={volume_error:.6f}")
                logger.debug(f"  D1e:    Achieved={achieved_D1e:.2f}mm, Target={self.target_D1e:.2f}mm, Err={d1e_error:.4f}")
                logger.debug(f"  D2e:    Achieved={achieved_D2e:.2f}mm, Target={self.target_D2e:.2f}mm, Err={d2e_error:.4f}")
                logger.debug(f"  ConstraintViolation: {constraint_violation:.6f}")
                logger.debug(f"  Weighted Err: {weighted_error:.4f}")

            if weighted_error < self.best_error:
                self.best_error = weighted_error
                logger.info(f"New best error: {self.best_error:.4f} found at Eval #{self.evaluation_count}")
            
            return weighted_error

        except (ValueError, IndexError, KeyError, RuntimeError) as e:
            logger.warning(f"Geometry failure for params r1={params[0]:.5f}. Reason: {e}", exc_info=False)
            return self.opt_config.penalty_geometric 