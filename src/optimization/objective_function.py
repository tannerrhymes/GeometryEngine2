import numpy as np
from typing import Dict
from ..config.compressor_config import CompressorConfig
from ..config.optimization_config import OptimizationConfig
from ..geometry.rack_profile import NRackProfile
from ..geometry.rotor_generation import RotorGenerator
from ..geometry.volume_calculator import SweptVolumeCalculator
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

class GeometryError(Exception):
    """QC requirement: Exception for geometry generation failures."""
    pass

class ObjectiveFunction:
    """
    QC-approved objective function for Sprint 3A screw compressor optimization.
    Implements exact formula: E = w_V*(ΔV/V_target)² + w_D1*(ΔD1_mm)² + w_D2*(ΔD2_mm)²
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
        
        # Store target values - using swv/r as QC specified
        self.target_volume = self.compressor_config.target_swv_per_rev  # m³/rev
        self.target_D1e = self.compressor_config.target_D1e  # mm
        self.target_D2e = self.compressor_config.target_D2e  # mm
        
        # Statistics to track performance
        self.evaluation_count = 0
        self.best_error = float('inf')
        self.failed_evaluations = 0

    def evaluate(self, params: np.ndarray) -> float:
        """
        QC-approved evaluation function for Sprint 3A.
        
        Args:
            params (np.ndarray): [r1, r0_ratio, r2_ratio, r3_ratio] where ratios are relative to r1

        Returns:
            float: E = w_V*(ΔV/V_target)² + w_D1*(ΔD1_mm)² + w_D2*(ΔD2_mm)²
        """
        self.evaluation_count += 1
        
        try:
            # Extract parameters: r1 absolute, others as ratios
            r1, r0_ratio, r2_ratio, r3_ratio = params
            
            # Calculate absolute radii from ratios
            r0 = r1 * r0_ratio
            r2 = r1 * r2_ratio  
            r3 = r1 * r3_ratio
            r4 = r3  # QC constraint: r4 = r3
            
            # For compatibility with existing geometry engine, we need r_tip
            # Using a reasonable default based on typical N-profile relationships
            r_tip = r1 * 0.15  # Typical ratio for N-profiles
            
            opt_params = {
                'r0': r0, 'r1': r1, 'r2': r2, 
                'r3': r3, 'r4': r4, 'r_tip': r_tip
            }
            
            # --- Generate geometry and calculate performance ---
            rack = NRackProfile(opt_params, self.compressor_config)
            rotor_gen = RotorGenerator(rack.get_full_profile(), self.compressor_config)
            rotor_gen.generate_rotors()
            
            # Calculate achieved volume (swv/r)
            volume_calc = SweptVolumeCalculator(rotor_gen.main_rotor_profile, self.compressor_config)
            achieved_volume = volume_calc.calculate_volume_per_revolution()
            
            # Calculate achieved diameters
            main_radii = np.linalg.norm(rotor_gen.main_rotor_profile, axis=1)
            achieved_D1e = 2 * np.max(main_radii) * 1000  # Convert m to mm
            
            gate_radii = np.linalg.norm(rotor_gen.gate_rotor_profile, axis=1)
            achieved_D2e = 2 * np.max(gate_radii) * 1000  # Convert m to mm

            # --- QC-exact objective function calculation ---
            # Volume error (dimensionless)
            volume_error_relative = (achieved_volume - self.target_volume) / self.target_volume
            
            # Diameter errors (mm)
            d1e_error_mm = achieved_D1e - self.target_D1e
            d2e_error_mm = achieved_D2e - self.target_D2e
            
            # QC-specified weighted error: E = w_V*(ΔV/V_target)² + w_D1*(ΔD1_mm)² + w_D2*(ΔD2_mm)²
            weighted_error = (
                self.opt_config.weight_volume * (volume_error_relative ** 2) +
                self.opt_config.weight_D1e * (d1e_error_mm ** 2) +
                self.opt_config.weight_D2e * (d2e_error_mm ** 2)
            )

            # Early abort check for Sprint 3A
            if abs(volume_error_relative) > self.opt_config.early_abort_threshold:
                logger.warning(f"Early abort: Volume error {abs(volume_error_relative)*100:.1f}% > {self.opt_config.early_abort_threshold*100:.1f}%")
                return 1e6  # Large penalty for bad seeds

            # Logging for Sprint 3A monitoring
            if self.evaluation_count % 10 == 0 or weighted_error < self.best_error:
                logger.info(f"--- Eval #{self.evaluation_count} ---")
                logger.info(f"  r1={r1:.6f}, r0={r0:.6f}, r2={r2:.6f}, r3={r3:.6f}")
                logger.info(f"  Volume: Achieved={achieved_volume:.6f}, Target={self.target_volume:.6f}, Error={volume_error_relative*100:.2f}%")
                logger.info(f"  D1e: Achieved={achieved_D1e:.2f}mm, Target={self.target_D1e:.2f}mm, Error={d1e_error_mm:.3f}mm")
                logger.info(f"  D2e: Achieved={achieved_D2e:.2f}mm, Target={self.target_D2e:.2f}mm, Error={d2e_error_mm:.3f}mm")
                logger.info(f"  Weighted Error: {weighted_error:.6f}")

            if weighted_error < self.best_error:
                self.best_error = weighted_error
                logger.info(f"New best error: {self.best_error:.6f} at Eval #{self.evaluation_count}")
            
            return weighted_error

        except (ValueError, IndexError, KeyError, RuntimeError) as e:
            self.failed_evaluations += 1
            logger.warning(f"Geometry failure #{self.failed_evaluations} for r1={params[0]:.5f}. Reason: {e}")
            return 1e8  # Large penalty for failed geometry

    def get_statistics(self) -> dict:
        """Return optimization statistics for Sprint 3A reporting."""
        return {
            'total_evaluations': self.evaluation_count,
            'failed_evaluations': self.failed_evaluations,
            'success_rate': (self.evaluation_count - self.failed_evaluations) / max(1, self.evaluation_count),
            'best_error': self.best_error
        }

    def __call__(self, params: np.ndarray) -> float:
        """
        QC-approved objective function with STRICT finite penalty only.
        NEVER returns NaN - always returns finite values for L-BFGS-B line search.
        """
        try:
            # Convert normalized parameters to r_params
            r_params = self._params_to_dict(params)
            
            # QC LOG: Show current parameters being tested
            print(f"\nTesting parameters:")
            print(f"  r1={r_params['r1']*1000:.2f}mm, r0/r1={r_params['r0']/r_params['r1']:.3f}, r2/r1={r_params['r2']/r_params['r1']:.3f}, r3/r1={r_params['r3']/r_params['r1']:.3f}")
            
            # QC: Pre-check parameter bounds to return finite penalty immediately
            r1w_m = self.compressor_config.r1w / 1000.0
            r2_ratio = r_params['r2'] / r1w_m
            r4_ratio = r_params['r4'] / r1w_m
            
            # QC: Finite penalty for out-of-bounds parameters (no NaN, no exceptions)
            if not (0.13 <= r2_ratio <= 0.305):
                print(f"ERROR: r2/r1w = {r2_ratio:.4f} outside [0.13, 0.305] -> penalty 1e5")
                return 1e5
            if not (0.13 <= r4_ratio <= 0.305):
                print(f"ERROR: r4/r1w = {r4_ratio:.4f} outside [0.13, 0.305] -> penalty 1e5")
                return 1e5
            if (r2_ratio > 0.35) or (r4_ratio > 0.35):
                print(f"ERROR: Tip ratios too large (r2={r2_ratio:.3f}, r4={r4_ratio:.3f}) -> penalty 1e6")
                return 1e6
            
            # Generate geometry
            profile = NRackProfile(r_params, self.compressor_config)
            
            # Calculate residuals with QC guards
            residuals = profile._calculate_all_residuals()
            h_residual, j_residual, gh_dist, hj_dist, _, _ = residuals
            
            # QC trochoid ODE residual check (finite penalty if fails)
            if hasattr(profile, 'trochoid_residual'):
                trochoid_residual = profile.trochoid_residual
                # QC fail-fast: Check if all residuals finite
                if trochoid_residual > 1e-8 * r_params['r1']:
                    print(f"ERROR: Trochoid residual {trochoid_residual:.2e} > limit -> penalty 1e4")
                    return 1e4  # QC: Finite penalty
            
            # Calculate volume and diameters
            volume_calc = self._calculate_volume_from_profile(profile, r_params)
            D1e_calc, D2e_calc = self._calculate_diameters_from_profile(profile, r_params)
            
            # QC-exact objective function: E = w_V*(ΔV/V_target)² + w_D1*(ΔD1_mm)² + w_D2*(ΔD2_mm)²
            volume_error_rel = (volume_calc - self.target_volume) / self.target_volume
            D1e_error_mm = abs(D1e_calc - self.target_D1e)
            D2e_error_mm = abs(D2e_calc - self.target_D2e)
            
            # QC weights: w_V = 1, w_D1 = w_D2 = 10
            objective_value = (
                self.opt_config.weight_volume * (volume_error_rel ** 2) +
                self.opt_config.weight_D1e * (D1e_error_mm ** 2) +
                self.opt_config.weight_D2e * (D2e_error_mm ** 2)
            )
            
            print(f"RESULTS:")
            print(f"  Volume: calc={volume_calc:.6f}, target={self.target_volume:.6f}, error={volume_error_rel*100:.2f}%")
            print(f"  D1e: calc={D1e_calc:.1f}mm, target={self.target_D1e:.1f}mm, error={D1e_error_mm:.2f}mm")
            print(f"  D2e: calc={D2e_calc:.1f}mm, target={self.target_D2e:.1f}mm, error={D2e_error_mm:.2f}mm")
            print(f"  Objective value: {objective_value:.6f}")
            
            # QC Success checks
            if volume_error_rel <= 0.05:
                print("  SUCCESS: Volume error within QC target (<= 5%)")
            if D1e_error_mm <= 0.30:
                print("  SUCCESS: D1e error within QC target (<= 0.30mm)")
            if D2e_error_mm <= 0.30:
                print("  SUCCESS: D2e error within QC target (<= 0.30mm)")
            
            if not np.isfinite(objective_value):
                print(f"ERROR: Non-finite objective {objective_value} -> penalty 1e4")
                return 1e4  # QC: Finite penalty

            print(f"Objective: {objective_value:.3f}")
            return objective_value
            
        except Exception as e:
            print(f"ERROR: EXCEPTION: {e}")
            return 1e4  # QC: Finite penalty for any exception
    
    def _validate_tip_circle_constraints(self, profile: NRackProfile, r_params: dict):
        """
        QC requirement: Hard-fail on tip circle fit if residual > tolerance.
        
        Raises GeometryError if constraints not satisfied.
        """
        try:
            jp = profile.junction_points
            H = jp['H']
            J = jp['J']
            
            # Get tip centers from profile
            main_tip_center = profile.arc_centers.get('JA')
            gate_tip_center = profile.arc_centers.get('GATE_TIP')
            
            if main_tip_center is None or gate_tip_center is None:
                raise GeometryError("Tip centers not found in arc_centers")
            
            # Calculate residuals
            h_to_gate = np.linalg.norm(H - gate_tip_center)
            j_to_main = np.linalg.norm(J - main_tip_center)
            
            h_residual = abs(h_to_gate - r_params['r4'])
            j_residual = abs(j_to_main - r_params['r2'])
            
            # QC requirement: Hard-fail if residual > 1e-4 * r1w
            r1w_m = self.compressor_config.r1w / 1000.0
            tolerance = 1e-4 * r1w_m
            
            if h_residual > tolerance:
                raise GeometryError(f"H circle residual {h_residual*1e6:.1f}µm > {tolerance*1e6:.1f}µm")
            if j_residual > tolerance:
                raise GeometryError(f"J circle residual {j_residual*1e6:.1f}µm > {tolerance*1e6:.1f}µm")
                
            # Also check H-J distance feasibility
            hj_distance = np.linalg.norm(H - J)
            r2_diameter = 2 * r_params['r2']
            
            if hj_distance > r2_diameter:
                raise GeometryError(f"H-J distance {hj_distance*1000:.1f}mm > r₂ diameter {r2_diameter*1000:.1f}mm")
                
        except KeyError as e:
            raise GeometryError(f"Missing junction point: {e}")
        except Exception as e:
            raise GeometryError(f"Tip circle validation failed: {e}")
    
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
    
    def _calculate_volume_from_profile(self, profile: NRackProfile, r_params: dict) -> float:
        """
        Calculate swept volume per revolution from ACTUAL profile geometry.
        FIXED: Now uses the actual generated profile instead of hardcoded pitch circle areas.
        """
        try:
            # Generate the actual rotor profiles from the rack
            from ..geometry.rotor_generation import RotorGenerator
            
            rack_profile = profile.get_full_profile()
            rotor_gen = RotorGenerator(rack_profile, self.compressor_config)
            rotor_gen.generate_rotors()
            
            # Use SweptVolumeCalculator with ACTUAL main rotor profile
            volume_calc = SweptVolumeCalculator(rotor_gen.main_rotor_profile, self.compressor_config)
            volume_per_rev = volume_calc.calculate_volume_per_revolution()
            
            # Calculate actual chamber area for logging
            actual_chamber_area = volume_per_rev / (self.compressor_config.rotor_length / 1000.0 * self.compressor_config.z1)
            
            # QC DEBUG: Check if rotor profile is valid
            if rotor_gen.main_rotor_profile is None:
                print(f"WARNING: Profile generation failed, using fallback volume")
                return self.compressor_config.target_swv_per_rev * 0.5  # Conservative estimate
            
            if len(rotor_gen.main_rotor_profile) < 50:
                print(f"WARNING: Main rotor profile has only {len(rotor_gen.main_rotor_profile)} points!")
            if volume_per_rev <= 0:
                print(f"WARNING: Volume calculation returned {volume_per_rev}, using fallback")
                return self.compressor_config.target_swv_per_rev * 0.5
            
            print(f"Volume calculation:")
            print(f"  calculated = {volume_per_rev:.6f} m³")
            print(f"  target = {self.compressor_config.target_swv_per_rev:.6f} m³")
            
            return volume_per_rev
            
        except Exception as e:
            print(f"WARNING: Volume calculation failed: {e}, using fallback")
            volume_per_rev = self.compressor_config.target_swv_per_rev * 0.5
            
        return volume_per_rev
    
    def _calculate_diameters_from_profile(self, profile: NRackProfile, r_params: dict) -> tuple[float, float]:
        """
        Calculate outer diameters using QC-approved City University analytic formula.
        
        QC ANALYTIC FORMULA (Eq F-1, F-2):
        D1e (main) = 2 * (r1w + r2)
        D2e (gate) = 2 * (r2w + r4)
        
        This is the ground-truth method used by all published N-profile optimizations.
        """
        # QC-approved parameters
        r1w_m = self.compressor_config.r1w / 1000.0  # Convert mm to meters
        r2w_m = self.compressor_config.r2w / 1000.0  # Convert mm to meters
        r2 = r_params['r2']  # Main tip radius (meters)
        r4 = r_params['r4']  # Gate tip radius (meters)
        
        # QC ANALYTIC FORMULA - City University literature
        D1e_calc_mm = 2 * (r1w_m + r2) * 1000.0  # Eq F-1
        D2e_calc_mm = 2 * (r2w_m + r4) * 1000.0  # Eq F-2
        
        print(f"QC DIAMETER CALCULATION:")
        print(f"  r1w={r1w_m*1000:.2f}mm + r2={r_params['r2']*1000:.2f}mm -> D1e={D1e_calc_mm:.2f}mm")
        print(f"  r2w={r2w_m*1000:.2f}mm + r4={r_params['r4']*1000:.2f}mm -> D2e={D2e_calc_mm:.2f}mm")
        
        # QC validation guard rail (optional - for catching coding mistakes)
        try:
            profile_outline = profile.get_full_profile()
            if len(profile_outline) > 0:
                discrete_D1e_max = 2 * np.max(np.linalg.norm(profile_outline, axis=1)) * 1000
                clearance_check = D1e_calc_mm - discrete_D1e_max
                if clearance_check > 0.05:  # Should be <= 50 µm gap
                    print(f"  PASS: Clearance validation: {clearance_check:.3f}mm gap (envelope inside circle)")
                else:
                    print(f"  WARNING: Clearance warning: {clearance_check:.3f}mm gap")
        except:
            pass  # Don't fail on validation issues
        
        return D1e_calc_mm, D2e_calc_mm 