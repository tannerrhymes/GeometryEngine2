from dataclasses import dataclass

@dataclass
class OptimizationConfig:
    """Configuration for optimization algorithm and parameters."""
    
    # Sprint 3A QC-approved algorithm settings
    algorithm: str = 'L-BFGS-B'          # Primary algorithm for Phase 3A
    fallback_algorithm: str = 'Powell'    # Manual fallback (not auto-triggered)
    maxiter: int = 1000                   # Maximum iterations
    tolerance: float = 1e-6               # Convergence tolerance
    max_function_evaluations: int = 100   # Early abort threshold
    
    # QC-approved objective function weights
    weight_volume: float = 1.0            # w_V = 1 (dimensionless volume error)
    weight_D1e: float = 10.0              # w_D1 = 10/mm² (diameter error in mm)
    weight_D2e: float = 10.0              # w_D2 = 10/mm² (diameter error in mm)
    
    # QC-approved parameter bounds (as ratios of r1)
    r0_ratio_bounds: tuple = (0.02, 0.08)    # r0 ∈ [0.02*r1, 0.08*r1]
    r2_ratio_bounds: tuple = (0.13, 0.30)    # r2 ∈ [0.13*r1, 0.30*r1]  
    r3_ratio_bounds: tuple = (0.08, 0.20)    # r3 ∈ [0.08*r1, 0.20*r1]
    # Note: r4 = r3 (hard constraint)
    
    # Sprint 3A success criteria
    target_volume_error: float = 0.05     # 5% volume error threshold
    target_diameter_error: float = 0.3    # 0.3mm diameter error threshold
    
    # Performance settings
    gradient_scaling: bool = True         # Normalize variables by r1 for L-BFGS-B
    early_abort_threshold: float = 0.30   # Abort if volume error > 30%
    initial_guess_margin: float = 0.10    # Start within 10% of CSV values 