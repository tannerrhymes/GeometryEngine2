"""
Geometric constants and dimensional scaling for the rack profile generator.

This module provides consistent dimensional scaling to ensure numerical stability
and dimensionless tolerances across all geometric calculations.
"""

import numpy as np

# Unit length scale - set to main rotor pitch radius for dimensional consistency
# This will be updated dynamically based on the actual configuration
L0_DEFAULT = 0.05  # 50mm default pitch radius in meters

class GeometryConstants:
    """Container for geometry-related constants and scaling factors."""
    
    def __init__(self, r1w_m: float):
        """
        Initialize geometry constants based on main rotor pitch radius.
        
        Args:
            r1w_m (float): Main rotor pitch radius in meters
        """
        self.L0 = r1w_m  # Unit length scale
        
        # Tolerance constants (dimensionless, scaled by L0)
        self.RESIDUAL_TOL = 1e-8  # For Eq. 2.5 residual validation
        self.CONTINUITY_TOL = 1e-6  # For C¹ continuity at junction H
        self.THETA_TOL = 1e-10  # For theta convergence in Newton solver
        
        # Numerical safety limits
        self.MAX_SLOPE = 1e3  # Maximum allowed |dy/dx| before penalty
        self.MAX_THETA = np.pi  # Theta must stay within [-π, π]
        self.MIN_POINTS = 50  # Minimum points for trochoid visualization
        
        # Penalty values for constraint violations
        self.SLOPE_PENALTY = 1e6
        self.THETA_PENALTY = 1e6
    
    def scale_residual_tolerance(self, residual: float) -> bool:
        """Check if residual meets scaled tolerance."""
        return abs(residual) < self.RESIDUAL_TOL * self.L0
    
    def scale_continuity_tolerance(self, cross_product: float) -> bool:
        """Check if C¹ continuity meets scaled tolerance."""
        return abs(cross_product) < self.CONTINUITY_TOL * self.L0**2

# Default instance for backward compatibility
DEFAULT_CONSTANTS = GeometryConstants(L0_DEFAULT) 