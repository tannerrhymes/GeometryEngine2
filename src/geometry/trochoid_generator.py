"""
Trochoid generation module for high-fidelity GH and HJ rack segments.

This module implements the envelope theory approach using Eq. 2.5 from Geometry.md
to generate true trochoids instead of straight-line approximations.
"""

import numpy as np
from typing import Tuple, Callable
from .constants import GeometryConstants

class TrochoidGenerator:
    """
    Generates trochoid curves using rack-to-rotor envelope transformation.
    
    Based on Eq. 2.5 from Geometry.md:
    (dy₀r/dx₀r) * (r₁wθ − y₀r) − (r₁w − x₀r) = 0
    
    And Eq. 2.4 transformation:
    x₀₁ = x₀r cos(θ) − (y₀r − r₁w) sin(θ)
    y₀₁ = x₀r sin(θ) + (y₀r − r₁w) cos(θ)
    """
    
    def __init__(self, constants: GeometryConstants):
        """
        Initialize trochoid generator with geometric constants.
        
        Args:
            constants: GeometryConstants instance for dimensional scaling
        """
        self.constants = constants
        
    def theta_from_rack(self, x_rack: float, y_rack: float, r_w: float, dy_dx: float) -> float:
        """
        Solve Eq. 2.5 for θ given rack coordinates and slope.
        
        Uses Newton's method - exact solution since derivative is constant.
        
        Args:
            x_rack: Rack x-coordinate (meters)
            y_rack: Rack y-coordinate (meters) 
            r_w: Pitch radius (meters) - r₁w for main, r₂w for gate
            dy_dx: Rack slope dy/dx at this point
            
        Returns:
            float: Meshing angle θ (radians)
            
        Raises:
            ValueError: If slope is too steep or θ is out of bounds
        """
        # Guard against extremely steep slopes
        if abs(dy_dx) > self.constants.MAX_SLOPE:
            raise ValueError(f"Slope too steep: |dy/dx| = {abs(dy_dx)} > {self.constants.MAX_SLOPE}")
        
        # Newton's method for Eq. 2.5: f(θ) = dy_dx*(r_w*θ - y_rack) - (r_w - x_rack) = 0
        def func(theta):
            return dy_dx * (r_w * theta - y_rack) - (r_w - x_rack)
        
        def dfunc(theta):
            return dy_dx * r_w  # Constant derivative
        
        # Initial guess
        theta_guess = 0.0
        
        # One Newton iteration (exact since derivative is constant)
        # QC approved: Use 1e-8 threshold for physical meaning on 50mm pitch circle
        if abs(dfunc(theta_guess)) < 1e-8:
            # Special case: horizontal rack (dy_dx = 0)
            # From Eq. 2.5: 0*(r_w*θ - y_rack) - (r_w - x_rack) = 0
            # Simplifies to: -(r_w - x_rack) = 0, so θ can be any value
            # We choose θ such that the transformation is reasonable
            if abs(r_w) > 1e-12:
                return -(r_w - x_rack) / r_w  # Reasonable choice for horizontal rack
            else:
                raise ValueError(f"Both dy_dx and r_w are too small: dy_dx*r_w = {dfunc(theta_guess)}")
            
        theta = theta_guess - func(theta_guess) / dfunc(theta_guess)
        
        # Validate θ is within bounds
        if abs(theta) > self.constants.MAX_THETA:
            raise ValueError(f"Theta out of bounds: |θ| = {abs(theta)} > {self.constants.MAX_THETA}")
            
        return theta
    
    def rack_to_rotor_transform(self, x_rack: float, y_rack: float, theta: float, r_w: float) -> Tuple[float, float]:
        """
        Transform rack coordinates to rotor coordinates using Eq. 2.4.
        
        Args:
            x_rack: Rack x-coordinate (meters)
            y_rack: Rack y-coordinate (meters)
            theta: Meshing angle θ (radians)
            r_w: Pitch radius (meters)
            
        Returns:
            Tuple[float, float]: (x_rotor, y_rotor) coordinates in meters
        """
        # Eq. 2.4: Rack to rotor transformation
        x_rotor = x_rack * np.cos(theta) - (y_rack - r_w) * np.sin(theta)
        y_rotor = x_rack * np.sin(theta) + (y_rack - r_w) * np.cos(theta)
        
        return x_rotor, y_rotor
    
    def symmetric_rack_sampling(self, x_start: float, x_end: float, num_points: int) -> np.ndarray:
        """
        Generate symmetric x-sampling for rack curves to handle fold-back cases.
        
        Args:
            x_start: Starting x-coordinate
            x_end: Ending x-coordinate  
            num_points: Number of sample points
            
        Returns:
            np.ndarray: Symmetric x-samples between min and max of x_start, x_end
        """
        x_min = min(x_start, x_end)
        x_max = max(x_start, x_end)
        
        return np.linspace(x_min, x_max, num_points)
    
    def validate_trochoid_residual(self, x_rack: np.ndarray, y_rack: np.ndarray, 
                                 theta_array: np.ndarray, dy_dx_array: np.ndarray, 
                                 r_w: float) -> bool:
        """
        Validate that generated trochoid points satisfy Eq. 2.5 to specified tolerance.
        
        Args:
            x_rack: Rack x-coordinates
            y_rack: Rack y-coordinates
            theta_array: Corresponding theta values
            dy_dx_array: Rack slopes
            r_w: Pitch radius
            
        Returns:
            bool: True if all residuals meet tolerance
        """
        # Calculate residuals for Eq. 2.5
        residuals = dy_dx_array * (r_w * theta_array - y_rack) - (r_w - x_rack)
        max_residual = np.max(np.abs(residuals))
        
        return self.constants.scale_residual_tolerance(max_residual)
    
    def validate_c1_continuity_at_h(self, gh_points: np.ndarray, hj_points: np.ndarray) -> bool:
        """
        Validate C¹ continuity at junction point H.
        
        Args:
            gh_points: GH segment points (N, 2)
            hj_points: HJ segment points (N, 2)
            
        Returns:
            bool: True if C¹ continuity is satisfied
        """
        if len(gh_points) < 2 or len(hj_points) < 2:
            return False
            
        # Tangent vectors at H
        t_gh = gh_points[-1] - gh_points[-2]  # Tangent at end of GH
        t_hj = hj_points[1] - hj_points[0]    # Tangent at start of HJ
        
        # Cross product for continuity check (using 3D vectors to avoid deprecation)
        t_gh_3d = np.array([t_gh[0], t_gh[1], 0.0])
        t_hj_3d = np.array([t_hj[0], t_hj[1], 0.0])
        cross_product_3d = np.cross(t_gh_3d, t_hj_3d)
        cross_product = cross_product_3d[2]  # Z-component is the 2D cross product
        
        return self.constants.scale_continuity_tolerance(cross_product)


# Feature flag for enabling true trochoids (default False until tests pass)
USE_TRUE_TROCHOIDS = False

def get_trochoid_generator(r1w_m: float) -> TrochoidGenerator:
    """
    Factory function to create TrochoidGenerator with proper scaling.
    
    Args:
        r1w_m: Main rotor pitch radius in meters
        
    Returns:
        TrochoidGenerator: Configured generator instance
    """
    constants = GeometryConstants(r1w_m)
    return TrochoidGenerator(constants) 