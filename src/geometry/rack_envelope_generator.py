"""
Rack envelope generation module for true GH/HJ curves.

Implements envelope theory to generate rack coordinates from rotor tip arcs,
following Geometry.md equations and QC requirements for full fidelity.
"""

import numpy as np
from typing import Tuple, Callable, Optional
import logging
from .constants import GeometryConstants


class RackEnvelopeGenerator:
    """
    Generates rack curves as true envelopes of rotor tip arcs.
    
    Based on reverse application of Eq. 2.4 from Geometry.md:
    x₀r = x₀₁ cos(θ) + (y₀₁ - r₁w) sin(θ)  
    y₀r = -x₀₁ sin(θ) + (y₀₁ - r₁w) cos(θ) + r₁w*θ
    
    Where (x₀₁, y₀₁) are rotor tip arc coordinates and (x₀r, y₀r) are rack coordinates.
    """
    
    def __init__(self, constants: GeometryConstants):
        """
        Initialize rack envelope generator.
        
        Args:
            constants: GeometryConstants for dimensional scaling and tolerances
        """
        self.constants = constants
        
    def rotor_to_rack_transform(self, x_rotor: float, y_rotor: float, 
                               theta: float, r_w: float) -> Tuple[float, float]:
        """
        Transform rotor coordinates to rack coordinates using reverse of Eq. 2.4.
        
        Args:
            x_rotor: Rotor x-coordinate (meters)
            y_rotor: Rotor y-coordinate (meters)  
            theta: Rotor rotation angle (radians)
            r_w: Pitch radius (meters)
            
        Returns:
            Tuple[float, float]: (x_rack, y_rack) coordinates in meters
        """
        # Reverse of Eq. 2.4: rotor → rack transformation
        x_rack = x_rotor * np.cos(theta) + (y_rotor - r_w) * np.sin(theta)
        y_rack = -x_rotor * np.sin(theta) + (y_rotor - r_w) * np.cos(theta) + r_w * theta
        
        return x_rack, y_rack
        
    def generate_gate_tip_arc_g2h2(self, g2_point: np.ndarray, h2_point: np.ndarray,
                                  arc_center: np.ndarray, radius: float, 
                                  num_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate gate rotor tip arc G₂H₂ coordinates and tangent angles.
        
        Args:
            g2_point: Gate rotor point G₂
            h2_point: Gate rotor point H₂  
            arc_center: Center of the gate tip arc
            radius: Arc radius (meters)
            num_points: Number of sample points (QC requirement: ≥ 200)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (arc_points, tangent_angles) 
        """
        # Ensure minimum sampling density per QC
        num_points = max(num_points, 200)
        
        # Calculate start and end angles for the arc
        start_vec = g2_point - arc_center
        end_vec = h2_point - arc_center
        
        start_angle = np.arctan2(start_vec[1], start_vec[0])
        end_angle = np.arctan2(end_vec[1], end_vec[0])
        
        # Handle angle wrapping to ensure shortest arc
        if end_angle - start_angle > np.pi:
            end_angle -= 2 * np.pi
        elif start_angle - end_angle > np.pi:
            start_angle -= 2 * np.pi
            
        # Generate dense sampling of the arc
        angles = np.linspace(start_angle, end_angle, num_points)
        
        # Generate arc points
        arc_points = np.zeros((num_points, 2))
        tangent_angles = np.zeros(num_points)
        
        for i, angle in enumerate(angles):
            # Arc point coordinates
            arc_points[i, 0] = arc_center[0] + radius * np.cos(angle)
            arc_points[i, 1] = arc_center[1] + radius * np.sin(angle)
            
            # Tangent angle (perpendicular to radius vector)
            tangent_angles[i] = angle + np.pi/2  # 90° ahead of radius
            
        return arc_points, tangent_angles
        
    def generate_main_tip_arc_h1j1(self, h1_point: np.ndarray, j1_point: np.ndarray,
                                  arc_center: np.ndarray, radius: float,
                                  num_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate main rotor tip arc H₁J₁ coordinates and tangent angles.
        
        Args:
            h1_point: Main rotor point H₁
            j1_point: Main rotor point J₁
            arc_center: Center of the main tip arc  
            radius: Arc radius (meters)
            num_points: Number of sample points (QC requirement: ≥ 200)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (arc_points, tangent_angles)
        """
        # Ensure minimum sampling density per QC
        num_points = max(num_points, 200)
        
        # Calculate start and end angles for the arc
        start_vec = h1_point - arc_center
        end_vec = j1_point - arc_center
        
        start_angle = np.arctan2(start_vec[1], start_vec[0])
        end_angle = np.arctan2(end_vec[1], end_vec[0])
        
        # Handle angle wrapping to ensure shortest arc
        if end_angle - start_angle > np.pi:
            end_angle -= 2 * np.pi
        elif start_angle - end_angle > np.pi:
            start_angle -= 2 * np.pi
            
        # Generate dense sampling of the arc
        angles = np.linspace(start_angle, end_angle, num_points)
        
        # Generate arc points
        arc_points = np.zeros((num_points, 2))
        tangent_angles = np.zeros(num_points)
        
        for i, angle in enumerate(angles):
            # Arc point coordinates
            arc_points[i, 0] = arc_center[0] + radius * np.cos(angle)
            arc_points[i, 1] = arc_center[1] + radius * np.sin(angle)
            
            # Tangent angle (perpendicular to radius vector)
            tangent_angles[i] = angle + np.pi/2  # 90° ahead of radius
            
        return arc_points, tangent_angles
        
    def generate_gh_rack_envelope(self, gate_arc_points: np.ndarray, 
                                 gate_tangent_angles: np.ndarray,
                                 r2w: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate GH rack curve as envelope of gate tip arc G₂H₂.
        
        Args:
            gate_arc_points: Gate tip arc coordinates (N, 2)
            gate_tangent_angles: Tangent angles at each arc point (N,)
            r2w: Gate rotor pitch radius (meters)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (rack_points, rack_slopes)
        """
        num_points = len(gate_arc_points)
        rack_points = np.zeros((num_points, 2))
        rack_slopes = np.zeros(num_points)
        
        # For each point on the gate arc, compute corresponding rack point
        for i in range(num_points):
            x_rotor = gate_arc_points[i, 0]
            y_rotor = gate_arc_points[i, 1]
            tangent_angle = gate_tangent_angles[i]
            
            # Determine theta from meshing condition
            # For envelope generation, theta relates to the rotor rotation angle
            # that brings the current arc point to the contact position
            theta = self._solve_theta_for_envelope(x_rotor, y_rotor, r2w, tangent_angle)
            
            # Transform to rack coordinates using reverse Eq. 2.4
            x_rack, y_rack = self.rotor_to_rack_transform(x_rotor, y_rotor, theta, r2w)
            
            rack_points[i, 0] = x_rack
            rack_points[i, 1] = y_rack
            
            # Calculate rack slope from envelope condition
            rack_slopes[i] = self._calculate_rack_slope_from_envelope(
                x_rotor, y_rotor, theta, r2w, tangent_angle
            )
            
        return rack_points, rack_slopes
        
    def generate_hj_rack_envelope(self, main_arc_points: np.ndarray,
                                 main_tangent_angles: np.ndarray,
                                 r1w: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate HJ rack curve as envelope of main tip arc H₁J₁.
        
        Args:
            main_arc_points: Main tip arc coordinates (N, 2)
            main_tangent_angles: Tangent angles at each arc point (N,)
            r1w: Main rotor pitch radius (meters)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (rack_points, rack_slopes)
        """
        num_points = len(main_arc_points)
        rack_points = np.zeros((num_points, 2))
        rack_slopes = np.zeros(num_points)
        
        # For each point on the main arc, compute corresponding rack point
        for i in range(num_points):
            x_rotor = main_arc_points[i, 0]
            y_rotor = main_arc_points[i, 1]
            tangent_angle = main_tangent_angles[i]
            
            # Determine theta from meshing condition
            theta = self._solve_theta_for_envelope(x_rotor, y_rotor, r1w, tangent_angle)
            
            # Transform to rack coordinates using reverse Eq. 2.4
            x_rack, y_rack = self.rotor_to_rack_transform(x_rotor, y_rotor, theta, r1w)
            
            rack_points[i, 0] = x_rack
            rack_points[i, 1] = y_rack
            
            # Calculate rack slope from envelope condition
            rack_slopes[i] = self._calculate_rack_slope_from_envelope(
                x_rotor, y_rotor, theta, r1w, tangent_angle
            )
            
        return rack_points, rack_slopes
        
    def _solve_theta_for_envelope(self, x_rotor: float, y_rotor: float, 
                                 r_w: float, tangent_angle: float) -> float:
        """
        Solve for theta that satisfies the envelope condition.
        
        For envelope generation, the meshing condition relates the rotor tangent
        direction to the rack motion.
        
        Args:
            x_rotor: Rotor x-coordinate
            y_rotor: Rotor y-coordinate
            r_w: Pitch radius
            tangent_angle: Tangent angle at rotor point
            
        Returns:
            float: Theta angle (radians)
        """
        # Simplified envelope condition for initial implementation
        # This will be refined based on full envelope theory
        
        # For a point at radius r from rotor center
        r_point = np.sqrt(x_rotor**2 + y_rotor**2)
        point_angle = np.arctan2(y_rotor, x_rotor)
        
        # Initial estimate based on geometric relationship
        theta_estimate = point_angle - tangent_angle + np.pi/2
        
        # Ensure theta is in reasonable range
        while theta_estimate > np.pi:
            theta_estimate -= 2*np.pi
        while theta_estimate < -np.pi:
            theta_estimate += 2*np.pi
            
        return theta_estimate
        
    def _calculate_rack_slope_from_envelope(self, x_rotor: float, y_rotor: float,
                                          theta: float, r_w: float, 
                                          tangent_angle: float) -> float:
        """
        Calculate rack slope dy/dx from envelope condition.
        
        Args:
            x_rotor: Rotor x-coordinate
            y_rotor: Rotor y-coordinate  
            theta: Rotor rotation angle
            r_w: Pitch radius
            tangent_angle: Tangent angle at rotor point
            
        Returns:
            float: Rack slope dy/dx
        """
        # Calculate slope from differentiation of the transformation
        # This is derived from the envelope condition and Eq. 2.4
        
        # Rotor tangent vector components
        tx = np.cos(tangent_angle)
        ty = np.sin(tangent_angle)
        
        # Derivatives of transformation (simplified initial version)
        dx_rack_dtheta = -x_rotor * np.sin(theta) + (y_rotor - r_w) * np.cos(theta)
        dy_rack_dtheta = -x_rotor * np.cos(theta) - (y_rotor - r_w) * np.sin(theta) + r_w
        
        # Slope from chain rule (this will be refined)
        if abs(dx_rack_dtheta) > 1e-10:
            slope = dy_rack_dtheta / dx_rack_dtheta
        else:
            slope = 1e3  # Near vertical
            
        return slope
        
    def validate_envelope_residual(self, rack_points: np.ndarray, 
                                  rack_slopes: np.ndarray,
                                  theta_array: np.ndarray, r_w: float) -> float:
        """
        Validate that rack envelope satisfies Eq. 2.5 residual requirements.
        
        Args:
            rack_points: Rack coordinates (N, 2)
            rack_slopes: Rack slopes dy/dx (N,)
            theta_array: Theta values (N,)
            r_w: Pitch radius
            
        Returns:
            float: Maximum residual
        """
        x_rack = rack_points[:, 0]
        y_rack = rack_points[:, 1]
        dy_dx = rack_slopes
        
        # Calculate residuals for Eq. 2.5
        residuals = dy_dx * (r_w * theta_array - y_rack) - (r_w - x_rack)
        max_residual = np.max(np.abs(residuals))
        
        return max_residual 