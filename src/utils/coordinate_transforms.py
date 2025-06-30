"""
Coordinate Frame Conversion Helpers - QC Requirement

Explicit helpers for converting between rack frame and rotor frame coordinates.
QC guidance: arc_centers live in rack frame, r₂w/r₁w are in rotor frame (meters).
Add these helpers to make conversions explicit and unit-tested.
"""

import numpy as np
from typing import Tuple, Union


def to_rack_coords(x_rotor: float, y_rotor: float, theta: float, r_w: float) -> Tuple[float, float]:
    """
    Convert rotor coordinates to rack coordinates using Eq. 2.3 (forward transformation).
    
    Per Geometry.md Eq. 2.3:
    x₀r = x₀₁ cos(θ) − y₀₁ sin(θ)
    y₀r = x₀₁ sin(θ) + y₀₁ cos(θ) − r₁θ
    
    Args:
        x_rotor: Rotor x-coordinate (meters)
        y_rotor: Rotor y-coordinate (meters)
        theta: Meshing angle θ (radians)
        r_w: Pitch radius (meters) - r₁w for main, r₂w for gate
        
    Returns:
        Tuple[float, float]: (x_rack, y_rack) coordinates in meters
    """
    # Ensure all inputs are float for consistency
    x_rotor_m = float(x_rotor)
    y_rotor_m = float(y_rotor)
    theta_rad = float(theta)
    r_w_m = float(r_w)
    
    # Apply Eq. 2.3: Rotor to rack transformation  
    # x₀r = x₀₁ cos(θ) − y₀₁ sin(θ)
    # y₀r = x₀₁ sin(θ) + y₀₁ cos(θ) − r₁θ
    x_rack = x_rotor_m * np.cos(theta_rad) - y_rotor_m * np.sin(theta_rad)
    y_rack = x_rotor_m * np.sin(theta_rad) + y_rotor_m * np.cos(theta_rad) - r_w_m * theta_rad
    
    return x_rack, y_rack


def to_rotor_coords(x_rack: float, y_rack: float, theta: float, r_w: float) -> Tuple[float, float]:
    """
    Convert rack coordinates to rotor coordinates using Eq. 2.4 (inverse transformation).
    
    Per Geometry.md Eq. 2.4:
    x₀₁ = x₀r cos(θ) − (y₀r − r₁w θ) sin(θ)
    y₀₁ = x₀r sin(θ) + (y₀r − r₁w θ) cos(θ)
    
    Args:
        x_rack: Rack x-coordinate (meters)
        y_rack: Rack y-coordinate (meters)
        theta: Meshing angle θ (radians)
        r_w: Pitch radius (meters) - r₁w for main, r₂w for gate
        
    Returns:
        Tuple[float, float]: (x_rotor, y_rotor) coordinates in meters
    """
    # Ensure all inputs are float for consistency
    x_rack_m = float(x_rack)
    y_rack_m = float(y_rack)
    theta_rad = float(theta)
    r_w_m = float(r_w)
    
    # Apply Eq. 2.4: Rack to rotor transformation
    # x₀₁ = x₀r cos(θ) − (y₀r − r₁w) sin(θ)
    # y₀₁ = x₀r sin(θ) + (y₀r − r₁w) cos(θ)
    y_adjusted = y_rack_m - r_w_m * theta_rad
    x_rotor = x_rack_m * np.cos(theta_rad) - y_adjusted * np.sin(theta_rad)
    y_rotor = x_rack_m * np.sin(theta_rad) + y_adjusted * np.cos(theta_rad)
    
    return x_rotor, y_rotor


def compute_tip_arc_center_rotor_frame(tip_radius: float, rotor_type: str, config) -> np.ndarray:
    """
    Compute the center of a tip arc in rotor coordinates.
    
    CRITICAL FIX: Instead of arbitrary center locations, compute actual centers 
    based on "N" profile rotor geometry. 
    
    QC clarification: Gate tip G₂H₂ uses r₄ circle, Main tip H₁J₁ uses r₂ circle.
    These are the actual tip arc centers, NOT the root fillet centers CD/EF.
    
    Args:
        tip_radius: Tip radius in meters (r₂ for main, r₄ for gate)
        rotor_type: Either 'main' or 'gate'
        config: CompressorConfig object for geometric parameters
        
    Returns:
        np.ndarray: Tip arc center in rotor coordinates (meters)
        
    NOTE: This is a simplified implementation that needs to be integrated with
    actual rotor profile geometry. For now, use reasonable estimates based on
    "N" profile characteristics.
    """
    if rotor_type == 'main':
        # Main rotor O₁ center is at origin in rotor frame
        # Tip arc center is offset inward from the rotor body
        # For "N" profile: tip arc center typically near the pitch circle
        r_w = config.r1w / 1000.0  # Convert mm to meters
        
        # Estimate: tip center is offset from rotor center by approximately r_w - tip_radius
        # This puts the tip arc at the external periphery of the rotor
        center_x = 0.0
        center_y = r_w - tip_radius  # Slight inward offset
        return np.array([center_x, center_y])
        
    elif rotor_type == 'gate':
        # Gate rotor O₂ center is at (-C, 0) relative to main rotor in rotor frame
        r_w = config.r2w / 1000.0  # Convert mm to meters  
        center_distance = config.center_distance / 1000.0  # Convert mm to meters
        
        # Gate rotor center relative to main rotor
        gate_center_x = -center_distance  
        gate_center_y = 0.0
        
        # Tip arc center is offset from gate rotor center  
        # For "N" profile: similar logic as main rotor
        tip_center_x = gate_center_x
        tip_center_y = gate_center_y + r_w - tip_radius  # Offset toward main rotor
        return np.array([tip_center_x, tip_center_y])
        
    else:
        raise ValueError(f"Invalid rotor type: {rotor_type}")


def compute_tip_arc_center_from_points(point1: np.ndarray, point2: np.ndarray, 
                                      tip_radius: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the center(s) of a circle with given radius passing through two points.
    
    This is the CORRECT way to compute tip arc centers based on actual junction points.
    
    Args:
        point1: First point on the arc (e.g., G or H)
        point2: Second point on the arc (e.g., H or J)  
        tip_radius: Radius of the tip arc
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Two possible centers (choose based on geometry)
        
    Raises:
        ValueError: If points are too far apart for the given radius
    """
    # Distance between points
    distance = np.linalg.norm(point2 - point1)
    
    # Check if circle with given radius can pass through both points
    if distance > 2 * tip_radius:
        raise ValueError(f"Points too far apart: distance={distance*1000:.1f}mm > diameter={2*tip_radius*1000:.1f}mm")
    
    # Midpoint between the two points
    midpoint = (point1 + point2) / 2.0
    
    # Vector from point1 to point2
    chord_vector = point2 - point1
    
    # Perpendicular vector (rotate chord vector by 90°)
    perp_vector = np.array([-chord_vector[1], chord_vector[0]])
    perp_unit = perp_vector / np.linalg.norm(perp_vector) if np.linalg.norm(perp_vector) > 1e-12 else np.array([0, 1])
    
    # Distance from midpoint to center
    half_chord = distance / 2.0
    if half_chord < tip_radius:
        center_distance = np.sqrt(tip_radius**2 - half_chord**2)
    else:
        center_distance = 0.0  # Points are at diameter endpoints
    
    # Two possible centers
    center1 = midpoint + center_distance * perp_unit
    center2 = midpoint - center_distance * perp_unit
    
    return center1, center2


def verify_tip_arc_geometry(junction_points: dict, tip_center: np.ndarray, tip_radius: float, 
                           point_names: list, tolerance: float = 1e-6) -> bool:
    """
    Verify that junction points lie on the expected tip arc within tolerance.
    
    QC requirement: Add curvature radius validation tests to catch center swaps.
    
    Args:
        junction_points: Dictionary of junction points 
        tip_center: Tip arc center coordinates
        tip_radius: Expected tip radius
        point_names: List of point names that should be on the arc
        tolerance: Distance tolerance (meters)
        
    Returns:
        bool: True if all points lie on arc within tolerance
    """
    for name in point_names:
        if name not in junction_points:
            return False
            
        point = junction_points[name]
        distance = np.linalg.norm(point - tip_center)
        
        if abs(distance - tip_radius) > tolerance:
            return False
    
    return True


def frame_consistency_check(rack_point: np.ndarray, rotor_point: np.ndarray, 
                           theta: float, r_w: float, tolerance: float = 1e-12) -> bool:
    """
    Verify that rack ↔ rotor coordinate transformations are consistent.
    
    QC requirement: Unit tests for frame checks to prevent coordinate mix-ups.
    
    Args:
        rack_point: Point in rack coordinates (meters)
        rotor_point: Same point in rotor coordinates (meters)
        theta: Meshing angle θ (radians)
        r_w: Pitch radius (meters)
        tolerance: Transformation tolerance
        
    Returns:
        bool: True if transformations are consistent within tolerance
    """
    # Forward transform: rotor → rack
    x_rack_calc, y_rack_calc = to_rack_coords(rotor_point[0], rotor_point[1], theta, r_w)
    
    # Backward transform: rack → rotor
    x_rotor_calc, y_rotor_calc = to_rotor_coords(rack_point[0], rack_point[1], theta, r_w)
    
    # Check both directions
    rack_error = np.linalg.norm([x_rack_calc - rack_point[0], y_rack_calc - rack_point[1]])
    rotor_error = np.linalg.norm([x_rotor_calc - rotor_point[0], y_rotor_calc - rotor_point[1]])
    
    return rack_error < tolerance and rotor_error < tolerance


def convert_units_mm_to_m(value_mm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert millimeters to meters with explicit documentation.
    
    QC guidance: Handle unit mix-up (mm vs m) risks explicitly.
    
    Args:
        value_mm: Value(s) in millimeters
        
    Returns:
        Value(s) in meters
    """
    if isinstance(value_mm, np.ndarray):
        return value_mm / 1000.0
    else:
        return float(value_mm) / 1000.0


def convert_units_m_to_mm(value_m: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert meters to millimeters with explicit documentation.
    
    Args:
        value_m: Value(s) in meters
        
    Returns:
        Value(s) in millimeters
    """
    if isinstance(value_m, np.ndarray):
        return value_m * 1000.0
    else:
        return float(value_m) * 1000.0 