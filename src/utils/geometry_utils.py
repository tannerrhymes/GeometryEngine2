import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import curve_fit

def find_tangent_line_between_curves(curve1_func, curve2_func, initial_guess):
    """
    Finds the common tangent line between two curves defined by functions.
    This is a placeholder for a more complex geometric solver.
    For Sprint 1, we will implement a simplified version.

    Args:
        curve1_func: A function that defines the first curve.
        curve2_func: A function that defines the second curve.
        initial_guess: An initial guess for the solver.

    Returns:
        A tuple containing the points of tangency on each curve.
    """
    # This is a complex problem. For Sprint 1, we will continue with the
    # simplified approach and focus on making the existing connections robust
    # before adding this layer of complexity.
    # A full implementation would involve solving a system of nonlinear equations
    # representing the tangency conditions.
    
    # Placeholder returns simplified points for now.
    # This logic will be improved in subsequent steps.
    pass

def solve_generalized_arc(start_point: np.ndarray, end_point: np.ndarray, p_exp: float, q_exp: float, num_points: int=20, a: float=None, b: float=None) -> np.ndarray:
    """
    Generates points for a generalized arc of the form a*x^p + b*y^q = 1
    that passes through the start and end points.

    If coefficients a and b are provided, it uses them directly. Otherwise,
    it solves for them.
    """
    # If a or b are not provided or are NaN, solve for them
    if a is None or b is None or np.isnan(a) or np.isnan(b):
        # Ensure points are not on the axes, which would make the system singular
        if np.any(np.isclose(start_point, 0)) or np.any(np.isclose(end_point, 0)):
             # Cannot solve if points are on the axes, return a straight line as fallback
            return np.linspace(start_point, end_point, num_points)

        # Set up the system of linear equations to find 'a' and 'b'
        A_matrix = np.array([
            [start_point[0]**p_exp, start_point[1]**q_exp],
            [end_point[0]**p_exp, end_point[1]**q_exp]
        ])
        B_vector = np.array([1, 1])

        try:
            # Solve for the coefficients a and b
            coeffs = np.linalg.solve(A_matrix, B_vector)
            a, b = coeffs[0], coeffs[1]
        except np.linalg.LinAlgError:
            # Matrix is singular, cannot solve. Fallback to a straight line.
            return np.linspace(start_point, end_point, num_points)

    # Generate points along the x-axis
    x_points = np.linspace(start_point[0], end_point[0], num_points)
    
    # Calculate the corresponding y-points
    # Use np.complex128 to handle potential negative numbers inside the root
    with np.errstate(invalid='ignore'): # Ignore warnings about invalid values
        inner_term = (1 - a * x_points**p_exp) / b
        # Ensure we don't take the root of a negative number if b is negative
        inner_term[inner_term < 0] = 0
        y_points = np.power(inner_term, 1/q_exp)

    return np.vstack([x_points, y_points]).T

def find_circle_circle_tangent_points(center1, r1, center2, r2):
    """
    Finds the four points that define the two external tangent lines between two circles.
    Returns the points on circle 1 and circle 2 for both tangent lines.
    This is a direct geometric solution.
    """
    d = np.linalg.norm(center2 - center1)
    if d <= abs(r1 - r2):
        # One circle is inside another, no external tangents
        return None

    # Angle of the line between centers
    gamma = np.arctan2(center2[1] - center1[1], center2[0] - center1[0])
    # Angle for tangent points relative to the center line
    beta = np.arccos((r1 - r2) / d)

    # Tangent points on circle 1
    t1_p1 = center1 + r1 * np.array([np.cos(gamma + beta), np.sin(gamma + beta)])
    t1_p2 = center1 + r1 * np.array([np.cos(gamma - beta), np.sin(gamma - beta)])

    # Tangent points on circle 2
    t2_p1 = center2 + r2 * np.array([np.cos(gamma + beta), np.sin(gamma + beta)])
    t2_p2 = center2 + r2 * np.array([np.cos(gamma - beta), np.sin(gamma - beta)])
    
    return ((t1_p1, t2_p1), (t1_p2, t2_p2))

def generate_circular_arc(center: np.ndarray, radius: float, start_point: np.ndarray, end_point: np.ndarray, num_points: int = 20) -> np.ndarray:
    """
    Generates points for a circular arc defined by a center, radius, and
    start/end points.

    Args:
        center (np.ndarray): The [x, y] coordinates of the circle's center.
        radius (float): The circle's radius.
        start_point (np.ndarray): The starting [x, y] coordinates on the arc.
        end_point (np.ndarray): The ending [x, y] coordinates on the arc.
        num_points (int): The number of points to generate for the segment.

    Returns:
        np.ndarray: A (num_points, 2) array of coordinates for the arc.
    """
    # Calculate the angles of the start and end points from the center
    start_angle = np.arctan2(start_point[1] - center[1], start_point[0] - center[0])
    end_angle = np.arctan2(end_point[1] - center[1], end_point[0] - center[0])

    # Handle the case where the arc crosses the -pi/pi boundary
    if end_angle < start_angle:
        end_angle += 2 * np.pi

    # Generate angles for the arc
    angles = np.linspace(start_angle, end_angle, num_points)
    
    # Calculate the x and y coordinates
    x_points = center[0] + radius * np.cos(angles)
    y_points = center[1] + radius * np.sin(angles)

    return np.vstack([x_points, y_points]).T

def generate_trochoid_segment(start_point: np.ndarray, end_point: np.ndarray, num_points: int = 20) -> np.ndarray:
    """
    Placeholder for the trochoid generation function.
    The actual implementation will require the full conjugate motion solution.
    For now, it returns a straight line.
    """
    # TODO: Implement the full trochoid generation based on envelope theory.
    # This will involve solving the meshing equations for the generating rotor arc.
    return np.linspace(start_point, end_point, num_points)

def find_circle_center_tangent_to_line(tangent_point: np.ndarray, line_direction: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Finds the two possible centers of a circle with a given radius that is
    tangent to a line at a specific point.

    The line is defined by the tangent point and a direction vector.

    Args:
        tangent_point (np.ndarray): The [x, y] coordinates of the point of tangency.
        line_direction (np.ndarray): A vector defining the direction of the line.
        radius (float): The radius of the circle.

    Returns:
        A tuple containing the two possible [x, y] center coordinates.
    """
    # The vector to the center is perpendicular to the line's direction vector.
    perp_vector = np.array([-line_direction[1], line_direction[0]])
    
    # Normalize the perpendicular vector to get a unit vector
    norm_perp_vector = perp_vector / np.linalg.norm(perp_vector)
    
    # The two possible centers are found by moving from the tangent point
    # along the normalized perpendicular vector by the radius distance.
    center1 = tangent_point + radius * norm_perp_vector
    center2 = tangent_point - radius * norm_perp_vector
    
    return (center1, center2)

def solve_circle_circle_tangency(center1: np.ndarray, radius1: float, center2: np.ndarray, radius2: float) -> np.ndarray:
    """
    Calculates the point of tangency between two circles that are touching.

    Args:
        center1 (np.ndarray): Center of the first circle.
        radius1 (float): Radius of the first circle.
        center2 (np.ndarray): Center of the second circle.
        radius2 (float): Radius of the second circle.

    Returns:
        np.ndarray: The [x, y] coordinates of the point of tangency.
                    Returns None if the circles do not touch at a single point.
    """
    d = np.linalg.norm(center2 - center1)
    
    # Check if circles are externally or internally tangent
    if not np.isclose(d, radius1 + radius2) and not np.isclose(d, abs(radius1 - radius2)):
        # Circles do not touch at a single point
        return None
        
    # The tangent point lies on the line connecting the two centers.
    # We can find it by moving from center1 towards center2 by radius1 distance.
    tangent_point = center1 + radius1 * (center2 - center1) / d
    
    return tangent_point

def find_circle_tangent_to_two_lines(line1_point: np.ndarray, line1_dir: np.ndarray, line2_point: np.ndarray, line2_dir: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds the center and tangent points of a circle that is tangent to two lines.

    Args:
        line1_point, line2_point (np.ndarray): A point on each line.
        line1_dir, line2_dir (np.ndarray): The direction vector for each line.
        radius (float): The radius of the tangent circle.

    Returns:
        A tuple containing:
        - The center of the circle.
        - The point of tangency on line 1.
        - The point of tangency on line 2.
        Returns None if lines are parallel.
    """
    # Normalize direction vectors
    v1 = line1_dir / np.linalg.norm(line1_dir)
    v2 = line2_dir / np.linalg.norm(line2_dir)

    # Check for parallel lines
    if np.isclose(np.abs(np.dot(v1, v2)), 1.0):
        return None, None, None

    # Find intersection of the two lines
    A = np.array([[v1[0], -v2[0]], [v1[1], -v2[1]]])
    b = line2_point - line1_point
    try:
        t, _ = np.linalg.solve(A, b)
        intersection_point = line1_point + t * v1
    except np.linalg.LinAlgError:
        return None, None, None # Parallel or coincident

    # Angle between the direction vectors
    cos_theta = np.dot(v1, v2)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    # Distance from intersection to circle center along the bisector
    dist_to_center = radius / np.sin(angle / 2.0)

    # Direction vector of the angle bisector
    bisector_dir = (v1 + v2) / np.linalg.norm(v1 + v2)
    
    # The center is found by moving from the intersection along the bisector.
    # We must choose the correct one of two possible centers.
    center1 = intersection_point + dist_to_center * bisector_dir
    center2 = intersection_point - dist_to_center * bisector_dir

    # A robust check to select the center "inside" the corner.
    # We assume the origin (0,0) is inside the corner for our geometry.
    if np.linalg.norm(center1) < np.linalg.norm(center2):
        center = center1
    else:
        center = center2

    # Find tangent points by projecting the center onto each line
    t1 = np.dot(center - line1_point, v1)
    tangent_point1 = line1_point + t1 * v1

    t2 = np.dot(center - line2_point, v2)
    tangent_point2 = line2_point + t2 * v2
    
    return center, tangent_point1, tangent_point2

def generate_rack_from_rotor_envelope(rotor_profile: np.ndarray, r_w: float, dydx_rotor: np.ndarray) -> np.ndarray:
    """
    Generates a rack profile from a rotor profile using envelope theory.
    This is the inverse of the operation in RotorGenerator.

    Args:
        rotor_profile (np.ndarray): Nx2 array of [x, y] coordinates for the rotor.
        r_w (float): Pitch radius of the rotor (in meters).
        dydx_rotor (np.ndarray): Gradient of the rotor profile.

    Returns:
        np.ndarray: A Nx2 array of [x, y] coordinates for the rack profile.
    """
    num_points = len(rotor_profile)
    rack_coords = np.zeros((num_points, 2))
    x_rotor, y_rotor = rotor_profile[:, 0], rotor_profile[:, 1]

    for j in range(num_points):
        def meshing_equation(theta):
            # This is the rearranged meshing equation to solve for the rack point
            # that corresponds to a given rotor point at a meshing angle theta.
            # It enforces that the normal to the profiles is perpendicular to the relative velocity.
            # (dy_dr/dx_r) * (r_w*theta - y_r) - (r_w - x_r) = 0
            # Since the rack is generated from the rotor, the logic is slightly different.
            # We use the rotor's gradient to find the angle.
            # Simplified for now, a full derivation is more complex.
            # Placeholder returns a simplified relation.
            return dydx_rotor[j] * (x_rotor[j] * np.sin(theta) + y_rotor[j] * np.cos(theta)) - \
                   (x_rotor[j] * np.cos(theta) - y_rotor[j] * np.sin(theta))

        try:
            theta_solution, = fsolve(meshing_equation, 0.0)
            
            # Inverse coordinate transformation
            x_rack = x_rotor[j] * np.cos(-theta_solution) + y_rotor[j] * np.sin(-theta_solution)
            y_rack = -x_rotor[j] * np.sin(-theta_solution) + y_rotor[j] * np.cos(-theta_solution) + r_w

            rack_coords[j] = [x_rack, y_rack]
        except Exception:
            rack_coords[j] = [np.nan, np.nan]

    return rack_coords[~np.isnan(rack_coords).any(axis=1)]

def fit_general_arc(points: np.ndarray, p: float, q: float) -> tuple[float, float]:
    """
    Fits a set of 2D points to the general arc equation a*x^p + b*y^q = 1.

    Args:
        points (np.ndarray): The Nx2 array of [x, y] coordinates to fit.
        p (float): The exponent for the x term.
        q (float): The exponent for the y term.

    Returns:
        A tuple containing the solved coefficients (a, b).
    """
    x_data = points[:, 0]
    y_data = points[:, 1]

    # Define the function to be fitted.
    # We are solving for y in terms of x, a, and b.
    def arc_func(x, a, b):
        # We need to handle the case where the term inside the root is negative.
        # This can happen with noisy data or a poor fit.
        # We return a large number to penalize such solutions.
        inner_term = (1 - a * x**p) / b
        # Ensure we don't take the root of a negative number
        inner_term[inner_term < 0] = 0
        return np.power(inner_term, 1/q)

    # Use curve_fit to find the best values for a and b.
    # Provide an initial guess for the solver to start.
    initial_guess = [1.0, 1.0]
    try:
        popt, _ = curve_fit(arc_func, x_data, y_data, p0=initial_guess)
        return popt[0], popt[1]
    except (RuntimeError, ValueError):
        # If curve fitting fails, return non-physical values
        return np.nan, np.nan

def phi_from_rack(x_rack: float, y_rack: float, r_w: float) -> float:
    """
    Inverts the rack-to-rotor transformation to find the rotor angle (phi)
    that corresponds to a given point on the rack. This is done by solving
    the transformation equations for phi using a robust analytical method.
    """
    # The transformation is a simple rotation of a coordinate system.
    # x_rack = x_rotor*cos(-phi) + (y_rotor-r_w)*sin(-phi)
    # y_rack = -x_rotor*sin(-phi) + (y_rotor-r_w)*cos(-phi)
    # We are solving for the angle phi that would rotate the rack point
    # back to the rotor's coordinate system.
    
    # Let y' = y_rack - r_w
    # Let a point on the rotor be (xr, yr).
    # x_rack = xr * cos(phi) + yr * sin(phi)
    # y'     = -xr * sin(phi) + yr * cos(phi)
    # This can be solved directly for phi if (xr, yr) is known.
    # However, we don't know the point on the rotor, only the rack point.
    
    # The original fsolve approach was flawed. A direct analytical solution is best.
    # The system represents a rotation. We can find the angle using arctan2.
    # The vector from the pitch line (y=r_w) to the rack point is (x_rack, y_rack - r_w).
    # The length of this vector should correspond to the radius of the generating point
    # on the rotor, but we are trying to find the angle of rotation *phi*.
    # Let the point on the rotor be (x_r, y_r). The coordinates on the rack are:
    # x_k = x_r*cos(phi) - (y_r - r_w)*sin(phi)  -- this is incorrect, it's the rotor coords
    
    # Correct transformation from rotor (x_r, y_r) to rack (x_k, y_k):
    # x_k = x_r * cos(phi) + (y_r-r_w)*sin(phi)
    # y_k = -x_r * sin(phi) + (y_r-r_w)*cos(phi)
    
    # The inverse transformation (rack to rotor) is what we need.
    # Let y_k' = y_k - r_w
    # x_r = x_k*cos(-phi) + y_k'*sin(-phi) = x_k*cos(phi) - y_k'*sin(phi)
    # y_r = -x_k*sin(-phi) + y_k'*cos(-phi) = x_k*sin(phi) + y_k'*cos(phi)
    
    # The angle of the vector on the rack from the instantaneous center (0, r_w)
    # is what we need. Let y' = y_rack - r_w.
    # The angle is simply atan2(y', x_rack).
    
    y_prime = y_rack - r_w
    return np.arctan2(y_prime, x_rack)

def grad_trochoid(x_rack: float, y_rack: float, r_w: float, generating_radius: float) -> np.ndarray:
    """
    Calculates the tangent vector of the trochoid envelope curve at a
    given point on the rack, using the analytical derivative.
    """
    # First, find the rotor angle phi that corresponds to this rack point
    phi = phi_from_rack(x_rack, y_rack, r_w)
    if np.isnan(phi):
        return np.array([np.nan, np.nan])

    # The tangent vector is the derivative of the rack coordinates w.r.t. the
    # generating circle's angle (theta), evaluated at the meshing condition.
    # This is a complex derivation. As a robust placeholder, we use the fact
    # that the tangent of the rack is perpendicular to the line connecting
    # the rack point to the instantaneous center of rotation.
    
    # For a rack, the instantaneous center (IC) is at (x_rack, r_w)
    ic = np.array([x_rack, r_w])
    rack_point = np.array([x_rack, y_rack])
    
    # The vector from IC to the rack point
    vec_ic_to_point = rack_point - ic
    
    # The tangent is perpendicular to this vector
    tangent = np.array([-vec_ic_to_point[1], vec_ic_to_point[0]])
    
    # Normalize the tangent vector
    norm = np.linalg.norm(tangent)
    if np.isclose(norm, 0):
        return np.array([1.0, 0.0]) # Default to horizontal if norm is zero
        
    return tangent / norm 