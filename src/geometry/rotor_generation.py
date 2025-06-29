import numpy as np
from scipy.optimize import fsolve
from ..config.compressor_config import CompressorConfig

class RotorGenerator:
    """
    Generates main and gate rotor profiles from a given rack profile using
    the envelope meshing theory.
    """
    
    def __init__(self, rack_profile: np.ndarray, config: CompressorConfig):
        """
        Initializes the RotorGenerator.

        Args:
            rack_profile (np.ndarray): A Nx2 array of [x, y] coordinates for the rack.
            config (CompressorConfig): The configuration for the compressor.
        """
        if rack_profile is None or rack_profile.size == 0:
            raise ValueError("Rack profile cannot be empty.")
            
        self.rack_profile = rack_profile
        self.config = config
        
        self.main_rotor_profile = None
        self.gate_rotor_profile = None

    def generate_rotors(self):
        """
        Generates both main and gate rotor profiles from the rack.
        """
        if self.rack_profile.shape[0] < 2:
             raise ValueError("Rack profile must have at least 2 points.")

        x_rack, y_rack = self.rack_profile[:, 0], self.rack_profile[:, 1]
        
        # Calculate the gradient of the rack profile (dy/dx)
        # Using second-order accurate central differences for interior points
        dydx_rack = np.gradient(y_rack, x_rack, edge_order=2)
        
        # Generate main rotor (z1)
        r1w_m = self.config.r1w / 1000.0
        self.main_rotor_profile = self._generate_one_rotor(r1w_m, x_rack, y_rack, dydx_rack)
        
        # Generate gate rotor (z2)
        r2w_m = self.config.r2w / 1000.0
        self.gate_rotor_profile = self._generate_one_rotor(r2w_m, x_rack, y_rack, dydx_rack)

    def _generate_one_rotor(self, r_w, x_rack, y_rack, dydx_rack) -> np.ndarray:
        """
        Generates a single rotor profile from the rack using envelope theory.

        Args:
            r_w (float): Pitch radius of the rotor to be generated (in meters).
            x_rack (np.ndarray): X-coordinates of the rack.
            y_rack (np.ndarray): Y-coordinates of the rack.
            dydx_rack (np.ndarray): Gradient of the rack profile.

        Returns:
            np.ndarray: A Nx2 array of [x, y] coordinates for the rotor profile.
        """
        num_points = len(x_rack)
        rotor_coords = np.zeros((num_points, 2))

        for j in range(num_points):
            def meshing_equation(theta):
                """
                This is Equation 2.5 from Geometry.md, rearranged to be solved for theta.
                (dy_dr/dx_r) * (r_w*theta - y_r) - (r_w - x_r) = 0
                """
                return dydx_rack[j] * (r_w * theta - y_rack[j]) - (r_w - x_rack[j])

            try:
                # Solve for the meshing angle theta
                theta_solution, = fsolve(meshing_equation, 0.0)
                
                # Apply coordinate transformation (Eq. 2.4 from Geometry.md)
                x_rotor = x_rack[j] * np.cos(theta_solution) + (y_rack[j] - r_w) * np.sin(theta_solution)
                y_rotor = -x_rack[j] * np.sin(theta_solution) + (y_rack[j] - r_w) * np.cos(theta_solution)
                
                rotor_coords[j] = [x_rotor, y_rotor]

            except Exception:
                # If solver fails, it indicates an issue with the rack geometry (e.g., discontinuity)
                # For robustness, we can skip the point, but it's better to fail fast.
                rotor_coords[j] = [np.nan, np.nan]

        # Remove any points where the solver failed
        return rotor_coords[~np.isnan(rotor_coords).any(axis=1)] 