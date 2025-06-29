import numpy as np
from shapely.geometry import Polygon, LineString
from ..config.compressor_config import CompressorConfig

class SweptVolumeCalculator:
    """
    Calculates the swept volume of the compressor from the main rotor profile.
    """

    def __init__(self, main_rotor_profile: np.ndarray, config: CompressorConfig):
        """
        Initializes the SweptVolumeCalculator.

        Args:
            main_rotor_profile (np.ndarray): Nx2 array of [x, y] coordinates for the main rotor.
            config (CompressorConfig): The configuration for the compressor.
        """
        if main_rotor_profile is None or main_rotor_profile.size < 3:
            raise ValueError("Main rotor profile must have at least 3 points.")
            
        self.main_rotor_profile = main_rotor_profile
        self.config = config

    def _shoelace_area(self, polygon_points: np.ndarray) -> float:
        """
        Calculates the area of a polygon using the Shoelace formula.
        Assumes the points are ordered (clockwise or counter-clockwise).
        """
        x = polygon_points[:, 0]
        y = polygon_points[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def calculate_volume_per_revolution(self) -> float:
        """
        Calculates the total swept volume per revolution using the formula:
        V_swept = A_groove_main * L * Z1
        """
        # Step 1: Define the outer circle (the "cap") from the target outer diameter
        r1e_m = self.config.target_D1e / 2000.0
        outer_circle = LineString(self._create_circle_points(r1e_m))

        # Step 2: Create a full rotor by repeating the single lobe profile Z1 times
        full_rotor_points = []
        for i in range(self.config.z1):
            angle = 2 * np.pi * i / self.config.z1
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated_profile = self.main_rotor_profile @ rotation_matrix.T
            full_rotor_points.extend(rotated_profile.tolist())

        full_rotor_poly = Polygon(full_rotor_points)

        # Step 3: The groove area is the area of the outer circle minus the area of the rotor
        # This simplifies the complex intersection logic for a robust calculation.
        rotor_area = full_rotor_poly.area
        outer_circle_area = np.pi * r1e_m**2
        
        total_groove_area = outer_circle_area - rotor_area
        
        if total_groove_area <= 0:
            # This can happen with invalid rotor geometries that self-intersect or exceed the outer circle
            return 0.0

        # The area of a single groove is the total groove area divided by the number of lobes.
        a_groove_main = total_groove_area / self.config.z1

        # Step 4: Calculate the final swept volume
        rotor_length_m = self.config.rotor_length / 1000.0
        swept_volume = a_groove_main * rotor_length_m * self.config.z1

        return swept_volume
        
    def _create_circle_points(self, radius, num_points=100):
        """Helper to create points for a circle."""
        angles = np.linspace(0, 2 * np.pi, num_points)
        return np.array([radius * np.cos(angles), radius * np.sin(angles)]).T 