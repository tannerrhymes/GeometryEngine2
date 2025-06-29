from dataclasses import dataclass, field
import pandas as pd
import numpy as np

@dataclass
class CompressorConfig:
    """Configuration for a single compressor optimization run based on CSV data."""
    
    # From CSV (required)
    model_id: str
    target_swv_per_rev: float        # swv/r (m³/rev)
    target_D1e: float                # D1e (mm)
    target_D2e: float                # D2e (mm)
    rotor_length: float              # L (mm)
    center_distance: float           # C (mm)
    z1: int                          # Main rotor lobes
    z2: int                          # Gate rotor lobes
    wrap_angle: float                # degrees
    
    # Calculated pitch circles (derived from CSV)
    r1w: float = field(init=False)   # Main rotor pitch radius (mm)
    r2w: float = field(init=False)   # Gate rotor pitch radius (mm)
    
    def __post_init__(self):
        """Calculate derived parameters from CSV data."""
        # Calculate pitch radii using literature equations
        if (1 + self.z2/self.z1) == 0:
            raise ValueError("Invalid gear ratio causing division by zero.")
        self.r1w = self.center_distance / (1 + self.z2/self.z1)
        self.r2w = self.center_distance - self.r1w
        
        # Validate gear ratio relationship
        if self.r1w == 0:
            raise ValueError("Main rotor pitch radius (r1w) cannot be zero.")
            
        gear_ratio_calculated = self.r2w / self.r1w
        gear_ratio_expected = self.z2 / self.z1
        
        if abs(gear_ratio_calculated - gear_ratio_expected) > 1e-6:
            raise ValueError(f"Gear ratio mismatch: calculated={gear_ratio_calculated:.6f}, "
                           f"expected={gear_ratio_expected:.6f}")

    @classmethod
    def from_csv_row(cls, row: pd.Series) -> 'CompressorConfig':
        """Create configuration from a CSV row."""
        try:
            return cls(
                model_id=str(row['Master Name']),
                target_swv_per_rev=float(row['swv/r']),
                target_D1e=float(row['D1e']),
                target_D2e=float(row['D2e']),
                rotor_length=float(row['L']),
                center_distance=float(row['C']),
                z1=int(row['Z1']),
                z2=int(row['Z2']),
                wrap_angle=float(row['wrap angle degrees'])
            )
        except KeyError as e:
            raise KeyError(f"Missing expected column in CSV row: {e}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Type conversion error for row {row.get('Master Name', 'Unknown')}: {e}") 