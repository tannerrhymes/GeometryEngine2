from dataclasses import dataclass

@dataclass
class OptimizationConfig:
    """Configuration for optimization algorithm and parameters."""
    
    # Differential Evolution parameters
    strategy: str = 'best1bin'
    maxiter: int = 100
    popsize: int = 15
    tolerance: float = 0.01
    
    # Objective function weights
    weight_volume: float = 1.0e6      # w1 - volume error priority
    weight_D1e: float = 1.0e-3        # w2 - main diameter error
    weight_D2e: float = 1.0e-3        # w3 - gate diameter error
    
    # Constraint penalties
    penalty_geometric: float = 1.0e8   # Invalid geometry penalty
    
    # Convergence criteria
    target_volume_error: float = 0.01  # 1% volume error
    target_diameter_error: float = 2.0  # 2mm diameter error 