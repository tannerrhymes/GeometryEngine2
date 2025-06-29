import pandas as pd
from typing import List
from ..config.compressor_config import CompressorConfig

class CSVDataManager:
    """Manages loading and validation of CSV compressor data."""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.data = self._load_and_validate_csv()
    
    def _load_and_validate_csv(self) -> pd.DataFrame:
        """Load CSV and validate required columns."""
        try:
            df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at path: {self.csv_path}")
        except Exception as e:
            raise ValueError(f"Failed to load CSV from {self.csv_path}: {e}")
        
        # Check required columns
        required_columns = [
            'Master Name', 'swv/r', 'D1e', 'D2e', 'L', 'C', 
            'Z1', 'Z2', 'wrap angle degrees'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
        
        return df
    
    def get_configuration(self, model_id: str) -> CompressorConfig:
        """Get configuration for a specific compressor model."""
        row = self.data[self.data['Master Name'] == model_id]
        
        if row.empty:
            available_models = self.data['Master Name'].tolist()
            raise ValueError(f"Model '{model_id}' not found. Available models: {available_models}")
        
        return CompressorConfig.from_csv_row(row.iloc[0])
    
    def get_all_configurations(self) -> List[CompressorConfig]:
        """Get configurations for all compressors in CSV."""
        return [CompressorConfig.from_csv_row(row) for _, row in self.data.iterrows()] 