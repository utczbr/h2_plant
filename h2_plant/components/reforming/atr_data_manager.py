
import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d
from typing import Dict, Optional, Any
from dataclasses import dataclass

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.core.component_registry import ComponentRegistry

# --- Constants & Unit Conversions ---
C_TO_K = 273.15
KW_TO_W = 1000.0
KG_HR_TO_KG_S = 1.0 / 3600.0
KMOL_HR_TO_MOL_S = 1000.0 / 3600.0

class ATRDataManager:
    """
    Singleton service to load and provide access to ATR interpolation functions.
    Reads the linear regression/interpolation data (surrogate model).
    """
    _instance = None
    _models: Dict[str, interp1d] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ATRDataManager, cls).__new__(cls)
        return cls._instance

    def load_data(self, csv_filename: str = 'ATR_linear_regressions.csv'):
        """
        Loads the regression data and creates interpolation functions.
        x column (F_O2) is the independent variable.
        """
        # Resolve path relative to this file or package data directory
        # Assuming structure: h2_plant/components/reforming/atr_data_manager.py
        # And data: h2_plant/data/ATR_linear_regressions.csv
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up two levels: components -> reforming -> h2_plant -> data
        data_dir = os.path.join(current_dir, '..', '..', 'data')
        csv_path = os.path.join(data_dir, csv_filename)

        if not os.path.exists(csv_path):
             # Fallback check if running from root
             if os.path.exists(f"h2_plant/data/{csv_filename}"):
                  csv_path = f"h2_plant/data/{csv_filename}"
             else:
                  # Last resort, assume absolute path or handle error
                  pass

        try:
            df = pd.read_csv(csv_path)
            # F_O2 is the independent variable 'x' in the CSV
            x = df['x'].values
            
            # Create interpolator for every column
            for col in df.columns:
                if col != 'x':
                    # fill_value="extrapolate" handles minor boundary float errors
                    self._models[col] = interp1d(x, df[col].values, kind='linear', fill_value="extrapolate")
            
            print(f"ATR Model loaded: {len(self._models)} functions available from {csv_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ATR model data from {csv_path}: {e}")

    def lookup(self, func_name: str, f_o2_kmol_h: float) -> float:
        """Retrieves interpolated value for a specific function name."""
        if func_name not in self._models:
            # Fallback or error handling
            return 0.0
        return float(self._models[func_name](f_o2_kmol_h))

class ATRBaseComponent(Component):
    """
    Base class for all ATR components (Heaters, Coolers, Reactors).
    Implements the logic to find the Plant Load (Oxygen Flow) and basic Lookup.
    """
    def __init__(self, component_id: str = None):
        super().__init__(component_id=component_id, config={})
        self.results: Dict[str, float] = {}

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        self.data_manager = ATRDataManager()
        # Ensure data is loaded (lazy load or pre-load)
        if not self.data_manager._models:
            self.data_manager.load_data()

    def get_oxygen_flow(self, streams: Dict[str, Stream]) -> float:
        """
        Determines the F_O2 (kmol/h) driving the ATR model.
        Strategy: Look for a stream tagged 'oxygen_feed' or assume global control signal.
        For this implementation, we assume the component receiving the O2 stream 
        broadcasts it, or we read the mass flow of the system input.
        """
        # --- Implementation Note ---
        # In a real sim, the 'O2 Flow' is likely a control signal. 
        # Here we attempt to calculate it if we are the O2 heater, 
        # otherwise we might need to read it from a shared registry value.
        # For robustness, we will check if 'F_O2' is in the config, 
        # else infer from specific input stream if available.
        
        # Valid Range Check: 7.125 - 23.75 kmol/hr
        # Default fallback for safety
        f_o2 = float(self.config.get('current_o2_flow_kmol_h', 15.0)) 
        return np.clip(f_o2, 7.125, 23.75)

    def _apply_thermal_model(self, stream: Stream, f_o2: float, q_key: str, t_out_key: str) -> None:
        """
        Applies the lookup table values to the stream.
        """
        # Lookup values
        duty_kw = self.data_manager.lookup(q_key, f_o2)
        t_out_c = self.data_manager.lookup(t_out_key, f_o2)
        
        # Update Stream State
        # 1. Update Temperature
        stream.temperature_k = t_out_c + C_TO_K
        
        # 2. Record Duty (convert kW -> W)
        # Positive Q in CSV = Heating (for heaters)
        # Negative Q in CSV = Cooling (for coolers)
        self.results['duty_w'] = duty_kw * KW_TO_W
        self.results['T_out_K'] = stream.temperature_k
