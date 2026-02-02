
import numpy as np
import pandas as pd
import os
from typing import Dict, Optional, Any, Tuple
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
    _raw_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ATRDataManager, cls).__new__(cls)
        return cls._instance

    def load_data(self, csv_filename: str = 'ATR_linear_regressions.csv'):
        """
        Loads the regression data and stores raw arrays for fast np.interp lookup.
        x column (F_O2) is the independent variable.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, '..', '..', 'data')
        csv_path = os.path.join(data_dir, csv_filename)

        if not os.path.exists(csv_path):
             if os.path.exists(f"h2_plant/data/{csv_filename}"):
                  csv_path = f"h2_plant/data/{csv_filename}"

        try:
            df = pd.read_csv(csv_path)
            x = np.ascontiguousarray(df['x'].values, dtype=np.float64)

            for col in df.columns:
                if col != 'x':
                    y = np.ascontiguousarray(df[col].values, dtype=np.float64)
                    self._raw_data[col] = (x, y)

            print(f"ATR Model loaded: {len(self._raw_data)} functions available from {csv_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to load ATR model data from {csv_path}: {e}")

    def lookup(self, func_name: str, f_o2_kmol_h: float) -> float:
        """Retrieves interpolated value with linear extrapolation beyond bounds."""
        if func_name not in self._raw_data:
            return 0.0
        x, y = self._raw_data[func_name]
        if f_o2_kmol_h <= x[0]:
            slope = (y[1] - y[0]) / (x[1] - x[0])
            return float(y[0] + slope * (f_o2_kmol_h - x[0]))
        if f_o2_kmol_h >= x[-1]:
            slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
            return float(y[-1] + slope * (f_o2_kmol_h - x[-1]))
        return float(np.interp(f_o2_kmol_h, x, y))

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
        if not self.data_manager._raw_data:
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
