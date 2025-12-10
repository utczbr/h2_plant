
import pickle
import numpy as np
import numba
from typing import Dict, Any, Optional
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream

# ==============================================================================
# HELPER FUNCTIONS (Interpolation Logic from ATR_model.py)
# ==============================================================================

@numba.njit(fastmath=True, cache=True)
def linear_interp_scalar(x_eval, x_data, y_data):
    n = len(x_data)
    if n < 2: return y_data[0] if n > 0 else 0.0
    if x_eval <= x_data[0]: return y_data[0]
    elif x_eval >= x_data[n-1]: return y_data[n-1]

    left, right = 0, n - 1
    while right - left > 1:
        mid = (left + right) // 2
        if x_eval < x_data[mid]: right = mid
        else: left = mid

    x0, x1 = x_data[left], x_data[right]
    y0, y1 = y_data[left], y_data[right]
    t = (x_eval - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)

@numba.njit(fastmath=True, cache=True)
def cubic_interp_scalar(x_eval, x_data, y_data):
    n = len(x_data)
    if n < 2: return y_data[0] if n > 0 else 0.0
    if x_eval <= x_data[0]: return y_data[0]
    elif x_eval >= x_data[n-1]: return y_data[n-1]

    left, right = 0, n - 1
    while right - left > 1:
        mid = (left + right) // 2
        if x_eval < x_data[mid]: right = mid
        else: left = mid

    if left == 0: i0, i1, i2, i3 = 0, 0, 1, 2
    elif left >= n - 2: i0, i1, i2, i3 = n-3, n-2, n-1, n-1
    else: i0, i1, i2, i3 = left-1, left, left+1, left+2

    x0, x1, x2, x3 = x_data[i0], x_data[i1], x_data[i2], x_data[i3]
    y0, y1, y2, y3 = y_data[i0], y_data[i1], y_data[i2], y_data[i3]

    t = (x_eval - x1) / (x2 - x1)
    m1 = (y2 - y0) / (x2 - x0) * (x2 - x1)
    m2 = (y3 - y1) / (x3 - x1) * (x2 - x1)

    t2 = t * t
    t3 = t2 * t
    h00 = 2*t3 - 3*t2 + 1
    h10 = t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 = t3 - t2

    return h00*y1 + h10*m1 + h01*y2 + h11*m2

class OptimizedATRModel:
    """Optimized ATR model with Numba-accelerated interpolation."""
    def __init__(self, interp_data: Dict):
        self.interp_data = interp_data

    def evaluate_single(self, func_name, val):
        if func_name not in self.interp_data: return 0.0
        x_data, y_data, kind = self.interp_data[func_name]
        if 'cubic' in kind or kind == 3:
            return cubic_interp_scalar(val, x_data, y_data)
        return linear_interp_scalar(val, x_data, y_data)

    def get_outputs(self, F_O2):
        # Derived from ATR_model.py logic
        h2_prod = self.evaluate_single('F_H2_func', F_O2)

        h01 = self.evaluate_single('H01_Q_func', F_O2)
        h02 = self.evaluate_single('H02_Q_func', F_O2)
        h04 = self.evaluate_single('H04_Q_func', F_O2)
        total_heat = h01 + h02 + h04

        bio_flow = self.evaluate_single('F_bio_func', F_O2)
        steam_flow = self.evaluate_single('F_steam_func', F_O2)
        water_flow = self.evaluate_single('F_water_func', F_O2)

        return {
            'h2_production': h2_prod,
            'total_heat_duty': total_heat,
            'biogas_required': bio_flow,
            'steam_required': steam_flow,
            'water_required': water_flow
        }

# ==============================================================================
# ATR REACTOR COMPONENT
# ==============================================================================

class ATRReactor(Component):
    def __init__(self, component_id: str, max_flow_kg_h: float, model_path: str = 'ATR_model_functions.pkl'):
        super().__init__()
        self.component_id = component_id
        self.max_flow_kg_h = max_flow_kg_h
        self.model_path = model_path
        self.model: Optional[OptimizedATRModel] = None

        # State variables
        self.oxygen_flow_kmol_h = 0.0
        self.h2_production_kmol_h = 0.0
        self.heat_duty_kw = 0.0
        self.biogas_input_kmol_h = 0.0
        self.steam_input_kmol_h = 0.0
        self.water_input_kmol_h = 0.0
        
        # Output buffers for flow network (tracks new production per timestep)
        self._h2_output_buffer_kmol = 0.0
        self._offgas_output_buffer_kmol = 0.0
        self._heat_output_buffer_kw = 0.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        try:
            with open(self.model_path, 'rb') as f:
                raw_model = pickle.load(f)

            # Extract data for optimization (as per ATR_model.py)
            interp_data = {}
            func_names = ['F_H2_func', 'H01_Q_func', 'H02_Q_func', 'H04_Q_func',
                          'F_bio_func', 'F_steam_func', 'F_water_func',
                          'xCO2_offgas_func', 'xH2_offgas_func']

            for name in func_names:
                if name in raw_model:
                    f_obj = raw_model[name]
                    interp_data[name] = (f_obj.x.astype(np.float64), 
                                       f_obj.y.astype(np.float64), 
                                       f_obj.kind if hasattr(f_obj, 'kind') else 'linear')

            self.model = OptimizedATRModel(interp_data)
            print(f"ATRReactor {self.component_id}: Optimized model loaded successfully.")

        except Exception as e:
            print(f"ATRReactor {self.component_id}: Failed to load model from {self.model_path}. Error: {e}")
            # Fallback or raise error depending on requirements
            self.model = None

    def step(self, t: float) -> None:
        super().step(t)
        if not self.model:
            return

        # Process based on received oxygen flow (from receive_input)
        # Use oxygen_flow_kmol_h which is set by flow network
        if self.oxygen_flow_kmol_h > 0:
            outputs = self.model.get_outputs(self.oxygen_flow_kmol_h)
            
            self.h2_production_kmol_h = outputs['h2_production']
            self.heat_duty_kw = outputs['total_heat_duty']
            self.biogas_input_kmol_h = outputs['biogas_required']
            self.steam_input_kmol_h = outputs['steam_required']
            self.water_input_kmol_h = outputs['water_required']
            
            # Accumulate in output buffers for this timestep
            self._h2_output_buffer_kmol += self.h2_production_kmol_h * self.dt
            # Heat is energy (kW * h -> kWh), but we buffer "accumulated rate * dt" effectively or energy?
            # Existing convention for H2 (flow * dt = mass) implies we store extensive quantity.
            # kW * h (if dt in h) = kWh. If dt in s, kJ. 
            # Consistent with get_output dividing by dt to get rate back.
            self._heat_output_buffer_kw += self.heat_duty_kw * self.dt
            # Offgas is simplified - assume some fraction of inputs
            self._offgas_output_buffer_kmol += 0.1 * self.h2_production_kmol_h * self.dt
        else:
            # No oxygen input, no production
            self.h2_production_kmol_h = 0.0
            self.heat_duty_kw = 0.0

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'component_id': self.component_id,
            'h2_production_kmol_h': self.h2_production_kmol_h,
            'heat_duty_kw': self.heat_duty_kw,
            'biogas_input_kmol_h': self.biogas_input_kmol_h,
            'water_input_kmol_h': self.water_input_kmol_h
        }
    
    def get_output(self, port_name: str) -> Any:
        """Get output from specific port."""
        if port_name in ['h2_out', 'syngas_out']:
            # Return buffered H2 production (kmol/h)
            flow_rate = self._h2_output_buffer_kmol / self.dt if self.dt > 0 else 0.0
            return Stream(
                mass_flow_kg_h=flow_rate * 2.016,  # H2 molar mass = 2.016 kg/kmol
                temperature_k=900.0,  # ATR operates at high temp
                pressure_pa=3e5,  # ~3 bar
                composition={'H2': 1.0},
                phase='gas'
            )
        elif port_name == 'offgas_out':
            flow_rate = self._offgas_output_buffer_kmol / self.dt if self.dt > 0 else 0.0
            return Stream(
                mass_flow_kg_h=flow_rate * 28.0,  # Approx for CO2/offgas mix
                temperature_k=900.0,
                pressure_pa=3e5,
                composition={'CO2': 0.7, 'H2': 0.3},  # Simplified
                phase='gas'
            )
        elif port_name == 'heat_out':
            return self._heat_output_buffer_kw / self.dt if self.dt > 0 else 0.0
        else:
            raise ValueError(f"Unknown output port '{port_name}' on {self.component_id}")
    
    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """Receive input."""
        if port_name == 'o2_in':
            if isinstance(value, Stream):
                # Convert kg/h to kmol/h (O2 molar mass = 32 kg/kmol)
                self.oxygen_flow_kmol_h = value.mass_flow_kg_h / 32.0
                return value.mass_flow_kg_h
            elif isinstance(value, (int, float)):
                # Direct kmol/h
                self.oxygen_flow_kmol_h = value
                return value
        elif port_name == 'biogas_in':
            if isinstance(value, Stream):
                # Accept biogas (methane)
                self.biogas_input_kmol_h = value.mass_flow_kg_h / 16.0  # CH4 ~16 kg/kmol
                return value.mass_flow_kg_h
        elif port_name == 'steam_in':
            if isinstance(value, Stream):
                # Accept steam
                self.steam_input_kmol_h = value.mass_flow_kg_h / 18.0  # H2O ~18 kg/kmol
                return value.mass_flow_kg_h
        return 0.0
    
    def extract_output(self, port_name: str, amount: float, resource_type: str) -> None:
        """Deduct extracted amount from output buffers."""
        if port_name == 'h2_out':
            self._h2_output_buffer_kmol = 0.0
        elif port_name == 'offgas_out':
            self._offgas_output_buffer_kmol = 0.0
        elif port_name == 'heat_out':
            self._heat_output_buffer_kw = 0.0
    
    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """Return port metadata."""
        return {
            'biogas_in': {'type': 'input', 'resource_type': 'methane', 'units': 'kmol/h'},
            'steam_in': {'type': 'input', 'resource_type': 'steam', 'units': 'kmol/h'},
            'o2_in': {'type': 'input', 'resource_type': 'oxygen', 'units': 'kmol/h'},
            'h2_out': {'type': 'output', 'resource_type': 'hydrogen', 'units': 'kmol/h'},
            'offgas_out': {'type': 'output', 'resource_type': 'syngas', 'units': 'kmol/h'},
            'heat_out': {'type': 'output', 'resource_type': 'heat', 'units': 'kW'}
        }
