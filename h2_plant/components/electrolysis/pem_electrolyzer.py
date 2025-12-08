from typing import Dict, Any, Optional
import numpy as np
import pickle
from pathlib import Path

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.component_ids import ComponentID
from h2_plant.config.constants_physics import PEMConstants
from h2_plant.models.pem_physics import calculate_eta_F
from h2_plant.models.thermal_inertia import ThermalInertiaModel
from h2_plant.core.stream import Stream

from numpy.polynomial import Polynomial
# from scipy.optimize import fsolve
# from scipy.interpolate import interp1d
import warnings
import pickle
import os
CONST = PEMConstants()

class DetailedPEMElectrolyzer(Component):
    """
    Mechanistic PEM model with degradation tracking.
    Implements white-box electrochemistry (Nernst, overvoltages, BoP).
    """
    
    def __init__(self, config: Any):
        """
        Initialize Detailed PEM Electrolyzer.
        
        Args:
            config (Union[dict, PEMPhysicsSpec]): Configuration object or dict.
        """
        super().__init__(config)
        
        # Handle Pydantic Model or Dict
        if hasattr(config, 'max_power_mw'):
            # It's likely a PEMPhysicsSpec
            self.max_power_mw = config.max_power_mw
            self.base_efficiency = config.base_efficiency
            self.use_polynomials = config.use_polynomials
            self.water_excess_factor = getattr(config, 'water_excess_factor', 0.02)
        else:
            # Legacy Dict
            self.config = config
            self.max_power_mw = config.get('max_power_mw', 5.0)
            self.base_efficiency = config.get('base_efficiency', 0.65)
            self.use_polynomials = config.get('use_polynomials', False)
            self.water_excess_factor = config.get('water_excess_factor', 0.02)
            if 'component_id' in config:
                self.component_id = config['component_id']
        
        # State variables
        self.t_op_h = 0.0
        self.P_consumed_W = 0.0
        self.m_H2_kg_s = 0.0
        self.m_H2O_kg_s = 0.0
        self.I_total = 0.0
        
        # Constants
        self.F = 96485.33
        self.R = 8.314
        self.P_ref = 1.0e5
        self.z = 2
        self.MH2 = 2.016e-3
        self.MO2 = 31.998e-3
        self.MH2O = 18.015e-3
        
        # Stack Config
        self.N_stacks = 35
        self.N_cell_per_stack = 85
        self.A_cell = 300 # cm2
        self.Area_Total = self.N_stacks * self.N_cell_per_stack * self.A_cell
        self.j_nom = 2.91
        
        # Operation
        self.T = 333.15 # 60C
        self.P_op = 40.0e5 # 40 bar
        
        # Physics
        self.delta_mem = 100 * 1e-4
        self.sigma_base = 0.1
        self.j0 = 1.0e-6
        self.alpha = 0.5
        self.j_lim = 4.0
        
        # BoP
        self.floss = 0.02
        self.P_nominal_sistema_W = self.max_power_mw * 1e6
        self.P_bop_fixo = 0.025 * self.P_nominal_sistema_W
        self.k_bop_var = 0.04
        
        # Degradation
        self.H_MES = 730.0
        self.polynomial_list = []
        if self.use_polynomials:
            self._load_polynomials()
        
        if self.use_polynomials:
            self._load_polynomials()
        
        # super().__init__() call REMOVED (was duplicate)
        self._lut = None
        
        # State variables (additional for compatibility)
        self.m_O2_kg_s = 0.0
        self.V_cell = 0.0
        self.heat_output_kw = 0.0
        self.state = "OFF"
        
        # Setpoint from coordinator
        self._target_power_mw = 0.0
        
        # Per-timestep outputs (for monitoring compatibility)
        self.h2_output_kg = 0.0
        self.o2_output_kg = 0.0
        
        # Accumulators
        self.cumulative_h2_kg = 0.0
        self.cumulative_o2_kg = 0.0
        self.cumulative_energy_kwh = 0.0
        
        # Initialize Degradation Model (Reference Alignment - Delta-based)
        # ==========================================================
        # 2. SETUP DEGRADATION INTERPOLATION (always needed)
        # ==========================================================
        self.t_op_h_table = np.array(CONST.DEGRADATION_YEARS) * 8760.0
        self.v_cell_table = np.array(CONST.DEGRADATION_V_STACK) / CONST.N_cell_per_stack
        
        try:
            # from scipy.interpolate import interp1d
            
            # self.interpolator = interp1d(
            #     t_op_h_table, v_cell_table, kind='linear',
            #     fill_value=(v_cell_table[0], v_cell_table[-1]), bounds_error=False
            # )
            
            # Calculate BOL (Beginning of Life) reference voltage
            from h2_plant.models import pem_physics as phys
            self.V_CELL_BOL_NOM = phys.calculate_Vcell_base(CONST.j_nom, self.T, self.P_op)
            
        except Exception as e:
            print(f"PEM Init Warning: Failed to init degradation interpolator: {e}")
            self.deg_interpolator = None
            self.V_CELL_BOL_NOM = 0.0
        
        # Thermal Model
        self.thermal_model = ThermalInertiaModel(
            C_thermal_J_K=2.6e6,      # ~5 MW stack + fluid
            h_A_passive_W_K=100.0,    # Natural convection
            T_initial_K=298.15,       # Start cold
            max_cooling_kw=100.0      # Chiller capacity
        )

    def _load_polynomials(self):
        """Load pre-calculated polynomial models for fast simulation."""
        # Clean relative path logic (robust across systems)
        try:
            # 1. Try local directory (where script runs)
            local_path = Path("degradation_polynomials.pkl")
            if local_path.exists():
                with open(local_path, 'rb') as f:
                    self.polynomial_list = pickle.load(f)
                    return

            # 2. Try package data directory relative to this file
            # h2_plant/components/electrolysis/pem_electrolyzer.py -> h2_plant/data/
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent # h2_plant/
            data_path = project_root / 'data' / 'degradation_polynomials.pkl'
            
            if data_path.exists():
                with open(data_path, 'rb') as f:
                    self.polynomial_list = pickle.load(f)
                    # print(f"PEM: Loaded polynomials from {data_path}")
            else:
                # print(f"PEM Warning: Polynomials not found at {data_path}")
                self.use_polynomials = False
                
        except Exception as e:
            print(f"PEM Error loading polynomials: {e}")
            self.use_polynomials = False

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        # Resolve LUT from registry (if available)
        if registry.has(ComponentID.LUT_MANAGER.value):
            self._lut = registry.get(ComponentID.LUT_MANAGER)
        
    def step(self, t: float) -> None:
        super().step(t)
        
        # 1. Get power setpoint
        P_setpoint_mw = self._target_power_mw
        
        # Clamp to max power
        if P_setpoint_mw > self.max_power_mw:
            P_setpoint_mw = self.max_power_mw
        
        # Check for Coordinator setpoint if not set manually
        if P_setpoint_mw <= 0:
            try:
                coordinator = self._registry.get(ComponentID.DUAL_PATH_COORDINATOR)
                if coordinator:
                    P_setpoint_mw = coordinator.pem_setpoint_mw
            except:
                pass
            
        if P_setpoint_mw <= 0.001: 
            # OFF State
            self.m_H2_kg_s = 0.0
            self.m_O2_kg_s = 0.0
            self.m_H2O_kg_s = 0.0
            self.P_consumed_W = 0.0
            self.V_cell = 0.0
            self.I_total = 0.0
            self.heat_output_kw = 0.0
            self.state = "OFF"
            self.h2_output_kg = 0.0
            self.o2_output_kg = 0.0
            return
            
        self.state = "ON"
        
        try:
            target_power_W = P_setpoint_mw * 1e6
            j_op = 0.0
            
            # --- POLYNOMIAL MODE (FAST) ---
            if self.use_polynomials and self.polynomial_list:
                 month_index = int(self.t_op_h / self.H_MES)
                 if month_index >= len(self.polynomial_list):
                     month_index = len(self.polynomial_list) - 1
                 poly_object = self.polynomial_list[month_index]
                 
                 if isinstance(poly_object, dict):
                     if target_power_W <= poly_object['split_point']:
                         j_op = poly_object['poly_low'](target_power_W)
                     else:
                         j_op = poly_object['poly_high'](target_power_W)
                 else:
                     j_op = poly_object(target_power_W)
            else:
                # --- SOLVER MODE (JIT) ---
                from h2_plant.optimization.numba_ops import solve_pem_j_jit
                
                # Initial guess
                j_guess = self.j_nom * (target_power_W / self.P_nominal_sistema_W)
                
                try:
                    j_op = solve_pem_j_jit(
                        target_power_W,
                        self.T,
                        self.P_op,
                        self.Area_Total,
                        self.P_bop_fixo,
                        self.k_bop_var,
                        j_guess,
                        self.R, self.F, self.z, self.alpha, self.j0, self.j_lim,
                        self.delta_mem, self.sigma_base, self.P_ref
                    )
                except Exception as e:
                    print(f"PEM JIT Solver failed: {e}. Using linear guess.")
                    j_op = j_guess
            
            # Clamp j
            j_op = max(0.001, min(j_op, CONST.j_lim))
            
            # Calculate Outputs
            self.I_total = j_op * self.Area_Total
            
            # Mass Flows
            from h2_plant.models import pem_physics as phys
            self.m_H2_kg_s, self.m_O2_kg_s, self.m_H2O_kg_s = phys.calculate_flows(j_op)
            
            # Voltage
            U_deg = self._calculate_U_deg(self.t_op_h)
            self.V_cell = phys.calculate_Vcell_base(j_op, self.T, CONST.P_op_default) + U_deg
            
            P_stack = self.I_total * self.V_cell
            P_bop = CONST.P_bop_fixo + CONST.k_bop_var * P_stack
            self.P_consumed_W = P_stack + P_bop
            
            # Heat Calculation
            # Heat = (V_cell - U_tn) * I + BoP_Loss
            # Using U_rev calculated at T + approx entropy (1.48V total)
            U_rev = self._calculate_U_rev(self.T)
            # Approx U_tn (thermo-neutral) is usually around 1.48V
            # We use calculated U_rev + T contribution or fixed 1.481 for heat balance consistency
            U_tn = 1.481 
            
            heat_power_W = self.I_total * (self.V_cell - U_tn)
            if heat_power_W < 0: heat_power_W = 0 # Endothermic operation covered by external heat if needed
            
            heat_power_W += self.P_consumed_W * 0.01 # BoP heat
            self.heat_output_kw = heat_power_W / 1000.0
            
            # Thermal Model Step
            dt_seconds = self.dt * 3600.0
            self.thermal_model.step(dt_seconds, heat_power_W, 333.15)
            self.thermal_model.T_current_K = 333.15 # Force constant T for now
            # self.T is updated by thermal model in real sim, here we force it.
            
        except Exception as e:
            print(f"PEM Calculation Error: {e}")
            import traceback
            traceback.print_exc()
            self.P_consumed_W = 0.0
            
        # Outputs
        dt_seconds = self.dt * 3600.0
        self.h2_output_kg = self.m_H2_kg_s * dt_seconds
        self.o2_output_kg = self.m_O2_kg_s * dt_seconds
        
        reaction_water_kg = self.m_H2O_kg_s * dt_seconds
        self.water_consumption_kg = reaction_water_kg * (1.0 + self.water_excess_factor)
        
        self.cumulative_h2_kg += self.h2_output_kg
        self.cumulative_o2_kg += self.o2_output_kg
        self.cumulative_energy_kwh += (self.P_consumed_W / 1000.0) * self.dt
        
        if hasattr(self, 'dt'):
             self.t_op_h += self.dt
    
    def _load_polynomials(self):
        """Load degradation polynomials from pickle file."""
        try:
            # Assuming data file is in h2_plant/data relative to project root
            # We need a robust way to find it. For now, try relative path.
            # Or assume CWD is project root.
            pkl_path = "h2_plant/data/degradation_polynomials.pkl"
            if not os.path.exists(pkl_path):
                 # Try absolute path based on known location
                 pkl_path = "/home/stuart/Documentos/Planta Hidrogenio/h2_plant/data/degradation_polynomials.pkl"
            
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    self.polynomial_list = pickle.load(f)
                print(f"PEM: Loaded {len(self.polynomial_list)} polynomials.")
            else:
                print(f"PEM Warning: Polynomial file not found at {pkl_path}. Fallback to solver.")
                self.use_polynomials = False
        except Exception as e:
            print(f"PEM Error loading polynomials: {e}. Fallback to solver.")
            self.use_polynomials = False

    def set_power_input_mw(self, P_mw: float) -> None:
        """Set power input setpoint for next step (called by coordinator)."""
        self._target_power_mw = max(0.0, P_mw)
    
    def shutdown(self) -> None:
        """Shut down electrolyzer (called by coordinator when PEM not needed)."""
        self._target_power_mw = 0.0
        self.state = "SHUTDOWN"

    def _calculate_U_rev(self, T: float) -> float:
        P_op = 40.0e5   # 40 bar
        U_rev_T = 1.229 - 0.9e-3 * (T - 298.15)
        pressure_ratio = P_op / 1.0e5
        Nernst_correction = (CONST.R * T) / (CONST.z * CONST.F) * np.log(pressure_ratio**1.5)
        return U_rev_T + Nernst_correction



    def _calculate_voltage_base(self, j_op: float, T: float) -> float:
        """Calculate base cell voltage (BOL) without degradation."""
        from h2_plant.models import pem_physics as phys
        return phys.calculate_Vcell_base(j_op, T, CONST.P_op_default)

    def _calculate_degradation_voltage(self, t_op_h: float) -> float:
        """Calculate voltage degradation using Delta-based approach (Reference Alignment)."""
        if self.deg_interpolator is None:
            return 0.0
    def _calculate_U_deg(self, t_op_h: float) -> float:
        """Calculate degradation overpotential from table."""
        # V_cell_degraded = self.interpolator(t_op_h)
        V_cell_degraded = np.interp(t_op_h, self.t_op_h_table, self.v_cell_table)
        U_deg = np.maximum(0.0, V_cell_degraded - self.V_CELL_BOL_NOM)
        return float(U_deg)

    def _calculate_voltage(self, j_op: float, T: float) -> float:
        """Calculate total voltage with degradation."""
        base_V = self._calculate_voltage_base(j_op, T)
        U_deg = self._calculate_U_deg(self.t_op_h)
        return base_V + U_deg
        
    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            "h2_production_kg_h": self.m_H2_kg_s * 3600,
            "o2_production_kg_h": self.m_O2_kg_s * 3600,
            "h2_output_kg": self.h2_output_kg,  # For monitoring compatibility
            "o2_output_kg": self.o2_output_kg,  # For monitoring compatibility
            "water_consumption_kg": getattr(self, 'water_consumption_kg', 0.0),
            "power_consumption_mw": self.P_consumed_W / 1e6,
            "heat_output_kw": self.heat_output_kw,
            "system_efficiency_percent": (self.m_H2_kg_s * CONST.LHVH2_kWh_kg * 3.6e6) / self.P_consumed_W * 100 if self.P_consumed_W > 0 else 0.0,
            "cell_voltage_v": self.V_cell,
            "state": self.state,
            "cumulative_h2_kg": self.cumulative_h2_kg,
            "cumulative_o2_kg": self.cumulative_o2_kg,
            "cumulative_energy_kwh": self.cumulative_energy_kwh
        }

    def get_output(self, port_name: str) -> Any:
        """Get output from specific port."""
        if port_name == 'h2_out':
            # Calculate unreacted water carryover for output stream
            water_fraction = CONST.unreacted_water_fraction
            m_H2O_carryover_kg_s = self.m_H2_kg_s * water_fraction / (1.0 - water_fraction)
            m_total_out_kg_s = self.m_H2_kg_s + m_H2O_carryover_kg_s

            return Stream(
                mass_flow_kg_h=m_total_out_kg_s * 3600.0,
                temperature_k=353.15,  # Approx 80C
                pressure_pa=30e5,      # 30 bar output
                composition={
                    'H2': self.m_H2_kg_s / m_total_out_kg_s if m_total_out_kg_s > 0 else 0.0,
                    'H2O': m_H2O_carryover_kg_s / m_total_out_kg_s if m_total_out_kg_s > 0 else 0.0
                },
                phase='gas' # Wet hydrogen
            )
        elif port_name == 'oxygen_out':
            return Stream(
                mass_flow_kg_h=self.m_O2_kg_s * 3600.0,
                temperature_k=353.15,
                pressure_pa=101325.0,  # Vented or low pressure
                composition={'O2': 1.0},
                phase='gas'
            )
        elif port_name == 'heat_out':
            # Return calculated waste heat
            return self.heat_output_kw
        else:
            raise ValueError(f"Unknown output port '{port_name}' on {self.component_id}")

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """Receive input."""
        if port_name == 'water_in':
            if isinstance(value, Stream):
                # We consume water based on reaction stoichiometry
                # m_H2O_consumed = m_H2 * (MH2O/MH2)
                # But we calculated m_H2O_kg_s in step() based on power.
                # So we just accept whatever is given up to what we need?
                # Or we just accept it all and assume it's stored/used?
                # For now, accept all.
                return value.mass_flow_kg_h
        elif port_name == 'power_in':
            if isinstance(value, (int, float)):
                # Power is set by coordinator usually, but if connected to grid/rectifier
                # we might use this.
                self._target_power_mw = float(value)
                return float(value)
        return 0.0

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """Return port metadata."""
        return {
            'water_in': {'type': 'input', 'resource_type': 'water', 'units': 'kg/h'},
            'power_in': {'type': 'input', 'resource_type': 'electricity', 'units': 'MW'},
            'h2_out': {'type': 'output', 'resource_type': 'hydrogen', 'units': 'kg/h'},
            'oxygen_out': {'type': 'output', 'resource_type': 'oxygen', 'units': 'kg/h'},
            'heat_out': {'type': 'output', 'resource_type': 'heat', 'units': 'kW'}
        }

# ============================================================================
# AUXILIARY COMPONENTS (Shared by other modules)
# ============================================================================

class RecirculationPump(Component):
    """Simple pump model."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.flow_rate_kg_h = 0.0
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
    def step(self, t: float) -> None:
        pass
    def get_state(self) -> Dict[str, Any]:
        return {"component_id": self.component_id}

class HeatExchanger(Component):
    """Simple heat exchanger model."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.inlet_flow_kg_h = 0.0
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
    def step(self, t: float) -> None:
        pass
    def get_state(self) -> Dict[str, Any]:
        return {"component_id": self.component_id}

class SeparationTank(Component):
    """Simple separation tank model."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.gas_inlet_kg_h = 0.0
        self.dry_gas_outlet_kg_h = 0.0
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
    def step(self, t: float) -> None:
        self.dry_gas_outlet_kg_h = self.gas_inlet_kg_h # Pass-through
    def get_state(self) -> Dict[str, Any]:
        return {"component_id": self.component_id}

class PSAUnit(Component):
    """Pressure Swing Adsorption unit model."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.feed_gas_kg_h = 0.0
        self.product_gas_kg_h = 0.0
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
    def step(self, t: float) -> None:
        self.product_gas_kg_h = self.feed_gas_kg_h # Pass-through
    def get_state(self) -> Dict[str, Any]:
        return {"component_id": self.component_id}

class RectifierTransformer(Component):
    """Power electronics model."""
    def __init__(self, max_power_kw: float, *args, **kwargs):
        super().__init__()
        self.max_power_kw = max_power_kw
        self.dc_output_kw = max_power_kw # Default to max for testing
        self.ac_input_kw = max_power_kw
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
    def step(self, t: float) -> None:
        pass
    def get_state(self) -> Dict[str, Any]:
        return {"component_id": self.component_id}
