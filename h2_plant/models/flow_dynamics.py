"""
Pressure and Flow Dynamics for Pump Systems and Gas Accumulators
Implements lumped-compliance model for transient response.
"""

import math

class PumpFlowDynamics:
    """
    Pump with transient flow response.
    Governs: (d/dt)Q = (1/L_eq) * (p_pump - p_system)
    """
    
    def __init__(
        self,
        pump_shutoff_pa: float = 350000.0,  # 3.5 bar
        pump_resistance_pa_per_m6_h2: float = 3.5e5,  # Tuned to rated curve
        system_resistance_pa_per_m6_h2: float = 2.0e5,  # System piping losses
        fluid_inertance_kg_m4: float = 1e9,  # L_eq (High inertia for stability with dt=60s)
        initial_flow_m3_h: float = 10.0,
        rho_kg_m3: float = 1000.0
    ):
        """
        Initialize pump dynamics model.
        
        Args:
            pump_shutoff_pa: Pump shutoff pressure (Pa) - max head
            pump_resistance_pa_per_m6_h2: Pump curve slope (Pa·h²/m⁶)
            system_resistance_pa_per_m6_h2: System friction curve (Pa·h²/m⁶)
            fluid_inertance_kg_m4: Lumped inertance for ramp-up
            initial_flow_m3_h: Initial flowrate at t=0 (m³/h)
            rho_kg_m3: Fluid density (kg/m³)
        """
        self.pump_shutoff_pa = pump_shutoff_pa
        self.pump_resistance = pump_resistance_pa_per_m6_h2
        self.system_resistance = system_resistance_pa_per_m6_h2
        self.L_eq = fluid_inertance_kg_m4
        self.rho = rho_kg_m3
        
        self.Q_m3_h = initial_flow_m3_h  # Current flowrate
        self.dQ_dt_m3_h_s = 0.0           # Rate of change
        
        # Diagnostics
        self.p_pump_pa = self._pump_pressure(initial_flow_m3_h)
        self.p_system_pa = self._system_pressure(initial_flow_m3_h)
    
    def _pump_pressure(self, Q_m3_h: float) -> float:
        """Compute pump pressure from characteristic curve."""
        Q_m6_h2 = Q_m3_h ** 2
        return max(0, self.pump_shutoff_pa - self.pump_resistance * Q_m6_h2)
    
    def _system_pressure(self, Q_m3_h: float) -> float:
        """Compute system resistance pressure drop."""
        Q_m6_h2 = Q_m3_h ** 2
        return self.system_resistance * Q_m6_h2
    
    def step(self, dt_s: float, pump_speed_fraction: float = 1.0) -> float:
        """
        Advance flow by one timestep using sub-stepping for stability.
        
        Args:
            dt_s: Timestep (60 s for 1-minute)
            pump_speed_fraction: Pump speed relative to nominal (0-1)
        
        Returns:
            Q_new_m3_h: New flowrate (m³/h)
        """
        # Sub-stepping configuration
        max_sub_dt = 0.1  # 100ms max step for stability
        n_steps = math.ceil(dt_s / max_sub_dt)
        sub_dt = dt_s / n_steps
        
        for _ in range(n_steps):
            # Scaled pump pressure (proportional to speed²)
            p_pump_scaled = self._pump_pressure(self.Q_m3_h) * pump_speed_fraction ** 2
            p_system = self._system_pressure(self.Q_m3_h)
            
            # Pressure difference drives acceleration
            dp_pa = p_pump_scaled - p_system
            
            # Convert to flow rate units (m³/h)
            # d(Q_m3_s)/dt = dp / L_eq
            # d(Q_m3_h)/dt = 3600 * d(Q_m3_s)/dt = 3600 * dp / L_eq
            
            if self.L_eq > 0:
                dQ_dt_m3_h_s = (3600.0 / self.L_eq) * dp_pa
            else:
                dQ_dt_m3_h_s = 0.0
                
            # Forward Euler step
            self.Q_m3_h += dQ_dt_m3_h_s * sub_dt
            self.Q_m3_h = max(0, self.Q_m3_h)  # Flow cannot be negative
            
            # Update diagnostics for the last sub-step
            self.dQ_dt_m3_h_s = dQ_dt_m3_h_s
            self.p_pump_pa = p_pump_scaled
            self.p_system_pa = p_system
        
        return self.Q_m3_h
    
    def get_state(self) -> dict:
        """Return flow dynamics state."""
        return {
            'flow_m3_h': self.Q_m3_h,
            'flow_kg_s': self.Q_m3_h / 3600 * self.rho,
            'pump_pressure_pa': self.p_pump_pa,
            'pump_pressure_bar': self.p_pump_pa / 1e5,
            'system_pressure_pa': self.p_system_pa,
            'pressure_delta_pa': self.p_pump_pa - self.p_system_pa,
            'flow_rate_of_change_m3_h_s': self.dQ_dt_m3_h_s
        }


try:
    from h2_plant.optimization.coolprop_lut import CoolPropLUT
    COOLPROP_AVAILABLE = True
except ImportError:
    COOLPROP_AVAILABLE = False
    CoolPropLUT = None

class GasAccumulatorDynamics:
    """
    Gas buffer tank dynamics for H2 storage/discharge.
    Uses Real Gas EOS (CoolPropLUT) if available, otherwise Ideal Gas.
    Governance: Mass Accumulation -> Density -> Pressure (EOS)
    """
    
    def __init__(
        self,
        V_tank_m3: float = 1.0,
        T_tank_k: float = 298.15,
        initial_pressure_pa: float = 4e6,  # 40 bar
        R_gas_j_kg_k: float = 4124.0,  # H2 gas constant
        max_pressure_pa: float = None  # Configurable max pressure (default: 350 bar)
    ):
        """
        Initialize gas accumulator.
        
        Args:
            V_tank_m3: Tank volume (m³)
            T_tank_k: Tank temperature (K)
            initial_pressure_pa: Starting pressure (Pa)
            R_gas_j_kg_k: Gas constant (J/kg·K) - Used for fallback
            max_pressure_pa: Maximum allowed pressure (Pa). Defaults to HIGH_PRESSURE_PA (350 bar).
        """
        from h2_plant.core.constants import StorageConstants
        
        self.V = V_tank_m3
        self.T = T_tank_k
        self.P = initial_pressure_pa
        self.R = R_gas_j_kg_k
        self.max_pressure_pa = max_pressure_pa if max_pressure_pa is not None else StorageConstants.HIGH_PRESSURE_PA
        
        # Calculate initial mass using EOS
        self.M_kg = self._calculate_mass_from_pressure(self.P)
        
        # Diagnostics
        self.m_dot_in_kg_s = 0.0
        self.m_dot_out_kg_s = 0.0
    
    def _calculate_mass_from_pressure(self, P_pa: float) -> float:
        """Calculate Mass from Pressure using Real Gas EOS."""
        if COOLPROP_AVAILABLE and CoolPropLUT:
            try:
                rho = CoolPropLUT.PropsSI('D', 'P', P_pa, 'T', self.T, 'H2')
                return rho * self.V
            except Exception:
                pass # Fallback
        
        # Ideal Gas Fallback
        return (P_pa * self.V) / (self.R * self.T)

    def _calculate_pressure_from_mass(self, M_kg: float) -> float:
        """Calculate Pressure from Mass using Real Gas EOS."""
        if M_kg <= 0: return 0.0
        
        rho = M_kg / self.V
        
        if COOLPROP_AVAILABLE and CoolPropLUT:
            try:
                return CoolPropLUT.PropsSI('P', 'D', rho, 'T', self.T, 'H2')
            except Exception:
                pass # Fallback
        
        # Ideal Gas Fallback
        return (M_kg * self.R * self.T) / self.V

    def step(
        self,
        dt_s: float,
        m_dot_in_kg_s: float,
        m_dot_out_kg_s: float
    ) -> float:
        """
        Advance state by one timestep.
        Uses Mass Conservation (Exact) instead of dP/dt.
        
        Args:
            dt_s: Timestep (60 s for 1-minute)
            m_dot_in_kg_s: Compressor input (kg/s)
            m_dot_out_kg_s: H2 consumption (kg/s)
        
        Returns:
            P_new_pa: New tank pressure (Pa)
        """
        self.m_dot_in_kg_s = m_dot_in_kg_s
        self.m_dot_out_kg_s = m_dot_out_kg_s
        
        # 1. Exact Mass Balance
        delta_mass_kg = (m_dot_in_kg_s - m_dot_out_kg_s) * dt_s
        self.M_kg += delta_mass_kg
        self.M_kg = max(0.0, self.M_kg)
        
        # 2. Update Pressure from new Mass
        P_new_pa = self._calculate_pressure_from_mass(self.M_kg)
        
        # 3. Physical bounds (Safety Relief / Burst Disc logic)
        if P_new_pa > self.max_pressure_pa:
            P_new_pa = self.max_pressure_pa
            # Adjust mass to match max pressure (venting)
            self.M_kg = self._calculate_mass_from_pressure(P_new_pa)
        
        self.P = P_new_pa
        
        return self.P
    
    def get_state(self) -> dict:
        """Return accumulator state."""
        return {
            'pressure_pa': self.P,
            'pressure_bar': self.P / 1e5,
            'mass_kg': self.M_kg,
            'volume_m3': self.V,
            'temperature_k': self.T,
            'density_kg_m3': self.M_kg / self.V if self.V > 0 else 0,
            'mass_flow_in_kg_s': self.m_dot_in_kg_s,
            'mass_flow_out_kg_s': self.m_dot_out_kg_s
        }
