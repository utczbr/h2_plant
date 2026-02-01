"""
Thermal Inertia Model for Electrolyzer Stack
Implements first-order ODE: dT/dt = (Q_in - Q_out - Q_loss) / C_thermal
"""

import math

class ThermalInertiaModel:
    """
    Lumped-parameter thermal model for electrolyzer stack.
    
    Attributes:
        C_thermal_J_K: Thermal mass (J/K) - invariant
        h_A_passive_W_K: Passive convection coefficient × area (W/K)
        T_ambient_K: Ambient temperature (K)
        m_dot_cool_kg_s: Cooling water mass flow rate (kg/s)
        Cp_cool: Specific heat of coolant (J/kg·K)
    """
    
    def __init__(
        self,
        C_thermal_J_K: float = 2.6e6,      # ~5 MW stack + fluid
        h_A_passive_W_K: float = 100.0,    # Natural convection
        T_ambient_K: float = 298.15,       # 25°C
        T_initial_K: float = 333.15,       # 60°C startup
        Cp_cool_J_kg_K: float = 4180.0,    # Water
        max_cooling_kw: float = 500.0      # Chiller capacity
    ):
        self.C_thermal_J_K = C_thermal_J_K
        self.h_A_passive_W_K = h_A_passive_W_K
        self.T_ambient_K = T_ambient_K
        self.T_K = T_initial_K
        self.Cp_cool = Cp_cool_J_kg_K
        self.max_cooling_kw = max_cooling_kw
        
        # Control setpoint
        self.T_setpoint_K = 333.15  # 60°C
        self.K_p_cooling = 150.0    # Proportional gain (W/K)
        
        # Diagnostics
        self.heat_generated_W = 0.0
        self.heat_removed_W = 0.0
        self.heat_lost_passive_W = 0.0
        self.tau_thermal_s = self._compute_time_constant()
    
    def _compute_time_constant(self) -> float:
        """Compute thermal time constant τ = C / (hA + ṁ·Cp)"""
        denominator = self.h_A_passive_W_K + 0.01 * self.Cp_cool  # Assume 0.01 kg/s default flow
        return self.C_thermal_J_K / denominator if denominator > 0 else float('inf')
    
    def step(
        self,
        dt_s: float,
        heat_generated_W: float,
        T_control_K: float = None,
        m_dot_cool_kg_s: float = 0.0
    ) -> float:
        """
        Advance temperature by one timestep using forward Euler.
        
        Args:
            dt_s: Timestep duration (seconds) - typically 60 for 1-minute resolution
            heat_generated_W: Heat from electrochemistry (W)
            T_control_K: Controller setpoint (K); uses self.T_setpoint if None
            m_dot_cool_kg_s: Cooling water mass flow (kg/s)
        
        Returns:
            T_new_K: New temperature (K)
        """
        if T_control_K is None:
            T_control_K = self.T_setpoint_K
        
        # 1. Passive convective loss to ambient
        self.heat_lost_passive_W = self.h_A_passive_W_K * (self.T_K - self.T_ambient_K)
        
        # 2. Forced cooling (proportional controller)
        error_K = self.T_K - T_control_K
        Q_cool_setpoint_W = self.K_p_cooling * max(0, error_K)  # Only cool if above setpoint
        self.heat_removed_W = min(Q_cool_setpoint_W, self.max_cooling_kw * 1000)
        
        # Alternative: if chiller flow rate is specified
        if m_dot_cool_kg_s > 0:
            Q_cool_from_flow_W = m_dot_cool_kg_s * self.Cp_cool * max(0, self.T_K - T_control_K)
            self.heat_removed_W = min(Q_cool_from_flow_W, self.max_cooling_kw * 1000)
        
        # 3. Store heat generated for diagnostics
        self.heat_generated_W = heat_generated_W
        
        # 4. Forward Euler integration: dT = (Q_in - Q_out - Q_loss) / C * dt
        dT_K = (self.heat_generated_W - self.heat_removed_W - self.heat_lost_passive_W) / self.C_thermal_J_K * dt_s
        T_new_K = self.T_K + dT_K
        
        # 5. Bounds checking (physical limits)
        T_new_K = max(self.T_ambient_K, min(T_new_K, 373.15))  # Clamp to [25°C, 100°C]
        
        self.T_K = T_new_K
        return T_new_K
    
    def get_state(self) -> dict:
        """Return thermal state for diagnostics."""
        return {
            'temperature_K': self.T_K,
            'temperature_C': self.T_K - 273.15,
            'heat_generated_W': self.heat_generated_W,
            'heat_removed_W': self.heat_removed_W,
            'heat_lost_passive_W': self.heat_lost_passive_W,
            'thermal_time_constant_s': self.tau_thermal_s,
            'error_from_setpoint_K': self.T_K - self.T_setpoint_K
        }
