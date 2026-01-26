"""
Central Cooling Utility Manager.

This service component simulates the plant's shared heat rejection utilities:
1.  **Central Dry Cooler Bank (Glycol Loop)**: Serves process heat exchangers (DryCooler nodes).
    -   Physics: Cross-flow ACHE (Air-Cooled Heat Exchanger).
    -   Variables: Aggregated Duty -> Glycol Supply Temperature.
    
2.  **Central Cooling Tower (Water Loop)**: Serves Chillers and Compressors.
    -   Physics: Evaporative Cooling (Merkel/Approach model).
    -   Variables: Aggregated Duty -> Cooling Water Supply Temperature.

Architecture:
    Components (DryCoolers/Chillers) register their loads during their `step()`.
    The Manager calculates the resulting loop temperatures for the *next* timestep.
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

from h2_plant.core.component import Component
from h2_plant.optimization import numba_ops
from h2_plant.core.constants import DryCoolerIndirectConstants as DCC

logger = logging.getLogger(__name__)


class CoolingManager(Component):
    """
    Central utility manager for shared cooling loops.

    This service component manages the thermal state of centralized cooling systems:
    1. Glycol Loop (Dry Cooler Bank) - for process heat exchangers
    2. Cooling Water Loop (Cooling Tower) - for chillers and compressors

    Components register their loads each timestep, and the manager calculates
    the supply temperatures for the next timestep.
    """

    def __init__(self, component_id: str = "cooling_manager", **kwargs) -> None:
        """
        Initialize the CoolingManager.

        Args:
            component_id (str): Unique identifier. Default: "cooling_manager".
            **kwargs: Configuration overrides:
                - glycol_inventory_kg (float): Total glycol mass in loop. Default: 5000.0.
                - initial_glycol_temp_c (float): Initial glycol supply temp. Default: 25.0.
                - dc_total_area_m2 (float): Total dry cooler area. Default: 2000.0.
                - dc_u_value (float): Dry cooler U-value (W/m2K). Default: 35.0.
                - dc_air_flow_kg_s (float): Air flow rate. Default: 500.0.
                - initial_cw_temp_c (float): Initial cooling water temp. Default: 20.0.
                - tower_approach_k (float): Cooling tower approach. Default: 4.0.
        """
        super().__init__()
        self.component_id = component_id

        # --- System 1: Glycol Loop (Dry Cooler Bank) ---
        self.glycol_total_inventory_kg = kwargs.get('glycol_inventory_kg', 5000.0)
        self.glycol_supply_temp_c = kwargs.get('initial_glycol_temp_c', 25.0)
        self.glycol_return_temp_c = self.glycol_supply_temp_c
        self.glycol_flow_total_kg_s = 0.0  # Sum of all users
        self.glycol_duty_kw = 0.0           # Sum of all users

        # Central Hardware Specs
        self.dc_total_area_m2 = kwargs.get('dc_total_area_m2', 2000.0)
        self.dc_u_value = kwargs.get('dc_u_value', 35.0)  # W/m2K
        self.dc_air_flow_kg_s = kwargs.get('dc_air_flow_kg_s', 500.0)
        self.glycol_fan_power_kw = 0.0

        # --- System 2: Cooling Water Loop (Cooling Tower) ---
        self.cw_supply_temp_c = kwargs.get('initial_cw_temp_c', 20.0)
        self.cw_return_temp_c = self.cw_supply_temp_c
        self.cw_flow_total_kg_s = 0.0
        self.cw_duty_kw = 0.0

        # Tower Specs
        self.tower_design_approach_k = kwargs.get('tower_approach_k', 4.0)
        self.tower_design_load_kw = kwargs.get('tower_design_load_kw', 5000.0)
        self.tower_fan_power_kw = 0.0

        # Ambient Conditions (Static for now, could link to weather)
        self.t_dry_bulb_c = kwargs.get('t_dry_bulb_c', 25.0)
        self.t_wet_bulb_c = kwargs.get('t_wet_bulb_c', 18.0)

        # Load Accumulators (Reset every step)
        self._current_step_glycol_load_kw = 0.0
        self._current_step_glycol_flow_kg_s = 0.0
        self._current_step_cw_load_kw = 0.0
        self._current_step_cw_flow_kg_s = 0.0

        # Inertia smoothing factor (0.2 = slow response, 1.0 = instant)
        self._alpha = kwargs.get('inertia_alpha', 0.2)

    def initialize(self, dt: float, registry: Any) -> None:
        """Prepare component for simulation."""
        super().initialize(dt, registry)

    def register_glycol_load(self, duty_kw: float, flow_kg_s: float, return_temp_c: float = None, source_id: str = "unknown") -> None:
        """
        Called by DryCoolers to register their demand.

        Args:
            duty_kw (float): Heat load added to glycol loop (kW).
            flow_kg_s (float): Glycol flow rate through the user (kg/s).
            return_temp_c (float, optional): Glycol return temperature from user.
            source_id (str): ID of the component registering the load.
        """
        if duty_kw > 0.1:
             print(f"DEBUG_LOAD: Manager received {duty_kw:.2f} kW from {source_id}", flush=True)
        self._current_step_glycol_load_kw += duty_kw
        self._current_step_glycol_flow_kg_s += flow_kg_s

    def register_cw_load(self, duty_kw: float, flow_kg_s: float) -> None:
        """
        Called by Chillers to register their heat rejection.

        Args:
            duty_kw (float): Heat rejected to cooling water loop (kW).
            flow_kg_s (float): Cooling water flow rate (kg/s).
        """
        self._current_step_cw_load_kw += duty_kw
        self._current_step_cw_flow_kg_s += flow_kg_s

    def step(self, t: float) -> None:
        """
        Calculate central loop temperatures based on registered loads.

        This updates the 'Supply Temperature' available for the NEXT timestep.
        """
        super().step(t)
        alpha = self._alpha

        # --- 1. Solve Central Dry Cooler (Glycol) ---
        # Update state from accumulators
        self.glycol_duty_kw = self._current_step_glycol_load_kw
        self.glycol_flow_total_kg_s = max(self._current_step_glycol_flow_kg_s, 1.0)  # Avoid div/0

        # Calculate Mix Return Temperature (Energy Balance)
        # Q = m * Cp * (T_return - T_supply)  => T_return = T_supply + Q/(m*Cp)
        cp_glycol = 3600.0  # Approx J/kgK (40% glycol)
        if self.glycol_flow_total_kg_s > 0:
            delta_t_loop = (self.glycol_duty_kw * 1000.0) / (self.glycol_flow_total_kg_s * cp_glycol)
        else:
            delta_t_loop = 0.0
        self.glycol_return_temp_c = self.glycol_supply_temp_c + delta_t_loop

        # Solve Heat Rejection to Air (e-NTU Crossflow)
        t_air_in_k = self.t_dry_bulb_c + 273.15
        t_glycol_in_k = self.glycol_return_temp_c + 273.15

        C_glycol = self.glycol_flow_total_kg_s * cp_glycol
        C_air = self.dc_air_flow_kg_s * DCC.CP_AIR_J_KG_K

        C_min = min(C_glycol, C_air)
        C_max = max(C_glycol, C_air)

        if C_min > 0 and C_max > 0:
            R = C_min / C_max
            NTU = (self.dc_u_value * self.dc_total_area_m2) / C_min

            # Calculate Effectiveness
            eff = numba_ops.dry_cooler_ntu_effectiveness(NTU, R)

            # Calculate Heat Rejected
            q_max = C_min * (t_glycol_in_k - t_air_in_k)
            q_rejected_w = eff * q_max

            # Calculate New Supply Temperature
            # T_supply = T_return - Q / C_glycol
            if C_glycol > 0:
                t_glycol_out_k = t_glycol_in_k - q_rejected_w / C_glycol
            else:
                t_glycol_out_k = t_glycol_in_k
        else:
            t_glycol_out_k = t_glycol_in_k

        # Calculate Fan Power for Central Dry Cooler Bank
        vol_air = self.dc_air_flow_kg_s / DCC.RHO_AIR_KG_M3
        power_j_s = (vol_air * DCC.DP_AIR_DESIGN_PA) / DCC.ETA_FAN
        self.glycol_fan_power_kw = power_j_s / 1000.0

        # Simple Thermal Inertia (Blending old supply with new result)
        new_supply_c = t_glycol_out_k - 273.15
        self.glycol_supply_temp_c = (1 - alpha) * self.glycol_supply_temp_c + alpha * new_supply_c

        # --- 2. Solve Cooling Tower (Water) ---
        self.cw_duty_kw = self._current_step_cw_load_kw

        # Simplified Cooling Tower Model (Approach Method)
        # T_cw_supply = T_wet_bulb + Approach
        # In reality, Approach rises with Load/Capacity ratio.
        # Linear approximation:
        if self.tower_design_load_kw > 0:
            load_factor = min(self.cw_duty_kw / self.tower_design_load_kw, 2.0)
        else:
            load_factor = 0.0
        actual_approach = self.tower_design_approach_k * (0.5 + 0.5 * load_factor)

        target_cw_temp = self.t_wet_bulb_c + actual_approach

        # Inertia
        self.cw_supply_temp_c = (1 - alpha) * self.cw_supply_temp_c + alpha * target_cw_temp

        # Tower fan power estimation (simplified)
        self.tower_fan_power_kw = 0.02 * abs(self.cw_duty_kw)  # ~2% of heat load

        # Reset Accumulators for next step
        self._current_step_glycol_load_kw = 0.0
        self._current_step_glycol_flow_kg_s = 0.0
        self._current_step_cw_load_kw = 0.0
        self._current_step_cw_flow_kg_s = 0.0

    def get_state(self) -> Dict[str, Any]:
        """Return current state of cooling utilities."""
        return {
            **super().get_state(),
            'glycol_supply_temp_c': self.glycol_supply_temp_c,
            'glycol_return_temp_c': self.glycol_return_temp_c,
            'glycol_duty_total_kw': self.glycol_duty_kw,
            'glycol_flow_total_kg_s': self.glycol_flow_total_kg_s,
            'cw_supply_temp_c': self.cw_supply_temp_c,
            'cw_duty_total_kw': self.cw_duty_kw,
            't_dry_bulb_c': self.t_dry_bulb_c,
            't_wet_bulb_c': self.t_wet_bulb_c,
            't_wet_bulb_c': self.t_wet_bulb_c,
            'tower_fan_power_kw': self.tower_fan_power_kw,
            'glycol_fan_power_kw': self.glycol_fan_power_kw,
            'power_kw': self.tower_fan_power_kw + self.glycol_fan_power_kw 
        }

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """CoolingManager has no physical ports (service component)."""
        return {}

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """CoolingManager doesn't receive inputs via ports."""
        return 0.0

    def get_output(self, port_name: str) -> Any:
        """CoolingManager doesn't provide outputs via ports."""
        return None
