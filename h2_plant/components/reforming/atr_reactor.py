"""
Autothermal Reforming (ATR) Reactor Component.

This module implements an autothermal reformer for converting methane (biogas)
to synthesis gas (syngas) containing hydrogen. ATR combines partial oxidation
with steam reforming in a single vessel, achieving thermal self-sufficiency.

Chemical Principles:
    - **Partial Oxidation (POX)**: Exothermic reaction of methane with sub-stoichiometric
      oxygen to produce CO and H₂. Provides heat for endothermic reforming.
          CH₄ + ½O₂ → CO + 2H₂ (ΔH ≈ -36 kJ/mol)
    - **Steam Methane Reforming (SMR)**: Endothermic conversion of methane with
      steam to produce CO and H₂.
          CH₄ + H₂O ↔ CO + 3H₂ (ΔH ≈ +206 kJ/mol)
    - **Water-Gas Shift (WGS)**: Equilibrium reaction converting CO to CO₂ with
      additional H₂ production.
          CO + H₂O ↔ CO₂ + H₂ (ΔH ≈ -41 kJ/mol)

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Loads pre-computed interpolation tables from pickle file.
    - `step()`: Calculates stoichiometry and production based on oxygen feed rate.
    - `get_state()`: Exposes production rates, heat duty, and feed requirements.

Model Approach:
    The ATR model uses pre-computed interpolation functions derived from detailed
    Aspen Plus simulations. These curves relate oxygen feed rate (kmol/h) to
    product flows, heat duties, and feed requirements. Numba-accelerated interpolation
    provides O(log N) lookup performance.

References:
    - Aasberg-Petersen, K. et al. (2011). Natural gas to synthesis gas – Catalysts
      and catalytic processes. Journal of Natural Gas Science and Engineering, 3(2), 423-459.
    - Rostrup-Nielsen, J.R. (1993). Production of synthesis gas. Catalysis Today, 18(4), 305-324.
"""

import pickle
import numpy as np
import numba
from typing import Dict, Any, Optional

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream


# ============================================================================
# Interpolation Functions (Numba-Accelerated)
# ============================================================================

@numba.njit(fastmath=True, cache=True)
def linear_interp_scalar(x_eval: float, x_data: np.ndarray, y_data: np.ndarray) -> float:
    """
    Perform linear interpolation at a single evaluation point.

    Uses binary search to locate the bracketing interval, then applies
    linear interpolation formula. Extrapolation uses boundary values.

    Args:
        x_eval (float): Point at which to evaluate interpolant.
        x_data (np.ndarray): Sorted array of x coordinates (ascending).
        y_data (np.ndarray): Array of corresponding y values.

    Returns:
        float: Interpolated y value at x_eval.

    Note:
        Compiled with Numba JIT for O(log N) performance in hot simulation loops.
    """
    n = len(x_data)
    if n < 2:
        return y_data[0] if n > 0 else 0.0
    if x_eval <= x_data[0]:
        return y_data[0]
    elif x_eval >= x_data[n-1]:
        return y_data[n-1]

    # Binary search for bracketing interval
    left, right = 0, n - 1
    while right - left > 1:
        mid = (left + right) // 2
        if x_eval < x_data[mid]:
            right = mid
        else:
            left = mid

    x0, x1 = x_data[left], x_data[right]
    y0, y1 = y_data[left], y_data[right]
    t = (x_eval - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


@numba.njit(fastmath=True, cache=True)
def cubic_interp_scalar(x_eval: float, x_data: np.ndarray, y_data: np.ndarray) -> float:
    """
    Perform cubic Hermite interpolation at a single evaluation point.

    Uses Catmull-Rom spline formulation for smooth interpolation with
    continuous first derivatives. Provides better accuracy than linear
    interpolation for smooth underlying functions.

    Args:
        x_eval (float): Point at which to evaluate interpolant.
        x_data (np.ndarray): Sorted array of x coordinates (ascending).
        y_data (np.ndarray): Array of corresponding y values.

    Returns:
        float: Interpolated y value at x_eval.

    Note:
        Boundary points use reflected indices to maintain C1 continuity.
    """
    n = len(x_data)
    if n < 2:
        return y_data[0] if n > 0 else 0.0
    if x_eval <= x_data[0]:
        return y_data[0]
    elif x_eval >= x_data[n-1]:
        return y_data[n-1]

    # Binary search for bracketing interval
    left, right = 0, n - 1
    while right - left > 1:
        mid = (left + right) // 2
        if x_eval < x_data[mid]:
            right = mid
        else:
            left = mid

    # Select four points for cubic interpolation (boundary handling)
    if left == 0:
        i0, i1, i2, i3 = 0, 0, 1, 2
    elif left >= n - 2:
        i0, i1, i2, i3 = n-3, n-2, n-1, n-1
    else:
        i0, i1, i2, i3 = left-1, left, left+1, left+2

    x0, x1, x2, x3 = x_data[i0], x_data[i1], x_data[i2], x_data[i3]
    y0, y1, y2, y3 = y_data[i0], y_data[i1], y_data[i2], y_data[i3]

    # Hermite basis with Catmull-Rom tangent estimation
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
    """
    Optimized ATR lookup model with Numba-accelerated interpolation.

    Wraps pre-computed interpolation tables from Aspen Plus simulations,
    providing fast evaluation of ATR outputs as functions of oxygen feed rate.
    Each interpolation function maps F_O₂ (kmol/h) to a specific output quantity.

    Attributes:
        interp_data (Dict): Dictionary of interpolation arrays keyed by function name.
            Each entry contains (x_array, y_array, interpolation_kind).

    Example:
        >>> model = OptimizedATRModel(interp_data)
        >>> outputs = model.get_outputs(F_O2=50.0)
        >>> h2_rate = outputs['h2_production']  # kmol/h
    """

    def __init__(self, interp_data: Dict):
        """
        Initialize the optimized ATR model.

        Args:
            interp_data (Dict): Pre-extracted interpolation data containing x/y
                arrays and interpolation type for each output function.
        """
        self.interp_data = interp_data

    def evaluate_single(self, func_name: str, val: float) -> float:
        """
        Evaluate a single interpolation function.

        Selects appropriate interpolation method (linear or cubic) based on
        the stored kind specification.

        Args:
            func_name (str): Name of the function to evaluate (e.g., 'F_H2_func').
            val (float): Input value (oxygen flow rate in kmol/h).

        Returns:
            float: Interpolated output value.
        """
        if func_name not in self.interp_data:
            return 0.0
        x_data, y_data, kind = self.interp_data[func_name]
        if 'cubic' in str(kind) or kind == 3:
            return cubic_interp_scalar(val, x_data, y_data)
        return linear_interp_scalar(val, x_data, y_data)

    def get_outputs(self, F_O2: float) -> Dict[str, float]:
        """
        Calculate all ATR outputs for a given oxygen feed rate.

        Evaluates interpolation functions to determine hydrogen production,
        heat duties, and feed requirements. Heat duty is aggregated across
        multiple zones (preheaters, reactor sections).

        Args:
            F_O2 (float): Oxygen feed rate in kmol/h.

        Returns:
            Dict[str, float]: Output dictionary containing:
                - h2_production (float): Hydrogen production rate (kmol/h).
                - total_heat_duty (float): Net heat duty across all zones (kW).
                - biogas_required (float): Biogas (methane) feed rate (kmol/h).
                - steam_required (float): Steam feed rate (kmol/h).
                - water_required (float): Process water rate (kmol/h).
        """
        h2_prod = self.evaluate_single('F_H2_func', F_O2)

        # Aggregate heat duties from multiple reactor zones
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


# ============================================================================
# ATR Reactor Component
# ============================================================================

class ATRReactor(Component):
    """
    Autothermal Reforming reactor for hydrogen production from biogas.

    Converts methane-rich biogas to hydrogen via combined partial oxidation
    and steam reforming. Oxygen feed rate is the primary control variable,
    with biogas and steam consumed in stoichiometric proportion.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Loads interpolation model from pickle file and
          extracts Numba-compatible arrays.
        - `step()`: Determines limiting reagent, computes actual conversion,
          and updates output buffers.
        - `get_state()`: Returns production rates and heat duty for monitoring.

    The reactor uses input buffering to handle asynchronous feed delivery:
    reactants accumulate each timestep, and consumption is limited by the
    least available reagent (stoichiometric constraint).

    Attributes:
        max_flow_kg_h (float): Maximum design flow capacity (kg/h).
        model (OptimizedATRModel): Loaded interpolation model.
        h2_production_kmol_h (float): Current H₂ production rate.
        heat_duty_kw (float): Current net heat duty.

    Example:
        >>> atr = ATRReactor(component_id='ATR_01', max_flow_kg_h=1000.0)
        >>> atr.initialize(dt=1/60, registry=registry)
        >>> atr.receive_input('o2_in', oxygen_stream, 'oxygen')
        >>> atr.receive_input('biogas_in', biogas_stream, 'methane')
        >>> atr.step(t=0.0)
        >>> syngas = atr.get_output('h2_out')
    """

    def __init__(
        self,
        component_id: str,
        max_flow_kg_h: float,
        model_path: str = 'ATR_model_functions.pkl'
    ):
        """
        Initialize the ATR reactor component.

        Args:
            component_id (str): Unique identifier for this component instance.
            max_flow_kg_h (float): Maximum design throughput in kg/h.
            model_path (str): Path to pickle file containing interpolation
                functions from Aspen Plus. Default: 'ATR_model_functions.pkl'.
        """
        super().__init__()
        self.component_id = component_id
        self.max_flow_kg_h = max_flow_kg_h
        self.model_path = model_path
        self.model: Optional[OptimizedATRModel] = None

        # ====================================================================
        # State Variables (Instantaneous Rates)
        # ====================================================================
        self.oxygen_flow_kmol_h = 0.0
        self.h2_production_kmol_h = 0.0
        self.heat_duty_kw = 0.0
        self.biogas_input_kmol_h = 0.0
        self.steam_input_kmol_h = 0.0
        self.water_input_kmol_h = 0.0

        # ====================================================================
        # Input Buffers (Molar Accumulation)
        # ====================================================================
        # Buffers accumulate feed from upstream components; consumption
        # is limited by stoichiometry to prevent over-reaction.
        self.buffer_oxygen_kmol = 0.0
        self.buffer_biogas_kmol = 0.0
        self.buffer_steam_kmol = 0.0

        # ====================================================================
        # Output Buffers (Accumulated Production)
        # ====================================================================
        self._h2_output_buffer_kmol = 0.0
        self._offgas_output_buffer_kmol = 0.0
        self._heat_output_buffer_kw = 0.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Prepare the reactor for simulation execution.

        Loads and optimizes the ATR model from a pickle file containing
        scipy.interpolate objects. Extracts underlying arrays for Numba
        compatibility to enable fast JIT-compiled interpolation.

        Fulfills the Component Lifecycle Contract initialization phase.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.

        Note:
            If model loading fails, the component operates in pass-through
            mode with zero production.
        """
        super().initialize(dt, registry)
        try:
            with open(self.model_path, 'rb') as f:
                raw_model = pickle.load(f)

            # Extract arrays from scipy interp1d objects for Numba compatibility
            interp_data = {}
            func_names = [
                'F_H2_func', 'H01_Q_func', 'H02_Q_func', 'H04_Q_func',
                'F_bio_func', 'F_steam_func', 'F_water_func',
                'xCO2_offgas_func', 'xH2_offgas_func'
            ]

            for name in func_names:
                if name in raw_model:
                    f_obj = raw_model[name]
                    interp_data[name] = (
                        f_obj.x.astype(np.float64),
                        f_obj.y.astype(np.float64),
                        f_obj.kind if hasattr(f_obj, 'kind') else 'linear'
                    )

            self.model = OptimizedATRModel(interp_data)

        except Exception as e:
            import warnings
            warnings.warn(f"ATRReactor {self.component_id}: Failed to load model from "
                         f"{self.model_path}. Error: {e}")
            self.model = None

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Implements the stoichiometric consumption logic:
        1. Calculate target reaction rate from available oxygen.
        2. Query model for stoichiometric feed requirements.
        3. Determine limiting reagent (oxygen, biogas, or steam).
        4. Scale reaction to limiting factor and consume feeds.
        5. Accumulate products in output buffers.

        This approach ensures mass balance closure regardless of
        asynchronous feed delivery from upstream components.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)
        if not self.model:
            return

        # ====================================================================
        # Limiting Reagent Determination
        # ====================================================================
        # Oxygen availability sets target reaction extent; biogas and steam
        # may further limit based on their stoichiometric requirements.
        available_o2_kmol = self.buffer_oxygen_kmol
        target_o2_rate_kmol_h = available_o2_kmol / self.dt if self.dt > 0 else 0.0

        if target_o2_rate_kmol_h > 1e-6:
            outputs = self.model.get_outputs(target_o2_rate_kmol_h)

            req_biogas_rate = outputs['biogas_required']
            req_steam_rate = outputs['steam_required']

            req_biogas_total = req_biogas_rate * self.dt
            req_steam_total = req_steam_rate * self.dt

            # Compute limiting factor as minimum availability ratio
            limit_factor = 1.0

            if req_biogas_total > 1e-9:
                if self.buffer_biogas_kmol < req_biogas_total:
                    limit_factor = min(limit_factor,
                                      self.buffer_biogas_kmol / req_biogas_total)

            if req_steam_total > 1e-9:
                if self.buffer_steam_kmol < req_steam_total:
                    limit_factor = min(limit_factor,
                                      self.buffer_steam_kmol / req_steam_total)

            # ================================================================
            # Consumption and Production
            # ================================================================
            actual_o2_rate = target_o2_rate_kmol_h * limit_factor

            # Re-evaluate at actual rate to capture non-linearities
            outputs_actual = self.model.get_outputs(actual_o2_rate)

            consumed_biogas = outputs_actual['biogas_required'] * self.dt
            consumed_steam = outputs_actual['steam_required'] * self.dt
            consumed_o2 = actual_o2_rate * self.dt

            # Debit input buffers
            self.buffer_biogas_kmol -= consumed_biogas
            self.buffer_steam_kmol -= consumed_steam
            self.buffer_oxygen_kmol -= consumed_o2

            # Clamp to prevent negative buffers from numerical precision
            self.buffer_biogas_kmol = max(0.0, self.buffer_biogas_kmol)
            self.buffer_steam_kmol = max(0.0, self.buffer_steam_kmol)
            self.buffer_oxygen_kmol = max(0.0, self.buffer_oxygen_kmol)

            # Update state variables
            self.h2_production_kmol_h = outputs_actual['h2_production']
            self.heat_duty_kw = outputs_actual['total_heat_duty']

            self.oxygen_flow_kmol_h = actual_o2_rate
            self.biogas_input_kmol_h = outputs_actual['biogas_required']
            self.steam_input_kmol_h = outputs_actual['steam_required']
            self.water_input_kmol_h = outputs_actual['water_required']

            # Accumulate outputs for downstream delivery
            self._h2_output_buffer_kmol += self.h2_production_kmol_h * self.dt
            self._heat_output_buffer_kw += self.heat_duty_kw * self.dt
            # Offgas production approximated as 10% of H₂ (simplified CO₂/unreacted)
            self._offgas_output_buffer_kmol += 0.1 * self.h2_production_kmol_h * self.dt

        else:
            # No oxygen feed: reactor idle
            self.h2_production_kmol_h = 0.0
            self.heat_duty_kw = 0.0
            self.oxygen_flow_kmol_h = 0.0

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access, returning
        key performance indicators for monitoring and logging.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - component_id (str): Instance identifier.
                - h2_production_kmol_h (float): Current H₂ rate (kmol/h).
                - heat_duty_kw (float): Net heat duty (kW, positive = released).
                - biogas_input_kmol_h (float): Biogas consumption rate (kmol/h).
                - water_input_kmol_h (float): Process water rate (kmol/h).
        """
        return {
            **super().get_state(),
            'component_id': self.component_id,
            'h2_production_kmol_h': self.h2_production_kmol_h,
            'heat_duty_kw': self.heat_duty_kw,
            'biogas_input_kmol_h': self.biogas_input_kmol_h,
            'water_input_kmol_h': self.water_input_kmol_h
        }

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve the output stream from a specified port.

        Converts buffered molar production to Stream objects with appropriate
        mass flow, temperature, and composition for downstream components.

        Args:
            port_name (str): Port identifier ('h2_out', 'syngas_out',
                'offgas_out', or 'heat_out').

        Returns:
            Stream: For gas ports, a Stream object with mass flow at reformer
                outlet conditions (~900 K, 3 bar). For 'heat_out', returns
                float heat duty in kW.

        Raises:
            ValueError: If port_name is not a valid output port.
        """
        if port_name in ['h2_out', 'syngas_out']:
            flow_rate = self._h2_output_buffer_kmol / self.dt if self.dt > 0 else 0.0
            return Stream(
                mass_flow_kg_h=flow_rate * 2.016,
                temperature_k=900.0,
                pressure_pa=3e5,
                composition={'H2': 1.0},
                phase='gas'
            )
        elif port_name == 'offgas_out':
            flow_rate = self._offgas_output_buffer_kmol / self.dt if self.dt > 0 else 0.0
            return Stream(
                mass_flow_kg_h=flow_rate * 28.0,
                temperature_k=900.0,
                pressure_pa=3e5,
                composition={'CO2': 0.7, 'H2': 0.3},
                phase='gas'
            )
        elif port_name == 'heat_out':
            return self._heat_output_buffer_kw / self.dt if self.dt > 0 else 0.0
        else:
            raise ValueError(f"Unknown output port '{port_name}' on {self.component_id}")

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """
        Accept an input stream at the specified port.

        Converts incoming mass flow to molar basis and accumulates in
        appropriate buffer for stoichiometric reaction limiting.

        Args:
            port_name (str): Target port ('o2_in', 'biogas_in', or 'steam_in').
            value (Any): Stream object or scalar flow rate.
            resource_type (str): Resource classification hint.

        Returns:
            float: Flow rate accepted (in original units), or 0.0 if rejected.

        Note:
            Molar masses used: O₂=32, CH₄=16, H₂O=18 kg/kmol.
        """
        if isinstance(value, Stream):
            mass_kg = value.mass_flow_kg_h * self.dt

            if port_name == 'o2_in':
                kmol = mass_kg / 32.0
                self.buffer_oxygen_kmol += kmol
                return value.mass_flow_kg_h
            elif port_name == 'biogas_in':
                kmol = mass_kg / 16.0
                self.buffer_biogas_kmol += kmol
                return value.mass_flow_kg_h
            elif port_name == 'steam_in':
                kmol = mass_kg / 18.0
                self.buffer_steam_kmol += kmol
                return value.mass_flow_kg_h

        elif isinstance(value, (int, float)):
            # Scalar input assumed to be rate in kmol/h
            amount_kmol = value * self.dt
            if port_name == 'o2_in':
                self.buffer_oxygen_kmol += amount_kmol
            elif port_name == 'biogas_in':
                self.buffer_biogas_kmol += amount_kmol
            elif port_name == 'steam_in':
                self.buffer_steam_kmol += amount_kmol
            return value

        return 0.0

    def extract_output(self, port_name: str, amount: float, resource_type: str) -> None:
        """
        Clear output buffer after downstream extraction.

        Called by the orchestrator after product has been delivered to
        downstream components, preventing double-counting.

        Args:
            port_name (str): Port from which product was extracted.
            amount (float): Amount extracted (not used; buffer is cleared).
            resource_type (str): Resource classification hint.
        """
        if port_name == 'h2_out':
            self._h2_output_buffer_kmol = 0.0
        elif port_name == 'offgas_out':
            self._offgas_output_buffer_kmol = 0.0
        elif port_name == 'heat_out':
            self._heat_output_buffer_kw = 0.0

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions with keys:
                - biogas_in: Methane-rich feed gas from biogas plant.
                - steam_in: Process steam for reforming reaction.
                - o2_in: Oxygen for partial oxidation.
                - h2_out: Hydrogen-rich syngas product.
                - offgas_out: CO₂-rich tail gas for venting or capture.
                - heat_out: Net heat duty for thermal integration.
        """
        return {
            'biogas_in': {'type': 'input', 'resource_type': 'methane', 'units': 'kmol/h'},
            'steam_in': {'type': 'input', 'resource_type': 'steam', 'units': 'kmol/h'},
            'o2_in': {'type': 'input', 'resource_type': 'oxygen', 'units': 'kmol/h'},
            'h2_out': {'type': 'output', 'resource_type': 'hydrogen', 'units': 'kmol/h'},
            'offgas_out': {'type': 'output', 'resource_type': 'syngas', 'units': 'kmol/h'},
            'heat_out': {'type': 'output', 'resource_type': 'heat', 'units': 'kW'}
        }
