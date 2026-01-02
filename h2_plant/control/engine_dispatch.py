"""
Integrated Dispatch Strategy for SimulationEngine.
OPTIMIZED: Pre-bound array access and zero-flow guarding.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, TYPE_CHECKING, List, Tuple
import numpy as np
import logging

from h2_plant.control.dispatch import (
    DispatchInput,
    DispatchState,
    DispatchResult,
    DispatchStrategy as BaseDispatchStrategy,
    ReferenceHybridStrategy,
    SoecOnlyStrategy
)

if TYPE_CHECKING:
    from h2_plant.core.component_registry import ComponentRegistry
    from h2_plant.config.plant_config import SimulationContext

# Import specific component types for type checking
from h2_plant.components.thermal.chiller import Chiller
from h2_plant.components.separation.coalescer import Coalescer
from h2_plant.components.purification.deoxo_reactor import DeoxoReactor
from h2_plant.components.separation.psa import PSA
from h2_plant.components.separation.knock_out_drum import KnockOutDrum
from h2_plant.components.separation.hydrogen_cyclone import HydrogenMultiCyclone
from h2_plant.components.thermal.interchanger import Interchanger
from h2_plant.components.compression.compressor_single import CompressorSingle
from h2_plant.components.cooling.dry_cooler import DryCooler
from h2_plant.components.thermal.heat_exchanger import HeatExchanger
from h2_plant.components.thermal.electric_boiler import ElectricBoiler
from h2_plant.components.water.drain_recorder_mixer import DrainRecorderMixer

logger = logging.getLogger(__name__)


@dataclass
class IntegratedDispatchState:
    P_soec_prev: float = 0.0
    force_sell: bool = False
    step_idx: int = 0
    cumulative_h2_kg: float = 0.0


@dataclass
class StreamRecorder:
    """
    Helper struct to hold pre-resolved array references for fast recording.
    Eliminates dict lookups and string formatting in the hot loop.
    """
    component: Any
    stream_attr: str  # e.g., 'outlet_stream'
    temp_arr: np.ndarray
    press_arr: np.ndarray
    flow_arr: np.ndarray
    h2o_frac_arr: np.ndarray
    mole_arrs: Tuple[np.ndarray, ...] # (H2, O2, N2, H2O, CH4, CO2)
    
    # Specific component metric arrays (optional)
    extra_metric_arrs: List[Tuple[str, np.ndarray]] = field(default_factory=list)


class EngineDispatchStrategy(ABC):
    @abstractmethod
    def initialize(self, registry: 'ComponentRegistry', context: 'SimulationContext', total_steps: int) -> None:
        pass

    @abstractmethod
    def decide_and_apply(self, t: float, prices: np.ndarray, wind: np.ndarray) -> None:
        pass

    @abstractmethod
    def get_history(self) -> Dict[str, np.ndarray]:
        pass


class HybridArbitrageEngineStrategy(ReferenceHybridStrategy):
    def __init__(self):
        self._registry: Optional['ComponentRegistry'] = None
        self._context: Optional['SimulationContext'] = None
        self._inner_strategy: Optional[BaseDispatchStrategy] = None
        self._state = IntegratedDispatchState()

        # Component references
        self._soec = None
        self._pem = None
        
        # Performance: Pre-bound recorders
        self._recorders: List[StreamRecorder] = []

        # Capacity cache
        self._soec_capacity: float = 0.0
        self._pem_max: float = 0.0

        # Pre-allocated history arrays
        self._history: Dict[str, np.ndarray] = {}
        self._total_steps: int = 0

    def initialize(
        self,
        registry: 'ComponentRegistry',
        context: 'SimulationContext',
        total_steps: int
    ) -> None:
        self._registry = registry
        self._context = context
        self._total_steps = total_steps

        # Detect topology
        self._soec = self._find_soec(registry)
        self._pem = self._find_pem(registry)
        
        if self._soec and not self._pem:
            logger.info("Topology detected: SOEC Only. Using SoecOnlyStrategy.")
            self._inner_strategy = SoecOnlyStrategy()
        else:
            logger.info("Topology detected: Hybrid. Using ReferenceHybridStrategy.")
            self._inner_strategy = ReferenceHybridStrategy()

        # Cache component capacities
        if self._soec:
            spec = context.physics.soec_cluster
            self._soec_capacity = spec.num_modules * spec.max_power_nominal_mw * spec.optimal_limit
        if self._pem:
            self._pem_max = context.physics.pem_system.max_power_mw

        # 1. Allocate Base History Arrays
        self._history = {
            'minute': np.zeros(total_steps, dtype=np.int32),
            'P_offer': np.zeros(total_steps, dtype=np.float64),
            'P_soec_actual': np.zeros(total_steps, dtype=np.float64),
            'P_pem': np.zeros(total_steps, dtype=np.float64),
            'P_sold': np.zeros(total_steps, dtype=np.float64),
            'spot_price': np.zeros(total_steps, dtype=np.float64),
            'h2_kg': np.zeros(total_steps, dtype=np.float64),
            'H2_soec_kg': np.zeros(total_steps, dtype=np.float64),
            'H2_pem_kg': np.zeros(total_steps, dtype=np.float64),
            'cumulative_h2_kg': np.zeros(total_steps, dtype=np.float64),
            'steam_soec_kg': np.zeros(total_steps, dtype=np.float64),
            'H2O_soec_out_kg': np.zeros(total_steps, dtype=np.float64),
            'soec_active_modules': np.zeros(total_steps, dtype=np.int32),
            'H2O_pem_kg': np.zeros(total_steps, dtype=np.float64),
            'O2_pem_kg': np.zeros(total_steps, dtype=np.float64),
            'pem_V_cell': np.zeros(total_steps, dtype=np.float64),
            'P_bop_mw': np.zeros(total_steps, dtype=np.float64),
            'tank_level_kg': np.zeros(total_steps, dtype=np.float64),
            'tank_pressure_bar': np.zeros(total_steps, dtype=np.float64),
            'compressor_power_kw': np.zeros(total_steps, dtype=np.float64),
            'sell_decision': np.zeros(total_steps, dtype=np.int8),
            # Specific PEM Impurity
            'PEM_o2_impurity_ppm_mol': np.zeros(total_steps, dtype=np.float64)
        }

        # 2. Identify Components & Pre-Bind Arrays
        self._recorders = []
        self._prebind_recorders(registry, total_steps)
        
        # SOEC Specific Modules
        if self._soec:
            num_modules = getattr(self._soec, 'num_modules', 0)
            for i in range(num_modules):
                self._history[f"soec_module_powers_{i+1}"] = np.zeros(total_steps, dtype=np.float64)

        self._state = IntegratedDispatchState()
        logger.info(f"Initialized HybridArbitrageEngineStrategy with {total_steps} steps")

    def _prebind_recorders(self, registry: 'ComponentRegistry', total_steps: int) -> None:
        """
        Scan registry, allocate arrays, and bind them to StreamRecorder objects.
        This enables O(1) access during the simulation loop.
        """
        # Mapping: Class Type -> (Stream Attribute Name, List of (Metric Name, Metric Attribute))
        CONFIG_MAP = {
            Chiller: ('outlet_stream', [('cooling_load_kw', 'cooling_load_kw'), ('electrical_power_kw', 'electrical_power_kw')]),
            Coalescer: ('output_stream', [('delta_p_bar', 'delta_p_bar'), ('drain_flow_kg_h', 'drain_flow_kg_h')]),
            DeoxoReactor: ('output_stream', [('outlet_o2_ppm_mol', 'outlet_o2_ppm_mol'), ('peak_temp_c', 'peak_temp_c')]),
            PSA: ('purified_gas_out', [('outlet_o2_ppm_mol', 'outlet_o2_ppm_mol')]), # Note: PSA returns stream via method, need handling
            KnockOutDrum: ('_gas_outlet_stream', [('water_removed_kg_h', 'water_removed_kg_h')]),
            HydrogenMultiCyclone: ('_outlet_stream', [('pressure_drop_mbar', 'pressure_drop_mbar')]),
            CompressorSingle: ('outlet', [('power_kw', 'power_kw')]),
            DryCooler: ('outlet_stream', [('heat_rejected_kw', 'heat_rejected_kw'), ('fan_power_kw', 'fan_power_kw')]),
            HeatExchanger: ('output_stream', [('heat_removed_kw', 'heat_removed_kw')]),
            ElectricBoiler: ('fluid_out', [('power_input_kw', 'power_input_kw')]),
            Interchanger: ('hot_out', [('q_transferred_kw', 'q_transferred_kw')])
        }

        # Also add SOEC Cluster if it has a stream
        soec = self._soec
        if soec:
             # Manually add SOEC
             cid = soec.component_id if hasattr(soec, 'component_id') else 'SOEC_Cluster'
             self._alloc_stream_history(cid, total_steps)
             # SOEC stream is often constructed on fly, handled specially in loop
        
        for cid, comp in registry.list_components():
            ctype = type(comp)
            
            # Special handling for inheritance or if exact type in map
            config = None
            for base_cls, conf in CONFIG_MAP.items():
                if isinstance(comp, base_cls):
                    config = conf
                    break
            
            if config:
                stream_attr, metrics = config
                
                # 1. Allocate Stream Arrays
                self._alloc_stream_history(cid, total_steps)
                
                # 2. Allocate Metric Arrays
                metric_recorders = []
                for hist_suffix, obj_attr in metrics:
                    hist_key = f"{cid}_{hist_suffix}"
                    self._history[hist_key] = np.zeros(total_steps, dtype=np.float64)
                    metric_recorders.append((obj_attr, self._history[hist_key]))
                
                # 3. Create Recorder
                # Bind the arrays we just created
                recorder = StreamRecorder(
                    component=comp,
                    stream_attr=stream_attr,
                    temp_arr=self._history[f"{cid}_outlet_temp_c"],
                    press_arr=self._history[f"{cid}_outlet_pressure_bar"],
                    flow_arr=self._history[f"{cid}_outlet_mass_flow_kg_h"],
                    h2o_frac_arr=self._history[f"{cid}_outlet_h2o_frac"],
                    mole_arrs=(
                        self._history[f"{cid}_outlet_H2_molf"],
                        self._history[f"{cid}_outlet_O2_molf"],
                        self._history[f"{cid}_outlet_N2_molf"],
                        self._history[f"{cid}_outlet_H2O_molf"],
                        self._history[f"{cid}_outlet_CH4_molf"],
                        self._history[f"{cid}_outlet_CO2_molf"]
                    ),
                    extra_metric_arrs=metric_recorders
                )
                self._recorders.append(recorder)

    def _alloc_stream_history(self, cid: str, total_steps: int) -> None:
        """Allocate standard outlet stream history arrays."""
        self._history[f"{cid}_outlet_temp_c"] = np.zeros(total_steps, dtype=np.float64)
        self._history[f"{cid}_outlet_pressure_bar"] = np.zeros(total_steps, dtype=np.float64)
        self._history[f"{cid}_outlet_mass_flow_kg_h"] = np.zeros(total_steps, dtype=np.float64)
        self._history[f"{cid}_outlet_h2o_frac"] = np.zeros(total_steps, dtype=np.float64)
        for sp in ['H2', 'O2', 'N2', 'H2O', 'CH4', 'CO2']:
            self._history[f"{cid}_outlet_{sp}_molf"] = np.zeros(total_steps, dtype=np.float64)

    def decide_and_apply(self, t: float, prices: np.ndarray, wind: np.ndarray) -> None:
        """
        Make dispatch decision and apply setpoints.
        """
        dt = self._context.simulation.timestep_hours
        step_idx = self._state.step_idx

        if step_idx >= self._total_steps:
            return

        minute = int(round(t * 60))
        P_offer = wind[step_idx]
        current_price = prices[step_idx]
        P_future = wind[min(step_idx + 60, len(wind) - 1)]

        soec_kwh_kg = getattr(self._context.physics.soec_cluster, 'kwh_per_kg', 37.5)
        pem_kwh_kg = getattr(self._context.physics.pem_system, 'kwh_per_kg', 50.0)

        d_input = DispatchInput(
            minute=minute,
            P_offer=P_offer,
            P_future_offer=P_future,
            current_price=current_price,
            soec_capacity_mw=self._soec_capacity,
            pem_max_power_mw=self._pem_max,
            soec_h2_kwh_kg=soec_kwh_kg,
            pem_h2_kwh_kg=pem_kwh_kg,
            ppa_price_eur_mwh=getattr(self._context.economics, 'ppa_price_eur_mwh', 50.0),
            h2_price_eur_kg=getattr(self._context.economics, 'h2_price_eur_kg', 9.6),
            arbitrage_threshold_eur_mwh=getattr(self._context.economics, 'arbitrage_threshold_eur_mwh', None)
        )

        d_state = DispatchState(
            P_soec_prev=self._state.P_soec_prev,
            force_sell=self._state.force_sell
        )

        result = self._inner_strategy.decide(d_input, d_state)
        self._state.force_sell = result.state_update.get('force_sell', False)

        if self._soec:
            self._soec.receive_input('power_in', result.P_soec, 'electricity')

        if self._pem:
            # NOTE: Water supply is handled by the topology (PEM_Water_Pump).
            # DO NOT add hardcoded water here - it causes double delivery!
            self._pem.set_power_input_mw(result.P_pem)

        self._history['minute'][step_idx] = minute
        self._history['P_offer'][step_idx] = P_offer
        self._history['spot_price'][step_idx] = current_price
        self._state.step_idx = step_idx

    def record_post_step(self) -> None:
        """
        Record component outputs.
        OPTIMIZED: Uses pre-bound recorders to avoid dict lookups.
        """
        step_idx = self._state.step_idx
        if step_idx >= self._total_steps:
            return

        # 1. Specialized Recording (SOEC/PEM Main metrics)
        # (These are kept specific because they involve logic, not just stream dumping)
        
        # SOEC Logic
        P_soec_actual = 0.0
        h2_soec = 0.0
        steam_soec = 0.0
        if self._soec:
            if hasattr(self._soec, 'real_powers'):
                P_soec_actual = float(np.sum(self._soec.real_powers))
            if hasattr(self._soec, 'last_step_h2_kg'): h2_soec = self._soec.last_step_h2_kg
            elif hasattr(self._soec, 'last_h2_output_kg'): h2_soec = self._soec.last_h2_output_kg
            elif hasattr(self._soec, 'h2_output_kg'): h2_soec = self._soec.h2_output_kg
            steam_soec = getattr(self._soec, 'last_step_steam_input_kg', 0.0)
            
            # Record unreacted steam out
            self._history['H2O_soec_out_kg'][step_idx] = getattr(self._soec, 'last_water_output_kg', 0.0)
            
            # Manually handle SOEC stream recording (dynamic stream)
            cid = self._soec.component_id if hasattr(self._soec, 'component_id') else 'SOEC_Cluster'
            if cid:
                try:
                    out_stream = self._soec.get_output('h2_out')
                    if out_stream and out_stream.mass_flow_kg_h > 1e-6:
                         # Using standard access here as SOEC stream is special/rarely zero
                         self._history[f"{cid}_outlet_mass_flow_kg_h"][step_idx] = out_stream.mass_flow_kg_h
                         self._history[f"{cid}_outlet_temp_c"][step_idx] = out_stream.temperature_k - 273.15
                         self._history[f"{cid}_outlet_pressure_bar"][step_idx] = out_stream.pressure_pa / 1e5
                         self._history[f"{cid}_outlet_h2o_frac"][step_idx] = out_stream.composition.get('H2O', 0.0)
                except Exception:
                    pass

        self._state.P_soec_prev = P_soec_actual

        # PEM Logic
        h2_pem = 0.0
        P_pem_actual = 0.0
        if self._pem:
            h2_pem = getattr(self._pem, 'h2_output_kg', 0.0)
            if hasattr(self._pem, 'P_consumed_W'): P_pem_actual = self._pem.P_consumed_W / 1e6
            
            # Record impurity if available
            imp = getattr(self._pem, 'o2_impurity_ppm_mol', 0.0)
            self._history['PEM_o2_impurity_ppm_mol'][step_idx] = imp
            
            # Record additional PEM metrics
            self._history['H2O_pem_kg'][step_idx] = getattr(self._pem, 'water_consumption_kg', 0.0)
            self._history['O2_pem_kg'][step_idx] = getattr(self._pem, 'o2_output_kg', 0.0)
            self._history['pem_V_cell'][step_idx] = getattr(self._pem, 'V_cell', 0.0)

        # Global Power and Component Logic
        P_bop_kw = 0.0
        # Check compressors and tanks if not already cached
        if not hasattr(self, '_compressors'):
            self._compressors = [comp for _, comp in self._registry.list_components() if isinstance(comp, CompressorSingle)]
        if not hasattr(self, '_tanks'):
            # Handling for TankArray not explicitly imported at top level scope in some contexts
            from h2_plant.components.storage.h2_tank import TankArray
            self._tanks = [comp for _, comp in self._registry.list_components() if isinstance(comp, TankArray)]

        # Compressor Total Power
        total_comp_power = 0.0
        for comp in self._compressors:
            if hasattr(comp, 'power_kw'):
                total_comp_power += comp.power_kw
        self._history['compressor_power_kw'][step_idx] = total_comp_power

        # Tank Levels
        total_tank_mass = 0.0
        avg_tank_pressure = 0.0
        if self._tanks:
            for tank in self._tanks:
                total_tank_mass += tank.get_total_mass()
            # Use pressure of first tank array as representative of storage pressure
            if len(self._tanks[0].pressures) > 0:
                avg_tank_pressure = np.mean(self._tanks[0].pressures) / 1e5
        self._history['tank_level_kg'][step_idx] = total_tank_mass
        self._history['tank_pressure_bar'][step_idx] = avg_tank_pressure

        # BOP Calculation (iterate all components)
        for _, comp in self._registry.list_components():
             if hasattr(comp, 'power_kw'): P_bop_kw += comp.power_kw
             if hasattr(comp, 'electrical_power_kw'): P_bop_kw += comp.electrical_power_kw

        P_bop_mw = P_bop_kw / 1000.0
        P_total_consumed = P_soec_actual + P_pem_actual + P_bop_mw
        P_offer = self._history['P_offer'][step_idx]
        P_sold_corrected = max(0.0, P_offer - P_total_consumed)
        total_h2 = h2_soec + h2_pem
        self._state.cumulative_h2_kg += total_h2

        # Fast Array Writes
        self._history['P_soec_actual'][step_idx] = P_soec_actual
        self._history['P_pem'][step_idx] = P_pem_actual
        self._history['P_sold'][step_idx] = P_sold_corrected
        self._history['h2_kg'][step_idx] = total_h2
        self._history['H2_soec_kg'][step_idx] = h2_soec
        self._history['H2_pem_kg'][step_idx] = h2_pem
        self._history['cumulative_h2_kg'][step_idx] = self._state.cumulative_h2_kg
        self._history['steam_soec_kg'][step_idx] = steam_soec
        self._history['P_bop_mw'][step_idx] = P_bop_mw
        self._history['sell_decision'][step_idx] = 1 if P_sold_corrected > 0 else 0
        
        # SOEC Modules
        if self._soec and hasattr(self._soec, 'real_powers'):
            self._history['soec_active_modules'][step_idx] = int(np.sum(self._soec.real_powers > 0.01))
            # Only record if we need to
            if self._total_steps < 1000000: # Skip module detail for massive runs?
                for i, power_mw in enumerate(self._soec.real_powers):
                    key = f"soec_module_powers_{i+1}"
                    if key in self._history: self._history[key][step_idx] = power_mw

        # 2. Optimized Component Recording Loop
        # Iterate over pre-bound recorders (O(N) where N is component count, no string hashing)
        for rec in self._recorders:
            # 2a. Get Stream
            # Handle method calls vs attributes
            if rec.stream_attr == 'outlet' or rec.stream_attr == 'purified_gas_out': # Compressor/PSA method
                stream = rec.component.get_output(rec.stream_attr)
            else:
                stream = getattr(rec.component, rec.stream_attr, None)

            # 2b. Zero-Flow Optimization
            if stream is not None and stream.mass_flow_kg_h > 1e-6:
                # Direct Array Write
                rec.temp_arr[step_idx] = stream.temperature_k - 273.15
                rec.press_arr[step_idx] = stream.pressure_pa / 1e5
                rec.flow_arr[step_idx] = stream.mass_flow_kg_h
                
                h2o_val = stream.composition.get('H2O', 0.0) + stream.composition.get('H2O_liq', 0.0)
                rec.h2o_frac_arr[step_idx] = h2o_val

                # Direct Mole Fraction Write (using cached array)
                # Note: stream.get_composition_arrays() triggers the JIT cache if needed
                _, mole_fracs, _, _ = stream.get_composition_arrays()
                
                # Unrolled tuple unpacking
                rec.mole_arrs[0][step_idx] = mole_fracs[0] # H2
                rec.mole_arrs[1][step_idx] = mole_fracs[1] # O2
                rec.mole_arrs[2][step_idx] = mole_fracs[2] # N2
                rec.mole_arrs[3][step_idx] = mole_fracs[3] # H2O
                rec.mole_arrs[4][step_idx] = mole_fracs[4] # CH4
                rec.mole_arrs[5][step_idx] = mole_fracs[5] # CO2
            
            # 2c. Record Extra Metrics (Independent of flow)
            for attr_name, metric_arr in rec.extra_metric_arrs:
                # We assume these attributes exist because we checked the class type in init
                val = getattr(rec.component, attr_name, 0.0)
                metric_arr[step_idx] = val

        self._state.step_idx += 1

    def get_history(self) -> Dict[str, np.ndarray]:
        actual_steps = self._state.step_idx
        return {k: v[:actual_steps] for k, v in self._history.items()}

    def _find_soec(self, registry):
        for _, comp in registry.list_components():
            if hasattr(comp, 'soec_state') or comp.__class__.__name__ == 'SOECOperator': return comp
        return None

    def _find_pem(self, registry):
        for _, comp in registry.list_components():
            if hasattr(comp, 'V_cell') or comp.__class__.__name__ == 'DetailedPEMElectrolyzer': return comp
        return None

    def print_summary(self):
        if self._registry: components = {cid: comp for cid, comp in self._registry.list_components()}
        else: components = {}
        from h2_plant.reporting.stream_table import print_stream_summary_table
        print_stream_summary_table(components, list(components.keys()))

