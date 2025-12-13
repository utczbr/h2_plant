import yaml
import os
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Orchestrator")

class Orchestrator:
    """
    The Orchestrator manages the simulation lifecycle, configuration loading,
    and component coordination.
    
    Note: For new simulations, consider using run_integrated_simulation.py
    which uses the unified SimulationEngine + DispatchStrategy architecture.
    This class is maintained for backward compatibility with existing GUI code.
    """
    def __init__(self, scenarios_dir: str, context: Optional['SimulationContext'] = None):
        self.scenarios_dir = scenarios_dir
        self.context = context
        self.components: Dict[str, Any] = {}
        self.simulation_state: Dict[str, Any] = {}
        
        # Load configurations if context not provided
        if self.context is None:
            from h2_plant.config.loader import ConfigLoader
            self.loader = ConfigLoader(scenarios_dir)
            self.context = self.loader.load_context()
        
        # === B3: StateManager integration for checkpointing ===
        from pathlib import Path
        from h2_plant.simulation.state_manager import StateManager
        self._output_dir = Path(scenarios_dir) / "simulation_output"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self.state_manager = StateManager(output_dir=self._output_dir)
        self._checkpoint_interval_hours = 168  # Weekly checkpoints (configurable)
        self._last_checkpoint_hour = -1
        
        # Build Graph
        from h2_plant.core.graph_builder import PlantGraphBuilder
        self.builder = PlantGraphBuilder(self.context)
        self.components = self.builder.build()
        
        # Initialize Components
        self.initialize_components()

    def initialize_components(self):
        """Initializes all components in the graph."""
        logger.info("Initializing components...")
        dt = self.context.simulation.timestep_hours
        
        # Create proper ComponentRegistry
        from h2_plant.core.component_registry import ComponentRegistry
        self.registry = ComponentRegistry()
        
        # Register all components in the registry
        for cid, comp in self.components.items():
            self.registry.register(cid, comp)
        
        # Initialize all components via registry
        self.registry.initialize_all(dt)
            
        # Build Process Flow Map
        self._build_process_flow_map()
            
        logger.info(f"Initialized {len(self.components)} components.")

    def run_simulation(self, hours: Optional[int] = None):
        """
        Main simulation loop.
        """
        import numpy as np
        from h2_plant.control.dispatch import DispatchInput, DispatchState, ReferenceHybridStrategy
        from h2_plant.data.price_loader import EnergyPriceLoader
        
        if hours is None:
            hours = self.context.simulation.duration_hours
            
        logger.info(f"Starting simulation for {hours} hours...")
        
        # 1. Load Data
        data_loader = EnergyPriceLoader(self.scenarios_dir)
        prices, wind = data_loader.load_data(
            self.context.simulation.energy_price_file,
            self.context.simulation.wind_data_file,
            hours,
            self.context.simulation.timestep_hours
        )
            
        steps = len(prices)
        
        # 2. Initialize State
        self.simulation_state = {
            'P_soec_prev': 0.0,
            'force_sell': False,
            'history': {
                'minute': [], 'P_offer': [], 'P_soec_actual': [], 'P_pem': [], 'P_sold': [],
                'spot_price': [], 'h2_kg': []
            }
        }
        
        # 1. Resolve Components from Graph (Dynamic Lookup)
        soec = None
        pem = None
        
        for comp_id, comp in self.components.items():
            class_name = comp.__class__.__name__
            
            # Check for SOEC
            if hasattr(comp, 'soec_state') or class_name == 'SOECOperator':
                soec = comp
            # Check for PEM
            if hasattr(comp, 'V_cell') or class_name == 'DetailedPEMElectrolyzer':
                pem = comp

        # 2. Initialize Strategy based on Topology
        from h2_plant.control.dispatch import SoecOnlyStrategy
        
        if soec and not pem:
            logger.info("Topology detected: SOEC Only. Using SoecOnlyStrategy.")
            strategy = SoecOnlyStrategy()
        else:
            logger.info("Topology detected: Hybrid (or default). Using ReferenceHybridStrategy.")
            strategy = ReferenceHybridStrategy()
            
        dispatch_state = DispatchState()
                
        # Get Component Constraints from Context (Single Source of Truth)
        soec_capacity = 0.0
        if soec:
             # --- Calculate total capacity from Config ---
             spec = self.context.physics.soec_cluster
             num_modules = spec.num_modules
             soec_capacity = num_modules * spec.max_power_nominal_mw * spec.optimal_limit
            
        pem_max = 0.0
        if pem:
            pem_max = self.context.physics.pem_system.max_power_mw
        
        # 3. Loop
        print("\n--- Starting Hybrid Management Simulation (Debug Mode) ---")
        print(" | Min | H | P. Offer | P. SOEC | P. PEM | P. Sold | Spot | Dec | H2 SOEC (kg/min) | H2 PEM (kg/min) | H2O PEM (kg/min) |")
        print(" |-----|---|-----------|---------|--------|---------|------|-----|------------------|-----------------|------------------|")

        for step_idx in range(steps):
            # Time tracking
            t_hours = step_idx * self.context.simulation.timestep_hours
            minute = int(round(t_hours * 60))
            
            P_offer = wind[step_idx]
            current_price = prices[step_idx]
            
            # Future offer
            if step_idx + 60 < steps:
                P_future = wind[step_idx + 60]
            else:
                P_future = P_offer
                
            # Prepare Input
            d_input = DispatchInput(
                minute=minute,
                P_offer=P_offer,
                P_future_offer=P_future,
                current_price=current_price,
                soec_capacity_mw=soec_capacity,
                pem_max_power_mw=pem_max,
                soec_h2_kwh_kg=getattr(self.context.physics.soec_cluster, 'kwh_per_kg', 37.5),
                pem_h2_kwh_kg=getattr(self.context.physics.pem_system, 'kwh_per_kg', 50.0)
            )
            
            # Update State
            dispatch_state.P_soec_prev = self.simulation_state['P_soec_prev']
            dispatch_state.force_sell = self.simulation_state['force_sell']
            
            # Dispatch Decision
            result = strategy.decide(d_input, dispatch_state)
            
            # Update Simulation State
            self.simulation_state['force_sell'] = result.state_update['force_sell']
            
            # Execute Components
            
            # 1. Production Layer (Electrolyzers)
            # SOEC
            P_soec_actual = 0.0
            h2_soec = 0.0
            if soec:
                # Set power setpoint FIRST via receive_input (fixes power dispatch)
                soec.receive_input('power_in', result.P_soec, 'electricity')
                # Now step with TIME (not power!)
                P_soec_actual, h2_soec, steam_soec = soec.step(t_hours)
                self.simulation_state['P_soec_prev'] = P_soec_actual
                
                # --- PROCESS FLOW: SOEC Downstream ---
                try:
                    # Get H2 Output Stream
                    if hasattr(soec, 'get_output'):
                        h2_stream = soec.get_output('h2_out')
                        self._step_downstream(soec.component_id, 'h2_out', h2_stream, t_hours)
                except Exception as e:
                    logger.warning(f"SOEC Flow Error: {e}")

            else:
                self.simulation_state['P_soec_prev'] = 0.0
            
            # PEM
            h2_pem = 0.0
            P_pem_actual = result.P_pem
            if pem:
                pem.set_power_input_mw(result.P_pem)
                pem.step(t=t_hours)
                h2_pem = pem.h2_output_kg
                # Get actual consumption if available
                if hasattr(pem, 'P_consumed_W'):
                    P_pem_actual = pem.P_consumed_W / 1e6
                
                # --- PROCESS FLOW: PEM Downstream ---
                try:
                    if hasattr(pem, 'get_output'):
                        h2_stream = pem.get_output('h2_out')
                        self._step_downstream(pem.component_id, 'h2_out', h2_stream, t_hours)
                except Exception as e:
                    logger.warning(f"PEM Flow Error: {e}")
                    
                    
            # 1.5 Execute Active Balance of Plant Components (Compressors, Pumps)
            # Iterate through all components to step them and propagate output
            for comp_id, comp in self.components.items():
                if comp is soec or comp is pem:
                    continue

                try:
                    # 1. Step Component
                    if hasattr(comp, 'step'):
                        comp.step(t_hours)

                    # 2. Propagate Output (Push)
                    # We skip Tanks here because they are handled in the specific "BoP Sweep" below
                    is_tank = hasattr(comp, 'current_level_kg') or (hasattr(comp, 'get_inventory_kg') and hasattr(comp, 'withdraw_kg'))
                    
                    if not is_tank and hasattr(comp, 'get_output'):
                        # Try standard output ports
                        # Priority: outlet > h2_out > water_out > out
                        for port in ['outlet', 'h2_out', 'water_out', 'out']:
                            try:
                                stream = comp.get_output(port)
                                if stream and hasattr(stream, 'mass_flow_kg_h') and stream.mass_flow_kg_h > 0:
                                    self._step_downstream(comp_id, port, stream, t_hours)
                            except Exception:
                                continue 
                except Exception as e:
                    logger.warning(f"BoP Component {comp_id} Step Error: {e}")

            # 2. Balance of Plant Sweep (BoP) - Resolve Buffer Components
            # Iterate over all components. If it's a Buffer (Tank) with inventory, 
            # propagate its output to downstream components via PUSH model.
            for comp_id, comp in self.components.items():
                is_legacy = hasattr(comp, 'current_level_kg')
                is_unified = hasattr(comp, 'get_inventory_kg') and hasattr(comp, 'withdraw_kg')

                if is_legacy or is_unified:
                    try:
                        # Determine Inventory
                        available = 0.0
                        if is_unified:
                            available = comp.get_inventory_kg()
                        elif is_legacy:
                            available = comp.current_level_kg
                        
                        if available <= 0:
                            continue

                        # Determine Output Port
                        # Try 'outlet' first, then 'h2_out'
                        out_port = 'outlet'
                        ports = comp.get_ports() if hasattr(comp, 'get_ports') else {}
                        if 'outlet' not in ports and 'h2_out' in ports:
                            out_port = 'h2_out'
                            
                        # Determine Max Outflow
                        max_out = getattr(comp, 'max_outflow_kg_h', 1000.0)
                        
                        # Calculate Push Flow
                        # Simple rule: push available mass up to max_out rate
                        current_dt_hours = self.context.simulation.timestep_hours
                        flow_kg_h = min(available / current_dt_hours, max_out)
                        
                        if flow_kg_h > 0:
                            # 1. Propagate downstream FIRST (Push)
                            # We construct a push stream
                            from h2_plant.core.stream import Stream
                            
                            # Try to get pressure/temp from component if possible
                            press = getattr(comp, 'pressure_pa', 
                                          getattr(comp, 'pressure_bar', 1.0) * 1e5)
                            # If pressure returns bar, handle it? 
                            # TankArray uses pressure_pa, H2StorageTankEnhanced uses pressure_bar.
                            # Let's standardize on Pa for Stream
                            if hasattr(comp, 'pressure_bar'):
                                press = comp.pressure_bar * 1e5
                            
                            temp = getattr(comp, 'temperature_k', 298.15)
                            
                            push_stream = Stream(
                                mass_flow_kg_h=flow_kg_h,
                                temperature_k=temp,
                                pressure_pa=press,
                                composition={'H2': 1.0}
                            )
                            
                            # Propagate
                            self._step_downstream(comp_id, out_port, push_stream, t_hours)
                            
                            # 2. Withdraw Mass Logic
                            # Now that we've pushed, we deduct the mass.
                            amount_to_withdraw = flow_kg_h * current_dt_hours
                            
                            if is_unified:
                                withdrawn = comp.withdraw_kg(amount_to_withdraw)
                                # overflow/underflow handling is inside withdraw_kg
                            elif is_legacy:
                                # Legacy direct manipulation
                                comp.current_level_kg -= amount_to_withdraw
                                comp.current_level_kg = max(0.0, comp.current_level_kg)
                            
                    except Exception as e:
                        logger.warning(f"BoP Sweep Error {comp_id}: {e}")
                        pass

            total_h2_produced = h2_soec + h2_pem
            
            # Correct P_sold based on actual consumption to ensure mass/energy balance
            # (Coordinator prediction might slightly differ from actual component physics)
            
            # 1. Sum Electrolyzer Power (MW)
            P_electrolyzers_mw = P_soec_actual + P_pem_actual
            
            # 2. Sum BoP Power (kW -> MW)
            P_bop_kw = 0.0
            
            # Helper to check for power attributes in active components
            # (We iterate anyway for logging later, but we need the value now for P_sold)
            for comp_id, comp in self.components.items():
                # Compressors and Pumps
                if hasattr(comp, 'power_kw'):
                    P_bop_kw += comp.power_kw
                # Chillers
                if hasattr(comp, 'electrical_power_kw'):
                    P_bop_kw += comp.electrical_power_kw
            
            P_bop_mw = P_bop_kw / 1000.0
            
            # 3. Calculate Final Consumption and Sold
            P_total_consumed = P_electrolyzers_mw + P_bop_mw
            P_sold_corrected = max(0.0, P_offer - P_total_consumed)
            
            # Log
            self.simulation_state['history']['minute'].append(minute)
            self.simulation_state['history']['P_offer'].append(P_offer)
            self.simulation_state['history']['P_soec_actual'].append(P_soec_actual) # Renamed to match legacy
            self.simulation_state['history']['P_pem'].append(P_pem_actual)
            self.simulation_state['history']['P_sold'].append(P_sold_corrected)
            self.simulation_state['history']['spot_price'].append(current_price) # Renamed to match legacy
            self.simulation_state['history']['h2_kg'].append(total_h2_produced)
            
            # Detailed Production Logging
            self.simulation_state['history'].setdefault('H2_soec_kg', []).append(h2_soec)
            self.simulation_state['history'].setdefault('H2_pem_kg', []).append(h2_pem)
            
            # Initialize steam_soec if not defined (e.g. no SOEC)
            if 'steam_soec' not in locals():
                steam_soec = 0.0
                
            self.simulation_state['history'].setdefault('steam_soec_kg', []).append(steam_soec)
            
            # SOEC Water Output (Unreacted Steam)
            h2o_soec_out = getattr(soec, 'last_water_output_kg', 0.0) if soec else 0.0
            self.simulation_state['history'].setdefault('H2O_soec_out_kg', []).append(h2o_soec_out)
            
            # PEM Water/O2
            # Use component reported values if available, else estimate
            h2o_pem = 0.0
            if pem:
                h2o_pem = getattr(pem, 'water_consumption_kg', h2_pem * 9.0 * 1.02)
            else:
                h2o_pem = h2_pem * 9.0 * 1.02 # Fallback estimate
                
            o2_pem = h2_pem * 8.0
            
            self.simulation_state['history'].setdefault('H2O_pem_kg', []).append(h2o_pem)
            self.simulation_state['history'].setdefault('O2_pem_kg', []).append(o2_pem)
            
            # Advanced State Logging (for specific charts)
            # PEM Voltage
            pem_v_cell = 0.0
            if pem and hasattr(pem, 'V_cell'):
                pem_v_cell = pem.V_cell
            self.simulation_state['history'].setdefault('pem_V_cell', []).append(pem_v_cell)
            
            # SOEC Active Modules & Detailed State
            soec_active = 0
            soec_powers = []
            soec_states = []
            if soec:
                if hasattr(soec, 'real_powers'):
                     import numpy as np
                     soec_active = int(np.sum(soec.real_powers > 0.01))
                     soec_powers = soec.real_powers.tolist()
                     soec_states = soec.real_states.tolist()
            
            self.simulation_state['history'].setdefault('soec_active_modules', []).append(soec_active)
            self.simulation_state['history'].setdefault('soec_module_powers', []).append(soec_powers)
            self.simulation_state['history'].setdefault('soec_module_states', []).append(soec_states)
            
            # Cumulative H2
            prev_cum = self.simulation_state['history'].get('cumulative_h2_kg', [0.0])[-1] if 'cumulative_h2_kg' in self.simulation_state['history'] else 0.0
            self.simulation_state['history'].setdefault('cumulative_h2_kg', []).append(prev_cum + total_h2_produced)

            # Log BoP (Dynamic Scanning)
            # Iterate through all components to find Tanks and Compressors
            for comp_id, comp in self.components.items():
                # We can check type or attributes
                # Tanks
                if hasattr(comp, 'current_level_kg'):
                    self.simulation_state['history'].setdefault(f'{comp_id}_level_kg', []).append(comp.current_level_kg)
                    self.simulation_state['history'].setdefault(f'{comp_id}_pressure_bar', []).append(comp.pressure_bar)
                
                # Compressors and Pumps (power_kw)
                if hasattr(comp, 'power_kw'):
                    self.simulation_state['history'].setdefault(f'{comp_id}_power_kw', []).append(comp.power_kw)
                
                # Chillers (electrical_power_kw)
                if hasattr(comp, 'electrical_power_kw'):
                    self.simulation_state['history'].setdefault(f'{comp_id}_power_kw', []).append(comp.electrical_power_kw)
            
            # Legacy keys for backward compatibility (map to specific known IDs if they exist)
            # Assuming "H2_Tank" and "H2_Compressor" are the main ones for now
            tank_main = self.components.get("H2_Tank")
            comp_main = self.components.get("H2_Compressor")
            
            self.simulation_state['history'].setdefault('tank_level_kg', []).append(tank_main.current_level_kg if tank_main else 0.0)
            self.simulation_state['history'].setdefault('tank_pressure_bar', []).append(tank_main.pressure_bar if tank_main else 0.0)
            self.simulation_state['history'].setdefault('compressor_power_kw', []).append(comp_main.power_kw if comp_main else 0.0)
            
            # Derived fields for legacy report
            self.simulation_state['history'].setdefault('sell_decision', []).append(1 if P_sold_corrected > 0 else 0)
            
            # DEBUG PRINT (Every 15 minutes)
            if minute % 15 == 0:
                sell_dec_str = 'SELL' if (P_sold_corrected > 0) else 'H2'
                # Convert kg/h to kg/min for display parity with reference
                h2_soec_min = h2_soec / 60.0
                h2_pem_min = h2_pem / 60.0
                h2o_pem_min = h2o_pem / 60.0
                
                print(
                    f" | {minute:03d} | {minute//60 + 1} | {P_offer:9.2f} | {P_soec_actual:7.2f} | {P_pem_actual:6.2f} | {P_sold_corrected:7.2f} | {current_price:5.0f} | {sell_dec_str:3s} | {h2_soec_min:16.4f} | {h2_pem_min:15.4f} | {h2o_pem_min:16.4f} |"
                )
            
            # === B3: Automatic checkpointing ===
            current_hour = int(t_hours)
            if (current_hour > 0 and 
                current_hour % self._checkpoint_interval_hours == 0 and 
                current_hour != self._last_checkpoint_hour):
                self._save_checkpoint(current_hour, step_idx)
                self._last_checkpoint_hour = current_hour
            
        # DEBUG SUMMARY
        print("\n## Simulation Summary (Total/Average Values)")
        import numpy as np
        hist = self.simulation_state['history']
        
        # Helper to sum and scale (assuming 1 min steps for now, or use dt)
        # Reference uses / 60 for MWh if power is MW and step is min?
        # Here we have dt_hours. Energy MWh = Power MW * dt_hours
        dt = self.context.simulation.timestep_hours
        
        E_total_offer = np.sum(hist['P_offer']) * dt
        E_soec = np.sum(hist['P_soec_actual']) * dt
        E_pem = np.sum(hist['P_pem']) * dt
        E_sold = np.sum(hist['P_sold']) * dt
        
        H2_soec_total = np.sum(hist.get('H2_soec_kg', []))
        H2_pem_total = np.sum(hist.get('H2_pem_kg', []))
        H2_total = H2_soec_total + H2_pem_total
        
        # Calculate Compressor Energy
        E_compressor_total_mwh = 0.0
        # Sum all keys ending in _power_kw
        for key, val in hist.items():
            if key.endswith('_power_kw'):
                # val is list of kW. Convert to MW then * dt
                # sum(val) (kW) * dt (h) = kWh. / 1000 = MWh
                E_comp_mwh = np.sum(val) * dt / 1000.0
                E_compressor_total_mwh += E_comp_mwh

        print(f"* Total Offered Energy: {E_total_offer:.2f} MWh")
        print(f"* Energy Supplied to SOEC: {E_soec:.2f} MWh")
        print(f"* Energy Supplied to PEM: {E_pem:.2f} MWh")
        print(f"* BoP Energy Consumption (Compressors, Pumps, Chillers): {E_compressor_total_mwh:.2f} MWh")
        print(f"* **Total System Hydrogen Production**: {H2_total:.2f} kg")
        print(f"  * SOEC Production: {H2_soec_total:.2f} kg")
        print(f"  * PEM Production: {H2_pem_total:.2f} kg")
        print(f"* Energy Sold to the Market: {E_sold:.2f} MWh")
        print("-------------------------------------------------------------------")

        logger.info("Simulation completed.")
        return self.simulation_state['history']

    def _map_topology_ids(self) -> Dict[str, str]:
        """Maps Topology UUIDs to Registered Component IDs."""
        mapping = {}
        from h2_plant.core.component_ids import ComponentID
        
        # Counters for array components
        counters = {
            "Chiller": 0,
            "Compressor": 0,
            "Tank": 0,
            "SteamGenerator": 0,
            "HeatExchanger": 0
        }
        
        # Iterate topology nodes
        if not self.context or not self.context.topology:
            return {}
            
        for node in self.context.topology.nodes:
            uuid = node.id
            b_type = node.type # "PEM", "SOEC", "Compressor", etc.
            
            target_id = None
            
            # Map based on type and counters
            if b_type == "PEM":
                target_id = ComponentID.PEM_ELECTROLYZER_DETAILED.value
            elif b_type == "SOEC":
                target_id = ComponentID.SOEC_CLUSTER.value
            elif b_type == "Chiller":
                target_id = f"chiller_{counters['Chiller']}"
                counters['Chiller'] += 1
            elif b_type == "SteamGenerator":
                target_id = f"steam_generator_{counters['SteamGenerator']}"
                counters['SteamGenerator'] += 1
            elif b_type == "Compressor":
                # Ambiguous: Filling vs Outgoing. 
                # PlantBuilder registers FILLING_COMPRESSOR and OUTGOING_COMPRESSOR.
                # Assuming order: Filling then Outgoing? Or check properties?
                # For now, map to generic if exists, or skip.
                # Let's check if we can distinguish by params
                pass
            
            if target_id:
                mapping[uuid] = target_id
                
        return mapping

    def _save_checkpoint(self, hour: int, step_idx: int, emergency: bool = False) -> None:
        """
        Save simulation checkpoint (B3: State Management).
        
        Args:
            hour: Current simulation hour
            step_idx: Current step index in the simulation
            emergency: True if this is an emergency checkpoint (e.g., crash)
        """
        checkpoint_type = "emergency" if emergency else "regular"
        logger.info(f"Saving {checkpoint_type} checkpoint at hour {hour}")
        
        try:
            component_states = self.registry.get_all_states()
            
            checkpoint_path = self.state_manager.save_checkpoint(
                hour=hour,
                component_states=component_states,
                metadata={
                    'step_idx': step_idx,
                    'checkpoint_type': checkpoint_type,
                    'simulation_state': {
                        'P_soec_prev': self.simulation_state.get('P_soec_prev', 0.0),
                        'force_sell': self.simulation_state.get('force_sell', False)
                    }
                }
            )
            logger.debug(f"Checkpoint saved to: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}", exc_info=True)

    def _build_process_flow_map(self):
        """Builds the process flow map from topology."""
        self.process_flow_map = {} # source_id -> {source_port -> [(target_id, target_port)]}
        
        if not self.context or not self.context.topology:
            return

        # Use UUIDs directly as Component IDs are now set to UUIDs
        for node in self.context.topology.nodes:
            source_id = node.id
            
            if source_id not in self.process_flow_map:
                self.process_flow_map[source_id] = {}
                
            for conn in node.connections:
                target_id = conn.target_name # This is the target UUID defined in GraphToConfigAdapter
                source_port = conn.source_port
                target_port = conn.target_port
                
                if source_port not in self.process_flow_map[source_id]:
                    self.process_flow_map[source_id][source_port] = []
                    
                self.process_flow_map[source_id][source_port].append((target_id, target_port))
                
        logger.info(f"Built process flow map with {len(self.process_flow_map)} sources.")

    def _step_downstream(self, source_id: str, source_port: str, value: Any, t: float, visited: set = None):
        """
        Propagate flow data to downstream components.
        
        CRITICAL FIXES (Phase A):
        - A1: REMOVED step(t) calls - Engine handles stepping, not flow propagation
        - A2: ADDED extract_output() calls to enforce mass conservation
        - A4: Wrap values in Stream objects for thermodynamic data preservation
        
        This method only transfers data between components. The simulation loop
        (SimulationEngine or run_simulation) is responsible for calling step()
        exactly once per component per timestep.
        """
        from h2_plant.core.stream import Stream
        
        if visited is None: 
            visited = set()
        
        # Avoid cycles
        state_key = (source_id, source_port)
        if state_key in visited: 
            return
        visited.add(state_key)
        
        # Find connections
        targets = self.process_flow_map.get(source_id, {}).get(source_port, [])
        
        if not targets:
            return
        
        # === A4: Ensure value is a Stream object ===
        if not isinstance(value, Stream):
            if isinstance(value, (int, float)):
                value = Stream(
                    mass_flow_kg_h=float(value),
                    temperature_k=298.15,
                    pressure_pa=101325.0,
                    composition={'H2': 1.0}
                )
            elif value is None:
                return  # No data to transfer
        
        for target_id, target_port in targets:
            comp = self.components.get(target_id)
            if not comp: 
                continue
            
            try:
                # Infer resource type from Stream composition
                resource_type = self._infer_resource_type(value)
                
                # Transfer input to downstream component
                accepted = comp.receive_input(target_port, value, resource_type=resource_type)
                
                # === A1 FIX: DO NOT call step(t) here! ===
                # The simulation loop handles stepping. Flow propagation only transfers data.
                # REMOVED: comp.step(t)
                
                # Propagate outputs from this component (if any)
                # Note: This is for intermediate buffering only, not physics execution
                ports = comp.get_ports() if hasattr(comp, 'get_ports') else {}
                for p_name, p_meta in ports.items():
                    if p_meta.get('type') == 'output':
                        try:
                            out_val = comp.get_output(p_name)
                            if out_val is not None:
                                # === A2 FIX: Extract output to enforce mass conservation ===
                                if hasattr(out_val, 'mass_flow_kg_h'):
                                    dt = self.context.simulation.timestep_hours
                                    mass_kg = out_val.mass_flow_kg_h * dt
                                    out_resource = p_meta.get('resource_type', 'hydrogen')
                                    
                                    # Deduct mass from source component
                                    if hasattr(comp, 'extract_output'):
                                        try:
                                            comp.extract_output(p_name, mass_kg, out_resource)
                                        except (NotImplementedError, ValueError):
                                            pass  # Component doesn't support extraction
                                
                                self._step_downstream(target_id, p_name, out_val, t, visited)
                        except NotImplementedError:
                            pass  # Skip if not implemented
                        except Exception:
                            pass  # Skip on error
                            
            except Exception as e:
                logger.warning(f"Flow propagation error {source_id} -> {target_id}: {e}")
    
    def _infer_resource_type(self, stream) -> str:
        """Infer resource type from Stream composition."""
        if not hasattr(stream, 'composition') or not stream.composition:
            return 'unknown'
        
        # Return dominant species
        try:
            dominant = max(stream.composition.items(), key=lambda x: x[1])
            species_map = {'H2': 'hydrogen', 'O2': 'oxygen', 'H2O': 'water', 'CO2': 'carbon_dioxide'}
            return species_map.get(dominant[0], dominant[0].lower())
        except (ValueError, TypeError):
            return 'unknown'

