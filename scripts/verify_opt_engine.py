
import sys
import os
from pathlib import Path
import logging
import numpy as np
from typing import Dict, Any

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from h2_plant.simulation.engine import SimulationEngine
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.config.plant_config import SimulationConfig
from h2_plant.control.engine_dispatch import HybridArbitrageEngineStrategy, StreamRecorder
from h2_plant.core.component import Component

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_opt")

class DummyComponent(Component):
    def __init__(self, component_id):
        # Pass component_id to super().__init__ if supported, or set it manually
        super().__init__(component_id=component_id)
        self.power_kw = 100.0
        self.electrical_power_kw = 100.0
        self.outlet_temp_c = 25.0
        self.outlet_pressure_bar = 1.0
        self.outlet_mass_flow_kg_h = 10.0
        
    def initialize(self, dt: float, registry: 'ComponentRegistry') -> None:
        super().initialize(dt, registry)
        
    def step(self, t: float) -> None:
        super().step(t)
        
    def get_state(self) -> Dict[str, Any]:
        return super().get_state()

    def get_output(self, port):
        # Mock stream for summary table
        from h2_plant.core.stream import Stream
        return Stream(
            mass_flow_kg_h=self.outlet_mass_flow_kg_h,
            temperature_k=self.outlet_temp_c + 273.15,
            pressure_pa=self.outlet_pressure_bar * 1e5,
            composition={'H2': 1.0},
            phase='gas'
        )

def main():
    print("Verifying Engine Optimization...")
    
    # 1. Setup minimal registry
    registry = ComponentRegistry()
    
    # Register components
    comp1 = DummyComponent("TestComp1")
    comp2 = DummyComponent("TestComp2")
    
    registry.register("TestComp1", comp1)
    registry.register("TestComp2", comp2)
    
    # 2. Setup config
    config = SimulationConfig(
        timestep_hours=1.0/60.0,
        duration_hours=1,
        start_hour=0,
        checkpoint_interval_hours=0
    )
    
    # 3. Setup Dispatch Strategy
    dispatch = HybridArbitrageEngineStrategy()
    
    # 4. Initialize Engine
    engine = SimulationEngine(
        registry=registry,
        config=config,
        dispatch_strategy=dispatch
    )
    
    print("Initializing Engine...")
    # Mock context for dispatch initialization
    class MockContext:
        class Physics:
            class SOEC:
                num_modules = 1
                max_power_nominal_mw = 1.0
                optimal_limit = 1.0
                kwh_per_kg = 40.0
            class PEM:
                max_power_mw = 1.0
                kwh_per_kg = 50.0
            soec_cluster = SOEC()
            pem_system = PEM()
        class Economics:
            ppa_price_eur_mwh = 50.0
            h2_price_eur_kg = 5.0
            arbitrage_threshold_eur_mwh = 10.0
        class Simulation:
            timestep_hours = 1.0/60.0
        
        physics = Physics()
        economics = Economics()
        simulation = Simulation()
    
    # Initialize engine (which initializes components)
    engine.initialize()
    
    # Manually initialize dispatch
    total_steps = 60
    dispatch.initialize(registry, MockContext(), total_steps)
    
    # 5. Verify Optimizations
    
    # Check Pre-resolved Execution List
    if hasattr(engine, '_execution_list') and len(engine._execution_list) > 0:
        print(f"✅ Engine has pre-resolved execution list with {len(engine._execution_list)} items.")
    else:
        print("❌ Engine missing _execution_list!")
        sys.exit(1)
        
    # Check Dispatch Recorders
    if hasattr(dispatch, '_recorders'):
        # Just verify the list exists efficiently
        print(f"✅ Dispatch strategy initialized recorders (Count: {len(dispatch._recorders)}).")
    else:
        print("❌ Dispatch strategy missing _recorders!")
        sys.exit(1)

    print("✅ Optimization Verification Passed!")
    
    print("\n--- Testing Summary Output ---")
    try:
        dispatch.print_summary()
        print("✅ print_summary executed successfully")
    except Exception as e:
        print(f"❌ print_summary failed: {e}")
        sys.exit(1)

"""
    def get_output(self, port):
        # Mock stream for summary table
        from h2_plant.core.stream import Stream
        return Stream(
            mass_flow_kg_h=self.outlet_mass_flow_kg_h,
            temperature_k=self.outlet_temp_c + 273.15,
            pressure_pa=self.outlet_pressure_bar * 1e5,
            composition={'H2': 1.0},
            phase='gas'
        )
"""

if __name__ == "__main__":
    main()
