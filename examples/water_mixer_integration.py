"""
Example configuration and integration for WaterMixer component.

This script demonstrates how to use the WaterMixer component in:
1. Standalone mode
2. Plant configuration with YAML
3. Process flow network integration
"""

import numpy as np
from pathlib import Path

from h2_plant.components.mixing import WaterMixer
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
from h2_plant.simulation.engine import SimulationEngine


def example_1_standalone_usage():
    """
    Example 1: Standalone WaterMixer usage
    
    Demonstrates basic usage without plant configuration.
    """
    print("=" * 70)
    print("EXAMPLE 1: Standalone WaterMixer Usage")
    print("=" * 70)
    print()
    
    # Create registry
    registry = ComponentRegistry()
    
    # Create mixer
    mixer = WaterMixer(
        outlet_pressure_kpa=200.0,
        fluid_type='Water',
        max_inlet_streams=5
    )
    
    # Register mixer
    registry.register('water_mixer_main', mixer)
    
    # Initialize
    registry.initialize_all(dt=1.0)
    
    # Create input streams
    streams = [
        Stream(
            mass_flow_kg_h=1800.0,  # 0.5 kg/s
            temperature_k=288.15,    # 15°C
            pressure_pa=200000.0,    # 200 kPa
            composition={'H2O': 1.0},
            phase='liquid'
        ),
        Stream(
            mass_flow_kg_h=1080.0,  # 0.3 kg/s
            temperature_k=353.15,    # 80°C
            pressure_pa=220000.0,    # 220 kPa
            composition={'H2O': 1.0},
            phase='liquid'
        ),
        Stream(
            mass_flow_kg_h=720.0,   # 0.2 kg/s
            temperature_k=323.15,    # 50°C
            pressure_pa=210000.0,    # 210 kPa
            composition={'H2O': 1.0},
            phase='liquid'
        )
    ]
    
    # Feed streams to mixer
    for i, stream in enumerate(streams):
        mixer.receive_input(f'inlet_{i}', stream, 'water')
    
    # Execute mixing
    mixer.step(t=0.0)
    
    # Get output
    output = mixer.get_output('outlet')
    
    # Display results
    print(f"Input Streams:")
    for i, s in enumerate(streams):
        print(f"  Stream {i+1}: {s.mass_flow_kg_h/3600:.3f} kg/s @ {s.temperature_k-273.15:.2f}°C")
    
    print(f"\nOutput Stream:")
    print(f"  Mass Flow: {output.mass_flow_kg_h/3600:.3f} kg/s ({output.mass_flow_kg_h:.1f} kg/h)")
    print(f"  Temperature: {output.temperature_k-273.15:.2f}°C ({output.temperature_k:.2f} K)")
    print(f"  Pressure: {output.pressure_pa/1000:.1f} kPa")
    
    # Get state
    state = mixer.get_state()
    print(f"\nMixer State:")
    print(f"  Active Inlets: {state['num_active_inlets']}/{state['max_inlets']}")
    print(f"  Outlet Enthalpy: {state['outlet_enthalpy_kj_kg']:.2f} kJ/kg")
    
    print()


def example_2_yaml_configuration():
    """
    Example 2: Using WaterMixer with YAML configuration
    
    Shows configuration structure for plant config files.
    """
    print("=" * 70)
    print("EXAMPLE 2: YAML Configuration Structure")
    print("=" * 70)
    print()
    
    yaml_config = """
# water_system_config.yaml
# Example configuration for water system with mixer

name: "Water System with Mixer"
version: "1.0"
description: "Water treatment and mixing system"

water_treatment:
  quality_test:
    enabled: true
    sample_interval_hours: 1.0
  
  treatment_block:
    enabled: true
    max_flow_m3h: 10.0
    power_consumption_kw: 20.0
  
  ultrapure_storage:
    capacity_l: 5000.0
    initial_fill_ratio: 0.5
  
  pumps:
    pump_a:
      enabled: true
      power_kw: 0.75
      power_source: "grid_or_battery"
      outlet_pressure_bar: 5.0
    
    pump_b:
      enabled: true
      power_kw: 1.5
      power_source: "grid"
      outlet_pressure_bar: 8.0
  
  # New: Water mixer configuration
  mixer:
    enabled: true
    outlet_pressure_kpa: 200.0
    fluid_type: "Water"
    max_inlet_streams: 10
    # Optional: specify inlet source IDs for automatic connection
    inlet_sources:
      - "pump_a"
      - "pump_b"
      - "treatment_block"

simulation:
  timestep_hours: 1.0
  duration_hours: 24
"""
    
    print("YAML Configuration Example:")
    print(yaml_config)
    print()


def example_3_process_flow_network():
    """
    Example 3: WaterMixer in Process Flow Network
    
    Demonstrates integration with other components in a flow network.
    """
    print("=" * 70)
    print("EXAMPLE 3: Process Flow Network Integration")
    print("=" * 70)
    print()
    
    # Create registry
    registry = ComponentRegistry()
    
    # Create water sources (simplified as constant streams for this example)
    class WaterSource:
        """Simple water source for demonstration."""
        def __init__(self, source_id, flow_kg_h, temp_k, pressure_pa):
            self.component_id = source_id
            self.flow_kg_h = flow_kg_h
            self.temp_k = temp_k
            self.pressure_pa = pressure_pa
            self._initialized = False
            self.dt = 1.0
            self._registry = None
            
        def set_component_id(self, cid):
            self.component_id = cid
            
        def initialize(self, dt, registry):
            self.dt = dt
            self._registry = registry
            self._initialized = True
            
        def step(self, t):
            pass
            
        def get_output(self, port_name):
            return Stream(
                mass_flow_kg_h=self.flow_kg_h,
                temperature_k=self.temp_k,
                pressure_pa=self.pressure_pa,
                composition={'H2O': 1.0},
                phase='liquid'
            )
        
        def get_state(self):
            return {
                'component_id': self.component_id,
                'flow_kg_h': self.flow_kg_h,
                'temp_k': self.temp_k
            }
    
    # Create sources
    cold_source = WaterSource('cold_water', 1800.0, 288.15, 200000.0)
    hot_source = WaterSource('hot_water', 1080.0, 353.15, 220000.0)
    
    # Create mixer
    mixer = WaterMixer(outlet_pressure_kpa=200.0)
    
    # Register all components
    registry.register('cold_water_source', cold_source)
    registry.register('hot_water_source', hot_source)
    registry.register('water_mixer', mixer)
    
    # Initialize
    registry.initialize_all(dt=1.0)
    
    # Simulation loop
    print("Running 5-hour simulation...")
    print()
    
    for hour in range(5):
        print(f"Hour {hour}:")
        
        # Get streams from sources
        cold_stream = cold_source.get_output('outlet')
        hot_stream = hot_source.get_output('outlet')
        
        # Feed to mixer
        mixer.receive_input('cold_inlet', cold_stream, 'water')
        mixer.receive_input('hot_inlet', hot_stream, 'water')
        
        # Step components
        cold_source.step(t=float(hour))
        hot_source.step(t=float(hour))
        mixer.step(t=float(hour))
        
        # Get mixed output
        output = mixer.get_output('outlet')
        
        if output:
            print(f"  Mixed output: {output.mass_flow_kg_h/3600:.3f} kg/s @ {output.temperature_k-273.15:.2f}°C")
        else:
            print(f"  No output")
        
        print()
    
    print("Simulation complete!")
    print()


def example_4_multi_source_mixing():
    """
    Example 4: Multi-source water mixing scenario
    
    Real-world scenario with multiple water sources at different temperatures.
    """
    print("=" * 70)
    print("EXAMPLE 4: Multi-Source Mixing Scenario")
    print("=" * 70)
    print()
    
    print("Scenario: Mixing water from different process stages")
    print("  - Cooling water return: 45°C, 1000 kg/h")
    print("  - Process outlet: 60°C, 800 kg/h")
    print("  - Fresh makeup: 20°C, 500 kg/h")
    print("  - Target outlet pressure: 250 kPa")
    print()
    
    # Create mixer
    mixer = WaterMixer(
        outlet_pressure_kpa=250.0,
        max_inlet_streams=10
    )
    
    registry = ComponentRegistry()
    registry.register('process_mixer', mixer)
    registry.initialize_all(dt=1.0)
    
    # Define sources
    sources = {
        'cooling_return': Stream(1000.0, 318.15, 250000.0, {'H2O': 1.0}, 'liquid'),
        'process_outlet': Stream(800.0, 333.15, 260000.0, {'H2O': 1.0}, 'liquid'),
        'fresh_makeup': Stream(500.0, 293.15, 300000.0, {'H2O': 1.0}, 'liquid'),
    }
    
    # Feed streams
    for name, stream in sources.items():
        mixer.receive_input(name, stream, 'water')
    
    # Mix
    mixer.step(t=0.0)
    
    # Results
    output = mixer.get_output('outlet')
    state = mixer.get_state()
    
    print("Results:")
    print(f"  Total flow: {output.mass_flow_kg_h:.1f} kg/h ({output.mass_flow_kg_h/3600:.3f} kg/s)")
    print(f"  Mixed temperature: {output.temperature_k-273.15:.2f}°C")
    print(f"  Output pressure: {output.pressure_pa/1000:.1f} kPa")
    print(f"  Specific enthalpy: {state['outlet_enthalpy_kj_kg']:.2f} kJ/kg")
    print()
    
    # Energy check
    total_mass_in = sum(s.mass_flow_kg_h for s in sources.values())
    print(f"Mass balance check:")
    print(f"  Input total: {total_mass_in:.1f} kg/h")
    print(f"  Output: {output.mass_flow_kg_h:.1f} kg/h")
    print(f"  Difference: {abs(total_mass_in - output.mass_flow_kg_h):.6f} kg/h")
    print()


if __name__ == "__main__":
    # Run all examples
    example_1_standalone_usage()
    example_2_yaml_configuration()
    example_3_process_flow_network()
    example_4_multi_source_mixing()
    
    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
