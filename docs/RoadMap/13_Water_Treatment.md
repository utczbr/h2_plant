## System Components

### 1. Water Quality Test Block

This component monitors incoming water conditions before treatment, recording purity, temperature, and other parameters.[1]

```python
class WaterQualityTestBlock(Component):
    """Records water quality parameters from incoming feed."""
    
    def __init__(self, sample_interval_hours: float = 1.0):
        super().__init__()
        self.sample_interval_hours = sample_interval_hours
        self.inlet_flow_m3h = 0.0
        self.inlet_temp_c = 20.0
        self.inlet_pressure_bar = 0.5
        self.purity_ppm = 0.0
        
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        if self.sample_interval_hours <= 0:
            raise ValueError("Sample interval must be positive")
    
    def step(self, t: float) -> None:
        super().step(t)
        # Read inlet conditions from external network
        # Update quality metrics
        
    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            "inlet_flow_m3h": float(self.inlet_flow_m3h),
            "inlet_temp_c": float(self.inlet_temp_c),
            "inlet_pressure_bar": float(self.inlet_pressure_bar),
            "purity_ppm": float(self.purity_ppm),
            "flows": {
                "inputs": {
                    "raw_water": {
                        "value": self.inlet_flow_m3h,
                        "unit": "m3/h",
                        "source": "external_network",
                        "flowtype": "WATER_MASS"
                    },
                    "electricity": {
                        "value": 0.0,
                        "unit": "kW",
                        "source": "grid_or_battery",
                        "flowtype": "ELECTRICAL_ENERGY"
                    }
                },
                "outputs": {
                    "tested_water": {
                        "value": self.inlet_flow_m3h,
                        "unit": "m3/h",
                        "destination": "water_treatment",
                        "flowtype": "WATER_MASS"
                    }
                }
            }
        }
```

### 2. Water Treatment Block

The main treatment unit that purifies water to ultrapure standards, consuming electricity and outputting high-purity water.[2][1]

```python
class WaterTreatmentBlock(Component):
    """Water treatment system producing ultrapure water."""
    
    def __init__(self, max_flow_m3h: float, power_consumption_kw: float):
        super().__init__()
        self.max_flow_m3h = max_flow_m3h
        self.power_consumption_kw = power_consumption_kw
        self.output_flow_kgh = 0.0
        self.output_temp_c = 20.0
        self.output_pressure_bar = 1.0
        self.test_flow_kgh = 0.0
        self.lut: LUTManager = None
        
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        if registry.has("lutmanager"):
            self.lut = registry.get("lutmanager")
        
    def step(self, t: float) -> None:
        super().step(t)
        # Receive tested water from quality block
        # Process to ultrapure standards
        # Approximately 10000 kg/h at 20°C, 1 bar
        self.output_flow_kgh = 10000.0
        self.test_flow_kgh = self.output_flow_kgh * 0.01  # 1% for testing
        
    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            "output_flow_kgh": float(self.output_flow_kgh),
            "output_temp_c": float(self.output_temp_c),
            "output_pressure_bar": float(self.output_pressure_bar),
            "power_consumption_kw": float(self.power_consumption_kw),
            "test_flow_kgh": float(self.test_flow_kgh),
            "flows": {
                "inputs": {
                    "tested_water": {
                        "value": self.output_flow_kgh / 1000.0,
                        "unit": "m3/h",
                        "source": "water_quality_test",
                        "flowtype": "WATER_MASS"
                    },
                    "electricity": {
                        "value": self.power_consumption_kw,
                        "unit": "kW",
                        "source": "grid_or_battery",
                        "flowtype": "ELECTRICAL_ENERGY"
                    }
                },
                "outputs": {
                    "ultrapure_water": {
                        "value": self.output_flow_kgh - self.test_flow_kgh,
                        "unit": "kg/h",
                        "destination": "ultrapure_storage_tank",
                        "flowtype": "WATER_MASS"
                    },
                    "test_sample": {
                        "value": self.test_flow_kgh,
                        "unit": "kg/h",
                        "destination": "test_lab",
                        "flowtype": "WATER_MASS"
                    }
                }
            }
        }
```

### 3. Ultrapure Water Storage Tank

A 5000L storage tank receiving ultrapure water from both the treatment block and SOEC electrolyzer.[2][1]

```python
class UltrapureWaterStorageTank(Component):
    """5000L ultrapure water storage with dual inputs."""
    
    def __init__(self, capacity_l: float = 5000.0):
        super().__init__()
        self.capacity_l = capacity_l
        self.capacity_kg = capacity_l  # Assuming water density ~1 kg/L
        self.current_mass_kg = 0.0
        self.temperature_c = 20.0
        self.pressure_bar = 1.0
        
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        if self.capacity_l <= 0:
            raise ValueError("Tank capacity must be positive")
    
    def fill(self, mass_kg: float) -> float:
        """Add water to tank, return actual amount added."""
        available = max(self.capacity_kg - self.current_mass_kg, 0.0)
        actual = min(mass_kg, available)
        self.current_mass_kg += actual
        return actual
    
    def withdraw(self, mass_kg: float) -> float:
        """Remove water from tank, return actual amount removed."""
        actual = min(mass_kg, self.current_mass_kg)
        self.current_mass_kg -= actual
        return actual
    
    def step(self, t: float) -> None:
        super().step(t)
        # Receive water from treatment block and SOEC
        
    def get_state(self) -> Dict[str, Any]:
        fill_ratio = self.current_mass_kg / self.capacity_kg if self.capacity_kg > 0 else 0.0
        return {
            **super().get_state(),
            "current_mass_kg": float(self.current_mass_kg),
            "capacity_kg": float(self.capacity_kg),
            "fill_ratio": float(fill_ratio),
            "temperature_c": float(self.temperature_c),
            "pressure_bar": float(self.pressure_bar),
            "flows": {
                "inputs": {
                    "from_treatment": {
                        "value": 0.0,  # Updated during step
                        "unit": "kg/h",
                        "source": "water_treatment",
                        "flowtype": "WATER_MASS"
                    },
                    "from_soec": {
                        "value": 0.0,  # Updated during step
                        "unit": "kg/h",
                        "source": "soec_electrolyzer",
                        "flowtype": "WATER_MASS"
                    }
                },
                "outputs": {
                    "to_pump_a": {
                        "value": 0.0,  # Updated during step
                        "unit": "kg/h",
                        "destination": "pump_a",
                        "flowtype": "WATER_MASS"
                    },
                    "to_pump_b": {
                        "value": 0.0,  # Updated during step
                        "unit": "kg/h",
                        "destination": "pump_b",
                        "flowtype": "WATER_MASS"
                    }
                }
            }
        }
```

### 4. Pump A and Pump B

Two separate pumps drawing from the storage tank with different power consumption.[1]

```python
class WaterPump(Component):
    """Generic water pump with configurable power."""
    
    def __init__(self, pump_id: str, power_kw: float, source: str):
        super().__init__()
        self.pump_id = pump_id
        self.power_kw = power_kw
        self.power_source = source  # "grid_or_battery" or "grid"
        self.flow_kgh = 0.0
        self.inlet_pressure_bar = 1.0
        self.outlet_pressure_bar = 5.0
        
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        if self.power_kw <= 0:
            raise ValueError(f"Pump {self.pump_id} power must be positive")
    
    def step(self, t: float) -> None:
        super().step(t)
        # Calculate flow based on available water and power
        
    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            "pump_id": self.pump_id,
            "flow_kgh": float(self.flow_kgh),
            "power_kw": float(self.power_kw),
            "inlet_pressure_bar": float(self.inlet_pressure_bar),
            "outlet_pressure_bar": float(self.outlet_pressure_bar),
            "flows": {
                "inputs": {
                    "water": {
                        "value": self.flow_kgh,
                        "unit": "kg/h",
                        "source": "ultrapure_storage_tank",
                        "flowtype": "WATER_MASS"
                    },
                    "electricity": {
                        "value": self.power_kw,
                        "unit": "kW",
                        "source": self.power_source,
                        "flowtype": "ELECTRICAL_ENERGY"
                    }
                },
                "outputs": {
                    "pressurized_water": {
                        "value": self.flow_kgh,
                        "unit": "kg/h",
                        "destination": f"downstream_{self.pump_id}",
                        "flowtype": "WATER_MASS"
                    }
                }
            }
        }
```

## YAML Configuration

The water treatment system can be configured using the zero-code YAML approach.[3]

```yaml
name: Water Treatment System
version: 1.0
description: Ultrapure water production and distribution system

water_treatment:
  quality_test:
    enabled: true
    sample_interval_hours: 1.0
    
  treatment_block:
    enabled: true
    max_flow_m3h: 10.0
    power_consumption_kw: 20.0
    output_temp_c: 20.0
    output_pressure_bar: 1.0
    
  ultrapure_storage:
    capacity_l: 5000.0
    initial_fill_ratio: 0.5
    input_sources:
      - water_treatment
      - soec_electrolyzer
      
  pumps:
    pump_a:
      enabled: true
      power_kw: 0.75
      power_source: grid_or_battery
      outlet_pressure_bar: 5.0
      
    pump_b:
      enabled: true
      power_kw: 1.5
      power_source: grid
      outlet_pressure_bar: 8.0

simulation:
  timestep_hours: 1.0
  duration_hours: 8760
  checkpoint_interval_hours: 168
```

## System Flow Diagram

The complete water treatment flow follows this topology:[3][2]

**External Network (Water, 10 m³/h)** → **Water Quality Test Block** → **Water Treatment Block** (consumes 20 kW) → splits into:
- **Test Sample** (1%)
- **Ultrapure Water** (99%) → **Ultrapure Storage Tank (5000L)**

**SOEC Electrolyzer** → **Ultrapure Storage Tank (5000L)**

**Ultrapure Storage Tank** → distributes to:
- **Pump A** (0.75 kW from grid/battery)
- **Pump B** (1.5 kW from grid)

## Integration Points

The water treatment system integrates with existing components through:[2][1]

- **ComponentRegistry**: All components register during PlantBuilder initialization
- **FlowTracker**: Flow metadata enables Sankey diagram generation showing water and energy flows
- **LUTManager**: Water property lookups for density and enthalpy calculations
- **Battery Storage**: Pump A can draw power from battery backup
- **SOEC Electrolyzer**: Provides recycled ultrapure water to storage tank
