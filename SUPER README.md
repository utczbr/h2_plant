# h2_plant: Hydrogen Production Plant Simulation System

## Overview

**h2_plant** is a comprehensive Python-based simulation framework for modeling and analyzing hydrogen production facilities. It provides a modular, extensible architecture for simulating dual-path hydrogen production systems that combine electrolyzer technology (PEM and SOEC) with autothermal reforming (ATR), enabling researchers and engineers to evaluate different production strategies, optimize plant performance, and analyze economic viability.

The system addresses a critical need in the transition to clean energy: understanding how hydrogen production facilities operate under varying conditions, how different production pathways interact, and what economic conditions make various hydrogen production strategies viable. By providing detailed component-level simulation with rigorous thermodynamic modeling, h2_plant enables users to move beyond simplified calculations to comprehensive facility analysis.

The simulation engine implements an event-driven architecture capable of modeling annual operation cycles (8,760 hours) with sub-hourly resolution, capturing the dynamic interactions between renewable energy availability, grid pricing, storage operations, and product demand. This makes it particularly valuable for analyzing renewable hydrogen production where electricity costs and availability fluctuate significantly.

### Key Capabilities

The system provides several distinct capabilities that set it apart from simpler hydrogen production models:

**Dual-Path Production Coordination**: h2_plant uniquely supports simultaneous operation of multiple production pathways—PEM electrolysis for green hydrogen from renewable electricity, solid oxide electrolysis (SOEC) for high-efficiency operation when heat and electricity are both available, and autothermal reforming for blue hydrogen production from natural gas with carbon capture. The dual-path coordinator intelligently allocates electricity and natural gas resources based on economic signals and operational constraints.

**Detailed Thermodynamic Modeling**: Rather than using simplified efficiency curves, the system implements rigorous thermodynamic models including lookup table (LUT) management for property calculations, Numba JIT-compiled performance-critical routines, and comprehensive gas mixture handling with phase equilibrium calculations. This enables accurate modeling of compression work, heat recovery, and energy balances across the entire plant.

**Configuration-Driven Architecture**: All plant specifications are defined in YAML configuration files, enabling users to modify plant designs, operational strategies, and economic parameters without modifying source code. The PlantBuilder factory constructs complete simulation models from these configurations, handling component instantiation, dependency wiring, and initialization automatically.

**Comprehensive Component Library**: The system includes pre-built components for every major element of hydrogen production facilities: electrolyzers and reformers, hydrogen storage tanks at various pressure levels, compressors and expanders, water treatment systems, oxygen management, battery energy storage, thermal management equipment, and utility components like demand schedulers and energy price trackers.

### Target Users

h2_plant serves multiple user communities:

- **Research Engineers**: Evaluating novel hydrogen production configurations or operational strategies
- **Plant Designers**: Sizing equipment and optimizing plant configuration before capital investment
- **Energy Analysts**: Assessing the economic viability of hydrogen production under various market conditions
- **Grid Operators**: Understanding how hydrogen production facilities can provide grid services
- **Academic Researchers**: Studying hydrogen production economics and operational dynamics

---

## Installation Instructions

### Prerequisites

Before installing h2_plant, ensure your development environment meets the following requirements:

**Python Version**: Python 3.9 or higher is required. The package has been tested with Python 3.9, 3.10, and 3.11. You can verify your Python version by running:

```bash
python --version
```

**System Dependencies**: The following system packages are required for the core dependencies:

- On Ubuntu/Debian: `sudo apt-get install build-essential python3-dev`
- On macOS: `xcode-select --install` (for LLVM/clang)
- On Windows: Visual Studio Build Tools with C++ support

### Complete Dependencies

The project requires the following Python packages. These are automatically installed when using the setup scripts:

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| numpy | >=1.24.0 | Numerical computing |
| numba | >=0.57.0 | JIT compilation for performance |
| scipy | >=1.9.0 | Scientific computing |
| pandas | >=2.0.0 | Data manipulation |
| pyyaml | >=6.0 | YAML configuration parsing |
| pydantic | >=2.0.0 | Data validation |
| h5py | >=3.1.0 | HDF5 checkpoint files |
| CoolProp | >=6.4.0 | Thermodynamic properties |
| PySide6 | >=6.4.0 | GUI framework |
| NodeGraphQt | >=0.6.1 | Node graph GUI |
| matplotlib | >=3.5.0 | Plotting library |
| plotly | >=5.0.0 | Interactive visualizations |
| windpowerlib | >=0.2.2 | Wind power modeling |
| entsoe-py | >=0.5.0 | Energy market data |
| pyarrow | >=8.0.0 | Parquet file support |
| psutil | >=5.9.0 | System utilities |
| tqdm | >=4.64.0 | Progress bars |
| pytest | >=7.0.0 | Testing framework |

### Installation Methods

#### Method 1: Windows PowerShell Setup (Recommended for Windows)

The repository includes an automated setup script that handles all installation steps including downloading required cache files:

```powershell
# Clone the repository
git clone https://github.com/utczbr/h2_plant.git
cd h2_plant

# Run the setup script
.\setup.ps1
```

The `setup.ps1` script performs the following actions:

1. **LFS Cache Download**: Creates `.h2_plant/lut_cache` directory and downloads thermodynamic lookup table cache files from Google Drive:
   - `lut_CH4_v1.pkl` - Methane properties
   - `lut_CO2_v1.pkl` - Carbon dioxide properties
   - `lut_H2_v1.pkl` - Hydrogen properties
   - `lut_H2O_v1.pkl` - Water properties
   - `lut_N2_v1.pkl` - Nitrogen properties
   - `lut_O2_v1.pkl` - Oxygen properties
   - `lut_water_saturation_v1.pkl` - Water saturation data

2. **Python Environment Setup**: Creates a virtual environment (`.venv`) and installs all dependencies

3. **Package Installation**: Installs the local project in editable mode

After running the script, activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

#### Method 2: Manual Installation from Source

For users who prefer manual control or are using Linux/macOS:

```bash
# Clone the repository
git clone https://github.com/utczbr/h2_plant.git
cd h2_plant

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .

# Verify installation
python -c "import h2_plant; print('h2_plant installed successfully')"
```

#### Method 3: Using setup.py Directly

You can also install using the setup.py script with optional dependency groups:

```bash
# Clone and navigate to repository
git clone https://github.com/utczbr/h2_plant.git
cd h2_plant

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Core dependencies only
pip install -e .

# With development tools
pip install -e ".[dev]"

# With GUI dependencies
pip install -e ".[gui]"

# With visualization tools
pip install -e ".[viz]"

# With CoolProp for LUT generation
pip install -e ".[coolprop]"

# Full installation (all extras)
pip install -e ".[dev,gui,viz,coolprop]"
```

#### Method 4: Conda Installation

```bash
# Create conda environment
conda create -n h2_plant python=3.11
conda activate h2_plant

# Install core dependencies
conda install numpy numba scipy pandas

# Install h2_plant from source
cd h2_plant
pip install -e .
```

### Console Script Entry Point

After installation, you can use the built-in console script:

```bash
# Run simulation using the console script (pass the config file path)
h2-simulate scenarios/plant_topology.yaml
```

This invokes `h2_plant.simulation.runner:main` as defined in the setup.py entry points, which uses argparse with `config_file` as a positional argument, plus `--output` and `--resume` options.

### Post-Installation Verification

After installation, verify that all components are working correctly:

```bash
# Run the test suite
pytest tests/ -v

# Check component imports
python -c "
from h2_plant.core.component import Component
from h2_plant.config.plant_builder import PlantBuilder
from h2_plant.simulation.engine import SimulationEngine
from h2_plant.components.production.pem_electrolyzer import DetailedPEMElectrolyzer
print('All core imports successful')
"

# List available components
python -c "
from h2_plant.components import *
print('Component library loaded successfully')
"
```

### Troubleshooting Common Issues

**Missing LFS Cache Files**: If you encounter errors about missing LUT cache files, run `setup.ps1` (Windows) or manually download the cache files from Google Drive to the `.h2_plant/lut_cache` directory.

**Numba Compilation Errors**: If you encounter errors related to Numba JIT compilation, ensure you have a compatible LLVM installation. On Ubuntu: `sudo apt-get install llvm`. On macOS: LLVM is typically pre-installed or available via Homebrew.

**PySide6 GUI Issues**: If the GUI fails to start, try installing the platform-specific Qt libraries directly. For headless servers, you may need to set `QT_QPA_PLATFORM=offscreen`.

**Import Errors**: Ensure you have installed all dependencies with `pip install -r requirements.txt`. Missing optional dependencies can cause import failures.

---

## Component Configuration (YAML)

### Configuration System Overview

h2_plant uses a declarative configuration system that defines plant specifications in YAML files located in the `scenarios/` directory. The configuration is processed by the `ConfigLoader` and `PlantGraphBuilder` classes, which construct the complete simulation model from node-based topology definitions. This approach separates configuration from code, enabling users to modify plant designs, test different scenarios, and conduct parametric studies without writing Python code.

The configuration system uses several YAML files within each scenario directory:

- **plant_topology.yaml**: Defines the physical components (nodes), their parameters, and connections (edges)
- **simulation_config.yaml**: Controls simulation duration, timestep, output options, and solver settings
- **physics_parameters.yaml**: Specifies thermodynamic properties, efficiency curves, and physical constants
- **economics_parameters.yaml**: Contains capital costs, operational costs, and economic assumptions
- **visualization_config.yaml**: Configures dashboard appearance and plotting options

### Node-Based Topology Configuration

The plant topology is defined using a node-and-connection graph structure. Each node represents a component, and connections define the flow of materials, energy, and signals between components.

#### Basic Structure

```yaml
scenario_name: "Your Plant Description"

nodes:
  - id: "component_name"
    type: "ComponentType"
    params:
      param1: value1
      param2: value2
    connections:
      - source_port: "output_port"
        target_name: "target_component"
        target_port: "input_port"
        resource_type: "stream"  # or "gas", "signal"
```

#### Key Configuration Elements

**Node Definition**: Each node requires an ID, type, and parameters. The type must match a valid component class in the system.

**Connection Definition**: Connections define how components are linked. Each connection specifies:

- `source_port`: The output port name on the source component
- `target_name`: The ID of the target component
- `target_port`: The input port name on the target component
- `resource_type`: The type of resource being transferred (`stream`, `gas`, `signal`)

**System Grouping**: Components can be grouped using `system_group` and `process_step` parameters for visualization and execution ordering.

### Complete YAML Configuration Example

Below is a comprehensive example demonstrating the node-based topology format:

```yaml
# =============================================================================
# H2_PLANT TOPOLOGY CONFIGURATION
# =============================================================================
# This example demonstrates a dual-path hydrogen production facility
# combining PEM electrolysis with SOEC (Solid Oxide Electrolysis).
#
# Edit this file to customize your plant configuration, then run:
#   h2-simulate scenarios/your_scenario/plant_topology.yaml
# =============================================================================

scenario_name: "Plant PEM+SOEC (No ATR) - 30 tanks with 52.33 m3 each"

nodes:
  # =============================================================================
  # CENTRALIZED COOLING UTILITY
  # =============================================================================
  - id: "cooling_manager"
    type: "CoolingManager"
    params:
      dc_total_area_m2: 5000.0
      dc_air_flow_kg_s: 1000.0
      dc_u_value: 35.0
      glycol_inventory_kg: 50000.0
      tower_design_load_kw: 300.0
      t_dry_bulb_c: 25.0
      t_wet_bulb_c: 18.0
      inertia_alpha: 0.3
      system_group: "Utilities"
      process_step: 0

  # =============================================================================
  # WATER SUPPLY CHAIN
  # =============================================================================
  # External Water Source
  - id: "Water_Source"
    type: "ExternalWaterSource"
    params:
      mode: "external_control"
      flow_rate_kg_h: 12000.0
      pressure_bar: 5.0
      temperature_c: 25.0
      system_group: "Water_Supply"
      process_step: 1
    connections:
      - source_port: "water_out"
        target_name: "Water_Purifier"
        target_port: "raw_water_in"
        resource_type: "stream"

  # Reverse Osmosis Water Purifier
  - id: "Water_Purifier"
    type: "WaterPurifier"
    params:
      max_flow_kg_h: 12000.0
      recovery_ratio: 0.75
      system_group: "Water_Supply"
      process_step: 2
    connections:
      - source_port: "ultrapure_out"
        target_name: "UltraPure_Tank"
        target_port: "ultrapure_in"
        resource_type: "stream"

  # Ultra-Pure Water Tank
  - id: "UltraPure_Tank"
    type: "UltraPureWaterTank"
    params:
      capacity_kg: 20000.0
      nominal_production_kg_h: 10000.0
      initial_fill_fraction: 0.7
      system_group: "Water_Supply"
      process_step: 3
    connections:
      - source_port: "control_signal"
        target_name: "Water_Source"
        target_port: "control_signal"
        resource_type: "signal"
      - source_port: "water_out_SOEC"
        target_name: "SOEC_Makeup_Mixer"
        target_port: "makeup_water_in"
        resource_type: "stream"
      - source_port: "water_out_PEM"
        target_name: "PEM_Makeup_Mixer"
        target_port: "makeup_water_in"
        resource_type: "stream"

  # =============================================================================
  # POWER TRANSFORMERS (Efficiency Loss Modeling)
  # =============================================================================
  - id: "SOEC_Transformer"
    type: "PowerTransformer"
    params:
      efficiency: 0.95
      rated_power_mw: 15.25
      system_group: "SOEC"
      process_step: 8

  - id: "PEM_Transformer"
    type: "PowerTransformer"
    params:
      efficiency: 0.95
      rated_power_mw: 5.57
      system_group: "PEM"
      process_step: 9

  - id: "BOP_Transformer"
    type: "PowerTransformer"
    params:
      efficiency: 0.95
      rated_power_mw: 5.0
      system_group: "BOP"
      process_step: 7

  # =============================================================================
  # SOEC CLUSTER (Solid Oxide Electrolysis)
  # =============================================================================
  - id: "SOEC_Cluster"
    type: "SOEC"
    params:
      num_modules: 6
      max_power_nominal_mw: 2.4
      optimal_limit: 0.80
      steam_input_ratio_kg_per_kg_h2: 10.3
      power_first_step_mw: 0.12
      ramp_step_mw: 0.24
      lifecycle: 61320
      process_step: 10
    connections:
      - source_port: "h2_out"
        target_name: "SOEC_H2_Interchanger_1"
        target_port: "hot_in"
        resource_type: "stream"
      - source_port: "o2_out"
        target_name: "SOEC_O2_Interchanger_1"
        target_port: "hot_in"
        resource_type: "stream"

  # =============================================================================
  # PEM ELECTROLYZER
  # =============================================================================
  - id: "PEM_Electrolyzer"
    type: "PEM"
    params:
      max_power_mw: 5.35
      use_polynomials: true
      water_excess_factor: 0.02
      out_pressure_pa: 4000000.0
      lifecycle: 87600
    connections:
      - source_port: "h2_out"
        target_name: "PEM_H2_KOD_1"
        target_port: "gas_inlet"
        resource_type: "stream"
      - source_port: "oxygen_out"
        target_name: "PEM_O2_KOD_1"
        target_port: "gas_inlet"
        resource_type: "stream"

  # =============================================================================
  # COMPRESSION TRAIN (Example: Stage 1)
  # =============================================================================
  - id: "SOEC_H2_Compressor_S1"
    type: "CompressorSingle"
    params:
      max_flow_kg_h: 400.0
      max_temp_c: 135.0
      temperature_limited: true
      outlet_pressure_bar: 200.0
      process_step: 100
    connections:
      - source_port: "outlet"
        target_name: "SOEC_H2_Intercooler_1"
        target_port: "fluid_in"
        resource_type: "stream"

  # Intercooler
  - id: "SOEC_H2_Intercooler_1"
    type: "DryCooler"
    params:
      target_outlet_temp_c: 40.0
      design_capacity_kw: 1800.0
      use_central_utility: true
      process_step: 101
    connections:
      - source_port: "fluid_out"
        target_name: "SOEC_H2_Compressor_S2"
        target_port: "h2_in"
        resource_type: "stream"

  # =============================================================================
  # PURIFICATION STAGE (PSA)
  # =============================================================================
  - id: "SOEC_H2_PSA_1"
    type: "PSA Unit"
    params:
      purity_target: 0.99995
      process_step: 300
    connections:
      - source_port: "purified_gas_out"
        target_name: "H2_Production_Mixer"
        target_port: "inlet_1"
        resource_type: "gas"

  # =============================================================================
  # STORAGE TANKS (Example: Single Tank)
  # =============================================================================
  - id: "H2_Tank_1"
    type: "H2StorageTank"
    params:
      capacity_kg: 50.0
      pressure_bar: 350.0
      temperature_k: 298.15
      system_group: "Storage"
      process_step: 500
```

### Component Types Reference

The system supports many component types. Below are the most commonly used:

#### Production Components

| Component Type | Description | Key Parameters |
|---------------|-------------|----------------|
| `SOEC` | Solid Oxide Electrolysis Cluster | `num_modules`, `max_power_nominal_mw`, `optimal_limit`, `lifecycle` |
| `PEM` | Proton Exchange Membrane Electrolyzer | `max_power_mw`, `use_polynomials`, `water_excess_factor`, `lifecycle` |
| `ATR` | Autothermal Reformer | `max_flow_kg_h`, `efficiency`, `reactor_temperature_k` |
| `ElectricBoiler` | Electric Steam Boiler | `max_power_kw`, `target_outlet_temp_c`, `efficiency` |

#### Storage Components

| Component Type | Description | Key Parameters |
|---------------|-------------|----------------|
| `H2StorageTank` | Hydrogen Storage Tank | `capacity_kg`, `pressure_bar`, `temperature_k` |
| `UltraPureWaterTank` | Ultrapure Water Storage | `capacity_kg`, `nominal_production_kg_h` |
| `BatteryStorage` | Battery Energy Storage | `capacity_mwh`, `max_power_mw`, `round_trip_efficiency` |

#### Compression Components

| Component Type | Description | Key Parameters |
|---------------|-------------|----------------|
| `CompressorSingle` | Single-Stage Compressor | `max_flow_kg_h`, `max_temp_c`, `outlet_pressure_bar` |
| `CompressorMulti` | Multi-Stage Compressor | Number of stages, intercooler configuration |

#### Separation and Purification

| Component Type | Description | Key Parameters |
|---------------|-------------|----------------|
| `PSA Unit` | Pressure Swing Adsorption | `purity_target` |
| `KnockOutDrum` | Gas-Liquid Separator | `diameter_m`, `delta_p_bar` |
| `DeoxoReactor` | Deoxygenation Reactor | - |
| `Chiller` | Refrigeration Unit | `cooling_capacity_kw`, `target_temp_k` |

#### Thermal Management

| Component Type | Description | Key Parameters |
|---------------|-------------|----------------|
| `DryCooler` | Air-Cooled Heat Exchanger | `target_outlet_temp_c`, `use_central_utility` |
| `CoolingManager` | Central Cooling System | `dc_total_area_m2`, `tower_design_load_kw` |
| `Interchanger` | Heat Recovery Exchanger | `min_approach_temp_k`, `target_cold_out_temp_c` |

#### Water and Steam

| Component Type | Description | Key Parameters |
|---------------|-------------|----------------|
| `WaterPurifier` | Reverse Osmosis Unit | `max_flow_kg_h`, `recovery_ratio` |
| `WaterPumpThermodynamic` | Thermodynamic Water Pump | `capacity_kg_h`, `target_pressure_pa` |
| `Attemperator` | Steam Temperature Control | `target_temp_k`, `max_water_flow_kg_h` |

#### Utility Components

| Component Type | Description | Key Parameters |
|---------------|-------------|----------------|
| `PowerTransformer` | Electrical Transformer | `efficiency`, `rated_power_mw` |
| `Valve` | Flow Control Valve | `P_out_pa`, `fluid` |
| `Mixer` | Stream Combiner | `volume_m3`, `continuous_flow` |
| `StreamSplitter` | Stream Divider | `split_ratio` |

---

## CLI Command Reference

### Primary Entry Point

The main CLI interface for running simulations is through the `h2-simulate` console script or directly via `h2_plant.simulation.runner`. This module provides the primary entry point for executing plant simulations.

### Command Syntax

```bash
h2-simulate <config_file> [options]
```

Or equivalently:

```bash
python -m h2_plant.simulation.runner <config_file> [options]
```

### Positional Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `config_file` | string | **Required.** Path to the plant topology YAML file (e.g., `scenarios/plant_topology.yaml`). |

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output` | Path | `<config_file>/simulation_output` | Directory for simulation outputs. |
| `--resume` | int | None | Hour number to resume simulation from (for checkpoint recovery). |

### Usage Examples

**Basic Simulation Run:**

```bash
# Run simulation using the console script
h2-simulate scenarios/plant_topology.yaml

# Run with custom output directory
h2-simulate scenarios/plant_topology.yaml --output ./my_results
```

**Resume Interrupted Simulation:**

```bash
# Resume from specific hour (e.g., hour 2400)
h2-simulate scenarios/plant_topology.yaml --resume 2400
```

### Integrated Simulation Runner

For running simulations with dispatch strategies, use the integrated runner module programmatically:

```python
from pathlib import Path
from h2_plant.run_integrated_simulation import run_with_dispatch_strategy
import numpy as np

# Run simulation from scenarios directory
history = run_with_dispatch_strategy(
    scenarios_dir="scenarios/",
    hours=8760,  # Optional: override duration
    output_dir=Path("./simulation_output"),  # Optional: custom output
    strategy="ECONOMIC_SPOT"  # Optional: dispatch strategy
)

# Access simulation results
print(f"Total H2 produced: {np.sum(history['h2_kg']):.1f} kg")
print(f"SOEC power: {np.sum(history['P_soec_actual']):.1f} MWh")
print(f"PEM power: {np.sum(history['P_pem_actual']):.1f} MWh")
```

### GUI Launch

For users preferring a graphical interface, the system includes a GUI launcher:

```bash
# Launch the GUI
python -m h2_plant.gui.main
```

On Windows, you can also use the provided batch file:

```bash
# Run the Windows GUI launcher
run_gui_windows.bat
```

On Linux, a shell script is provided:

```bash
# Run the Linux GUI launcher
bash run_gui_debian.sh
```

### Testing Commands

The system includes comprehensive test suites that can be run via pytest:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=h2_plant --cov-report=html

# Run specific test categories
pytest tests/components/      # Component tests
pytest tests/core/           # Core framework tests
pytest tests/integration/    # Integration tests
pytest tests/e2e/           # End-to-end tests

# Run performance benchmarks
pytest tests/performance/ --benchmark-only

# Run specific test file
pytest tests/core/test_component.py -v
```

---

## Tool Usage Guide

The h2_plant repository includes a comprehensive suite of utility tools located in the `tools/` and `scripts/` directories. These tools support various tasks including topology validation, economics regeneration, visualization, data analysis, and system diagnostics.

### Economics and Calculation Tools

#### regenerate_capex.py

**Purpose**: Recalculates capital expenditure (CAPEX) values based on updated component specifications.

**Usage**:

```bash
python tools/regenerate_capex.py <scenarios_dir> [options]
```

**Positional Arguments**:

| Argument | Type | Description |
|----------|------|-------------|
| `scenarios_dir` | string | **Required.** Path to the scenarios directory containing YAML configuration files. |

**Optional Arguments**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output-dir` | Path | `<scenarios_dir>/economics_output` | Directory for CAPEX output files. |
| `--capex-config` | Path | None | Custom CAPEX configuration file path. |
| `--capacity-mode` | string | `design` | Capacity mode: `design` or `history`. |

**Description**: This tool computes the total capital investment required for the plant based on component sizes, quantities, and unit costs. It uses the physics and economics parameters to estimate equipment costs, installation, and auxiliary systems. Output includes detailed cost breakdown by subsystem.

#### regenerate_lcoh.py

**Purpose**: Recalculates the Levelized Cost of Hydrogen (LCOH) for the configured plant.

**Usage**:

```bash
python tools/regenerate_lcoh.py <simulation_output_dir> [options]
```

**Positional Arguments**:

| Argument | Type | Description |
|----------|------|-------------|
| `simulation_output_dir` | string | **Required.** Path to the simulation output directory (containing metrics/). |

**Optional Arguments**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--economics-dir` | Path | `<simulation_output_dir>/economics` | Directory for economics output files. |
| `--discount-rate` | float | 0.08 | Discount rate for LCOH calculation (0.08 = 8%). |
| `--operating-hours` | int | 8760 | Annual operating hours. |
| `--electrolyzer-efficiency` | float | None | Override electrolyzer efficiency. |

**Description**: Computes the LCOH metric using the specified economic parameters, including capital costs, operational costs, maintenance, and projected production. The tool supports multiple LCOH methodologies including OPEX/LCOH thesis methodology for design mode calculations.

**Output Metrics**:

- Total CAPEX (EUR)
- Annual OPEX (EUR/year)
- Levelized Cost of Hydrogen (EUR/kg)
- Production cost breakdown
- Sensitivity analysis results

#### regenerate_graphs.py

**Purpose**: Generates visualization graphs from simulation results.

**Usage**:

```bash
python tools/regenerate_graphs.py <output_dir> [options]
```

**Positional Arguments**:

| Argument | Type | Description |
|----------|------|-------------|
| `output_dir` | string | **Required.** Path to the simulation output directory. |

**Optional Arguments**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--timeout` | int | 300 | Timeout in seconds for graph generation. |
| `--max-memory-mb` | int | 4096 | Maximum memory usage in MB. |
| `--downscale` | string | `none` | Time series downsampling: `none`, `hourly`, or `daily`. |

**Description**: Creates standard plots from simulation output including production time series, storage levels, price profiles, and component utilization. Output formats include PNG for static images and HTML for interactive Plotly visualizations.

#### regenerate_scenario_summary.py

**Purpose**: Generates comprehensive scenario summary reports.

**Usage**:

```bash
python tools/regenerate_scenario_summary.py <output_dir> [options]
```

**Positional Arguments**:

| Argument | Type | Description |
|----------|------|-------------|
| `output_dir` | string | **Required.** Path to the simulation output directory. |

**Optional Arguments**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--scenarios-dir` | Path | None | Path to scenarios directory. |
| `--scenario-name` | string | None | Name of the scenario. |

**Description**: Combines results from multiple scenario runs into a single comparison report. Outputs key metrics, performance indicators, and economic results in both JSON and HTML formats.

#### rewrite_economics_approx_design.py

**Purpose**: Updates approximate design economics calculations with new parameters.

**Usage**:

```bash
python tools/rewrite_economics_approx_design.py <scenarios_dir> [options]
```

**Positional Arguments**:

| Argument | Type | Description |
|----------|------|-------------|
| `scenarios_dir` | string | **Required.** Path to the scenarios directory. |

**Optional Arguments**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output-dir` | Path | None | Output directory for results. |

**Description**: This tool recalculates simplified economic models using updated assumptions. It provides quick LCOH estimates without running full simulations, useful for preliminary design studies and sensitivity analysis.

### Topology and Structure Tools

#### check_topology_order.py

**Purpose**: Validates the topological ordering of components in the plant configuration.

**Usage**:

```bash
python tools/check_topology_order.py <config_file>
```

**Description**: This tool analyzes the component graph defined in the configuration file and verifies that all dependencies are properly ordered. It detects circular dependencies, missing connections, and ordering violations that could cause simulation failures.

### Data Generation and Analysis Tools

#### generate_scenario_visual_layout.py

**Purpose**: Creates visual layout diagrams of the plant topology.

**Usage**:

```bash
python tools/generate_scenario_visual_layout.py --config <config_file> --output <layout_file>
```

**Description**: Generates schematic diagrams showing component arrangements, flow paths, and connectivity. Useful for documentation and understanding plant configuration.

#### create_audit_test_data.py

**Purpose**: Creates test data files for audit verification.

**Usage**:

```bash
python tools/create_audit_test_data.py --output <data_dir>
```

**Description**: Generates reference datasets that can be used to verify simulation correctness and for audit trail purposes. Creates deterministic test cases with known expected outputs.

### Debugging and Diagnostics Scripts

The `scripts/` directory contains extensive debugging and validation scripts. Below are the most commonly used:

#### debug_stream_table.py

**Purpose**: Validates thermodynamic property calculations against reference values.

**Usage**:

```bash
python scripts/debug_stream_table.py
```

**Description**: Tests the stream property calculation system (enthalpy, entropy, composition) against known values to verify thermodynamic model correctness.

#### debug_mass_loss.py

**Purpose**: Diagnoses mass balance issues in simulation results.

**Usage**:

```bash
python scripts/debug_mass_loss.py --results <results_dir>
```

**Description**: Analyzes simulation output to identify components or time periods where mass is not conserved. Outputs detailed mass flow breakdowns to help locate imbalances.

#### diagnose_bottleneck.py

**Purpose**: Identifies capacity constraints limiting plant performance.

**Usage**:

```bash
python scripts/diagnose_bottleneck.py --results <results_dir>
```

**Description**: Analyzes component utilization and identifies which equipment is operating at capacity limits, constraining production or causing inefficiencies.

#### profile_simulation.py

**Purpose**: Profiles simulation performance to identify bottlenecks.

**Usage**:

```bash
python scripts/profile_simulation.py --config <config_file>
```

**Description**: Runs the simulation with profiling enabled, measuring execution time for each component and identifying optimization opportunities.

#### validate_precision.py

**Purpose**: Validates numerical precision of simulation results.

**Usage**:

```bash
python scripts/validate_precision.py --results <results_dir>
```

**Description**: Checks for numerical instability, precision loss, or convergence issues in simulation results.

### Running Tool Chains

For comprehensive analysis workflows, multiple tools can be chained together:

```bash
# Full analysis pipeline
h2-simulate scenarios/plant_topology.yaml --output sim_results
python tools/regenerate_graphs.py sim_results/metrics
python tools/regenerate_capex.py scenarios/ --output-dir capex_output
python tools/regenerate_lcoh.py sim_results/metrics --economics-dir sim_results/economics
python tools/regenerate_scenario_summary.py sim_results
```

---

## Architecture Deep Dive

### System Architecture

h2_plant implements a sophisticated architecture that separates concerns and enables modular development:

**Core Layer** (`h2_plant/core/`)

The foundation layer provides essential abstractions used throughout the system:

- `Component`: Abstract base class defining the component lifecycle (initialize, step, get_state)
- `ComponentRegistry`: Dependency injection container managing all simulation components
- `ComponentID`: Enumerated identifiers for standard components
- `PlantGraphBuilder`: Builds component graphs from topology configurations

**Component Layer** (`h2_plant/components/`)

The component layer implements the physical equipment models organized by function:

- **Production**: SOEC, PEM, ATR, ElectricBoiler
- **Storage**: H2StorageTank, UltraPureWaterTank, BatteryStorage
- **Compression**: CompressorSingle, CompressorMulti
- **Separation**: PSA Unit, KnockOutDrum, DeoxoReactor
- **Thermal**: DryCooler, CoolingManager, Interchanger, Chiller
- **Water**: WaterPurifier, WaterPumpThermodynamic, Attemperator
- **Utility**: PowerTransformer, Valve, Mixer, StreamSplitter

**Simulation Layer** (`h2_plant/simulation/`)

The execution layer runs the simulation:

- `SimulationEngine`: Main simulation loop with checkpointing and monitoring
- Hybrid dispatch strategies for economic optimization

**Configuration Layer** (`h2_plant/config/`)

The configuration layer translates declarative YAML into executable models:

- `ConfigLoader`: Loads and validates scenario configurations
- `PlantGraphBuilder`: Constructs component graphs from node-based topology

### Component Lifecycle

All components implement a standardized lifecycle:

1. **Construction**: Component is instantiated with configuration parameters
2. **Initialization**: `initialize(dt, registry)` is called once before simulation starts
3. **Simulation Loop**: `step(t)` is called for each timestep (up to 8,760 times for annual simulation)
4. **State Query**: `get_state()` returns current state for monitoring/checkpointing

---

## Troubleshooting

### Common Installation Issues

**Numba Compilation Errors**: If you encounter errors related to Numba JIT compilation, ensure you have a compatible LLVM installation. On Ubuntu: `sudo apt-get install llvm`. On macOS: LLVM is typically pre-installed or available via Homebrew.

**PySide6 GUI Issues**: If the GUI fails to start, try installing the platform-specific Qt libraries directly. For headless servers, you may need to set `QT_QPA_PLATFORM=offscreen`.

**Import Errors**: Ensure you have installed all dependencies with `pip install -e ".[dev,gui,viz]"`. Missing optional dependencies can cause import failures.

### Common Simulation Issues

**Configuration Validation Errors**: If the ConfigLoader reports configuration errors, verify your YAML syntax using a YAML validator. Common issues include incorrect indentation, missing required fields, invalid component types, or malformed connections.

**Missing Data Files**: Ensure all referenced data files (energy prices, wind data) exist in the expected locations. The simulation expects these files relative to the scenarios directory.

**Convergence Failures**: If the solver fails to converge, try increasing the timestep temporarily, or check for physically unrealistic parameters (e.g., negative efficiencies, zero capacities).

**Memory Issues**: For long-duration simulations with detailed components, reduce memory usage by setting shorter checkpoint intervals or using HDF5 output format instead of CSV.

---

## Additional Resources

### Testing and Validation

The project includes comprehensive test coverage:

```bash
# Unit tests for individual components
pytest tests/components/ -v

# Core framework tests
pytest tests/core/ -v

# Integration tests
pytest tests/integration/ -v

# End-to-end tests
pytest tests/e2e/ -v

# Performance benchmarks
pytest tests/performance/ --benchmark-only
```

### Configuration Examples

The `scenarios/` directory contains example configurations demonstrating various plant configurations:

- `plant_topology.yaml`: Complete PEM+SOEC+ATR plant with full process train
- `plant_topology_PEM+SOEC.yaml`: PEM+SOEC plant without ATR
- `simulation_config.yaml`: Simulation parameters
- `economics_parameters.yaml`: Economic assumptions
- `physics_parameters.yaml`: Thermodynamic parameters

### Understanding Output

Simulation outputs are written to the specified output directory:

- `metrics/`: Main output directory containing simulation results
  - `timeseries.csv`: Hour-by-hour component states and flows
  - `summary.json`: Aggregated metrics and KPIs
- `history_chunks/`: Periodic saves of simulation state in Parquet format
- `economics/`: Economic analysis results
  - `lcoh_report.json/csv`: Levelized Cost of Hydrogen analysis
  - `capex_report.json`: Capital expenditure breakdown
  - `opex_report.json`: Operational expenditure details

---

## License and Contributing

This project is released under the MIT License. Contributions are welcome. Please ensure all tests pass before submitting pull requests, and maintain the existing code style as enforced by the configured linters (black, flake8, isort).

---

*Document Version: 2.0*
*Last Updated: 2026-03-06*
*Author: MiniMax Agent*
