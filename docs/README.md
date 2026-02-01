# Hydrogen Production Plant Simulation System

A modular, dual-path hydrogen production system with advanced simulation capabilities. This system implements a complete software architecture for simulating hydrogen production facilities with both electrolyzer and autothermal reforming (ATR) pathways.

## Features

- **Modular Architecture**: Five-layer architecture with clear separation of concerns.
- **Dual-Path Production**: Separate electrolyzer and ATR production pathways.
- **Advanced Simulation**: Event-driven engine with checkpointing and monitoring.
- **Performance Optimized**: Numba JIT compilation and LUT-based property lookups.
- **Configuration Driven**: YAML-based plant setup without code changes.
- **Comprehensive Testing**: 95%+ test coverage with performance benchmarks.
- **External Inputs**: Models external sources for oxygen and waste heat.
- **Battery Storage**: Simulates a BESS for grid backup and load leveling.
- **Thermodynamic Gas Mixer**: Rigorous multi-component gas mixing with phase equilibrium.
- **Water Treatment System**: Detailed modeling of ultrapure water production and distribution.
- **Enhanced GUI**: Visual node editor with collapse functionality, organized property tabs, and comprehensive parameter configuration.

## Architecture Layers

### 1. Core Foundation (h2_plant/core/)
- `Component` abstract base class with standardized lifecycle.
- `ComponentRegistry` for dependency injection and management.

### 2. Performance Optimization (h2_plant/optimization/)
- `LUTManager` for thermodynamic property caching.
- Numba-compiled hot-path operations.

### 3. Component Standardization (h2_plant/components/)
- **Production**: `ElectrolyzerSource`, `ATRProductionSource`, `DetailedPEMElectrolyzer`.
- **Storage**: `TankArray`, `SourceIsolatedTanks`, `OxygenBuffer`, `BatteryStorage`.
- **Compression**: `FillingCompressor`, `OutgoingCompressor`.
- **Mixing**: `OxygenMixer`, `MultiComponentMixer`.
- **External**: `ExternalOxygenSource`, `ExternalHeatSource`.
- **Water**: `WaterQualityTestBlock`, `WaterTreatmentBlock`, `UltrapureWaterStorageTank`, `WaterPump`.
- **Utility**: `DemandScheduler`, `EnergyPriceTracker`.

### 4. Configuration System (h2_plant/config/)
- Declarative plant configuration via YAML/JSON.
- `PlantBuilder` factory for configuration-driven assembly.

### 5. Pathway Integration (h2_plant/pathways/)
- `IsolatedProductionPath` for self-contained production chains.
- `DualPathCoordinator` with multiple allocation strategies.

### 6. Simulation Engine (h2_plant/simulation/)
- Modular execution framework with state persistence and event scheduling.
- `FlowTracker` for topology-aware flow analysis.

## Quick Start

```bash
# Install package in development mode
pip install -e ".[dev]"

# Create configuration
cp configs/plant_baseline.yaml configs/my_plant.yaml
# Edit my_plant.yaml with your specifications

# Run simulation
python -m h2_plant.simulation.runner configs/my_plant.yaml
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=h2_plant --cov-report=html

# Run performance benchmarks
pytest tests/performance/ --benchmark-only

# Run tests by category
pytest tests/components/
pytest tests/core/
pytest tests/integration/
pytest tests/e2e/
```

## Directory Structure

```
h2_plant/                    # Main package
├── core/                   # Layer 1: Foundation components
├── optimization/           # Layer 2: Performance enhancements
├── components/             # Layer 3: Standardized components
│   ├── production/
│   ├── storage/
│   ├── compression/
│   ├── mixing/
│   ├── external/
│   ├── water/
│   └── utility/
├── pathways/               # Layer 4: Pathway orchestration
├── config/                 # Layer 5: Configuration system
├── simulation/             # Layer 6: Execution engine
└── legacy/                 # Backward compatibility layer

tests/
├── components/
│   └── water/
├── core/
├── integration/
├── performance/
└── e2e/
```

