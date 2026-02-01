# Headless Operation Guide

This guide explains how to run the H2 Plant Simulation without the Graphical User Interface (GUI). This is useful for:
- Batch processing on remote servers/HPC.
- Automated testing and CI/CD.
- Long-duration simulations.

## Recommended Method: Integrated Dispatch Runner

The primary headless entry point is `h2_plant/run_integrated_simulation.py`.

### Basic Usage

Use Python's `-m` flag to run the module, pointing it to your configuration directory:

```bash
python -m h2_plant.run_integrated_simulation scenarios
```

By default, this will:
1. Load configuration from `scenarios/`.
2. Run the simulation for the duration specified in `scenarios/simulation_config.yaml`.
3. Save results to `scenarios/simulation_output/`.

### Options

| Argument | Description | Example |
|---|---|---|
| `scenarios_dir` | Path to the directory containing config files. | `configs/test_case_A` |
| `--hours N` | Override simulation duration. | `--hours 8760` (1 year) |
| `--output-dir PATH` | Custom output directory. | `--output-dir ./results_v1` |
| `--verbose` | Enable debug logging. | `--verbose` |

**Example:**
```bash
python -m h2_plant.run_integrated_simulation scenarios --hours 168 --output-dir ./weekly_test
```

---

## Batch Processing: Scenario Runner

For running multiple scenarios sequentially (e.g., comparing topologies), use `run_scenarios.py`.

### Usage

1. Open `run_scenarios.py`.
2. Edit the `SCENARIOS` list to include your scenario definitions:
   ```python
   SCENARIOS = [
       {"name": "PEM_Only", "topology": "path/to/topology_pem.yaml"},
       {"name": "Hybrid", "topology": "path/to/topology_hybrid.yaml"},
   ]
   ```
3. Run the script:
   ```bash
   python run_scenarios.py
   ```

It will execute each scenario and print a comparison table at the end.

---

## Configuration Directory Structure

The `scenarios_dir` must contain the following files:

- **`simulation_config.yaml`**: Time settings (start, duration, timestep) and data paths.
- **`physics_parameters.yaml`**: Component behavior parameters (efficiency, degradation).
- **`economics_parameters.yaml`**: Prices and cost assumptions.
- **`plant_topology.yaml`**: Graph definition of components and connections.
- **`plant_topology.yaml.bak`**: (Optional) Backup if auto-generated.
- **Data Files**: CSVs for energy prices and wind/solar profiles (referenced in config).
