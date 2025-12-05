# Hydrogen Plant Simulation System

This project simulates a modular dual-path hydrogen production plant.

## Project Structure

- `h2_plant/`: Main package containing the simulation logic.
- `scripts/`: Execution scripts for running simulations and validations.
- `vendor/`: Third-party or legacy dependencies (e.g., `libs`).
- `configs/`: Configuration files for simulations.
- `Hidrogenio/`: Python virtual environment.

## Setup

The virtual environment is located in `Hidrogenio`. Activate it using:

```bash
source Hidrogenio/bin/activate
```

## Running Scripts

All scripts have been moved to the `scripts/` directory. They are configured to run from the project root or from within the `scripts/` directory, but running from the root is recommended for path consistency with config files.

Example:

```bash
# Run the 8-hour validation test
python3 scripts/run_8hour_validation.py

# Run the minute-level simulation
python3 scripts/run_minute_level_simulation.py
```

## Key Scripts

- `run_8hour_validation.py`: Validates the system against a reference manager for 8 hours.
- `run_minute_level_simulation.py`: Runs a full-year minute-level simulation.
- `test_system_quick.py`: A quick system test to verify components and basic simulation.
