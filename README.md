# Hydrogen Plant Simulation System

This project simulates a modular dual-path hydrogen production plant (Arbitration).

## Prerequisites

- Python 3.10+
- Virtual environment (recommended)

## Setup

1.  Clone the repository.
2.  Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

To start the GUI simulation:

```bash
python3 run_gui_simulation.py
```

## Project Structure

- `h2_plant/`: Core package containing all simulation logic and components.
- `h2_plant/gui/`: Graphical User Interface implementation (PySide6/NodeGraphQt).
- `h2_plant/components/`: Modular plant components (Electrolyzers, Tanks, etc.).
- `messages/`: Inter-module communication definitions.
