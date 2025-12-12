# Hydrogen Plant Simulation System

This project simulates a modular dual-path hydrogen production plant (Arbitration).
**Status:** Operational & Verified (Dec 2025). Supports Hybrid (Simple/Cascade) topologies with realistic Balance of Plant energy accounting.

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
## Troubleshooting

### Linux Qt/XCB Error
If you see an error like `qt.qpa.plugin: Could not load the Qt platform plugin "xcb"`, you are likely missing global system libraries.
Run:
```bash
sudo apt-get install libxcb-cursor0 libxcb-xinerama0
```

### Headless Execution (Colab/Servers)
If you are running in a headless environment (like Google Colab or a VM without a display), **do not run `main.py`**.
Instead, use the scenario runner which works without a GUI:
```bash
python run_scenarios.py
```
Or ensure you have an X server (like `xvfb`) running if you must test the GUI components.
To start the GUI simulation:

```bash
python3 run_gui_simulation.py
```

## Project Structure

- `h2_plant/`: Core package containing all simulation logic and components.
- `h2_plant/gui/`: Graphical User Interface implementation (PySide6/NodeGraphQt).
- `h2_plant/components/`: Modular plant components (Electrolyzers, Tanks, etc.).
- `messages/`: Inter-module communication definitions.
