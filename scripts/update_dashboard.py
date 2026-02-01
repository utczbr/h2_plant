import json
import logging
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from h2_plant.visualization.dashboard_generator import DashboardGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)

def update_dashboard():
    output_dir = Path("simulation_output_weekly")
    metrics_path = output_dir / "metrics" / "dashboard_data.json"
    
    if not metrics_path.exists():
        print(f"Error: Metrics file not found at {metrics_path}")
        return

    print(f"Loading metrics from {metrics_path}...")
    with open(metrics_path, 'r') as f:
        results = json.load(f)

    print("Regenerating dashboard...")
    generator = DashboardGenerator(output_dir)
    generator.generate(results)
    print("Dashboard updated successfully.")

if __name__ == "__main__":
    update_dashboard()
