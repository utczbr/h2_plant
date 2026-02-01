from h2_plant.orchestrator import Orchestrator
from h2_plant.reporting.report_generator import ReportGenerator
import os

def test_reporting():
    scenarios_dir = "/home/stuart/Documentos/Planta Hidrogenio/scenarios"
    orchestrator = Orchestrator(scenarios_dir)
    
    # Run short simulation (24 hours)
    print("Running simulation...")
    history = orchestrator.run_simulation(hours=24)
    
    # Load Config
    import yaml
    config_path = os.path.join(scenarios_dir, "visualization_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Generate Reports
    print("Generating reports...")
    reporter = ReportGenerator(output_dir="test_reports", config=config)
    reporter.generate_all(history)
    
    # Verify files exist
    expected_files = [
        "dispatch_strategy_stacked.png",
        "dispatch_detailed_overlay.png",
        "energy_price_over_time.png",
        "total_h2_production_stacked.png",
        "pem_h2_production_over_time.png",
        "soec_h2_production_over_time.png",
        "cumulative_h2_production.png",
        "pem_cell_voltage_over_time.png",
        "soec_active_modules_over_time.png",
        "report_energy_pie.png",
        "oxygen_production_stacked.png",
        "water_consumption_stacked.png",
        "dispatch_curve_scatter.png",
        "price_histogram.png",
        "arbitrage_scatter.png",
        "soec_modules_temporal.png",
        "soec_modules_stats.png",
        "compressor_power_stacked.png"
    ]
    
    for f in expected_files:
        path = os.path.join("test_reports", f)
        if os.path.exists(path):
            print(f"PASS: {f} created.")
        else:
            print(f"FAIL: {f} not found.")

if __name__ == "__main__":
    test_reporting()
