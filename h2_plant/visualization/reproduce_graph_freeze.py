
import pandas as pd
import numpy as np
import time
import logging
import sys
from pathlib import Path

# Fix paths to allow imports
sys.path.append('/home/stuart/Documentos/Planta Hidrogenio')

from h2_plant.visualization.graphs import thermal, economics, production, performance, soec, storage
from h2_plant.visualization import utils

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Verification')

def create_mock_history(rows=100000):
    logger.info(f"Creating mock history with {rows} rows...")
    df = pd.DataFrame({
        'minute': np.arange(rows) * 60,
        'P_soec': np.random.rand(rows) * 100,
        'P_pem': np.random.rand(rows) * 50,
        'P_sold': np.random.rand(rows) * 20,
        'compressor_power_kw': np.random.rand(rows) * 5000,
        'spot_price': np.random.rand(rows) * 150,
        'H2_soec_kg': np.random.rand(rows),
        'H2_pem_kg': np.random.rand(rows),
        'o2_production_kg_h': np.random.rand(rows) * 100,
        'tank_level_kg': np.random.rand(rows) * 1000,
        'tank_pressure_bar': np.random.rand(rows) * 200,
        # Thermal data
        'cooling_manager_glycol_duty_kw': np.random.rand(rows) * 1000,
        'cooling_manager_cw_duty_kw': np.random.rand(rows) * 1000,
        'cooling_manager_glycol_supply_temp_c': np.random.rand(rows) * 50,
        'cooling_manager_cw_supply_temp_c': np.random.rand(rows) * 30,
        'CoolingManager_cooling_load_kw': np.random.rand(rows) * 100,
        # SOEC specific
        'soec_active_modules': np.random.randint(0, 50, rows),
        # PPA
        'ppa_price_effective_eur_mwh': np.random.rand(rows) * 80,
    })
    
    # Add module powers for heatmap
    for i in range(10):
        df[f'soec_module_powers_{i}'] = np.random.rand(rows) * 2.0
    
    return df

def test_graph_function(func, df, name):
    logger.info(f"Testing {name}...")
    start_time = time.time()
    try:
        fig = func(df, ['Component_1'], f"Test {name}", {})
        end_time = time.time()
        duration = end_time - start_time
        
        if fig is None:
            logger.warning(f"Result: SKIPPED (returned None) in {duration:.4f}s")
            return
            
        # Check point count in first axis
        points_plotted = 0
        if fig.axes:
            ax = fig.axes[0]
            for line in ax.lines:
                points_plotted = max(points_plotted, len(line.get_xdata()))
        
        logger.info(f"Result: SUCCESS in {duration:.4f}s | Max points plotted: {points_plotted}")
        
        if points_plotted > 5000:
             logger.error(f"FAILURE: {name} plotted {points_plotted} points (Expected < 5000). Downsampling failed!")
        else:
             logger.info(f"PASS: Downsampling verified for {name}")

    except Exception as e:
        logger.error(f"Result: FAILED with error: {e}")

def run_tests():
    # 525600 minutes = 1 year
    rows = 100000  # Enough to be slow without downsampling, fast with it
    df = create_mock_history(rows)
    
    tests = [
        ("thermal.plot_thermal_time_series", thermal.plot_thermal_time_series),
        ("thermal.plot_central_cooling_performance", thermal.plot_central_cooling_performance),
        ("economics.plot_time_series", economics.plot_time_series),
        ("economics.plot_effective_ppa", economics.plot_effective_ppa),
        ("economics.plot_arbitrage", economics.plot_arbitrage),
        ("production.plot_time_series", production.plot_time_series),
        ("production.plot_cumulative", production.plot_cumulative),
        ("performance.plot_time_series", performance.plot_time_series),
        ("soec.plot_active_modules", soec.plot_active_modules),
        ("storage.plot_tank_levels", storage.plot_tank_levels),
        ("storage.plot_compressor_power", storage.plot_compressor_power),
    ]
    
    for name, func in tests:
        test_graph_function(func, df, name)

if __name__ == "__main__":
    run_tests()
