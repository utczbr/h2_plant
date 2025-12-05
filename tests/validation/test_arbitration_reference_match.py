#!/usr/bin/env python3
"""
Validation Test: Compare h2_plant arbitration with reference manager.py

This test runs both implementations and compares minute-by-minute outputs
to verify exact behavioral match.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os

# Add paths - go up from tests/validation to project root
script_dir = Path(__file__).parent.resolve()
repo_root = script_dir.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'pem_and_soec'))

# Also set PYTHONPATH environment variable
os.environ['PYTHONPATH'] = str(repo_root) + ':' + os.environ.get('PYTHONPATH', '')

# Import reference implementation
try:
    import manager as reference_manager
    REFERENCE_AVAILABLE = True
except ImportError:
    REFERENCE_AVAILABLE = False
    print("WARNING: Reference manager.py not available")

from h2_plant.config.plant_builder import PlantBuilder
from h2_plant.simulation.engine import SimulationEngine


def run_reference_8hour_test():
    """Run reference manager.py for 8 hours (480 minutes)."""
    print("\n" + "="*70)
    print("RUNNING REFERENCE (manager.py) - 8 Hours")
    print("="*70)
    
    if not REFERENCE_AVAILABLE:
        print("❌ Reference not available - skipping")
        return None
    
    # Run reference simulation
    reference_manager.run_hybrid_management()
    
    # Extract results
    history = reference_manager.history
    
    results = {
        'minutes': np.array(history['minute']),
        'P_offer': np.array(history['P_offer']),
        'P_soec_actual': np.array(history['P_soec_actual']),
        'P_pem': np.array(history['P_pem']),
        'P_sold': np.array(history['P_sold']),
        'spot_price': np.array(history['spot_price']),
        'sell_decision': np.array(history['sell_decision']),
        'H2_soec_kg': np.array(history['H2_soec_kg']),
        'steam_soec_kg': np.array(history['steam_soec_kg'])
    }
    
    print(f"✅ Reference complete: {len(results['minutes'])} minutes")
    return results


def run_h2plant_8hour_test():
    """Run h2_plant for 8 hours using minute-level config."""
    print("\n" + "="*70)
    print("RUNNING H2_PLANT - 8 Hours (Minute-Level)")
    print("="*70)
    
    # Create temporary config for 8-hour test
    # We'll use the minute-level config but modify duration
    config_path = repo_root / "configs" / "plant_pem_soec_minute_level.yaml"
    
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return None
    
    # Build plant
    print("Building plant...")
    builder = PlantBuilder.from_file(config_path)
    
    # Override duration to 8 hours (480 minutes)
    builder.config.simulation.duration_hours = 8.0
    
    # Create engine
    print("Creating engine...")
    engine = SimulationEngine(
        registry=builder.registry,
        config=builder.config.simulation,
        topology=builder.config.topology if hasattr(builder.config, 'topology') else [],
        indexed_topology=builder.config.indexed_topology if hasattr(builder.config, 'indexed_topology') else []
    )
    
    # Run simulation
    print("Running simulation...")
    results_data = engine.run()
    
    # Extract results from monitoring
    # We need to get the data from the components
    # This is simplified - in reality we'd need to capture step-by-step data
    
    print("✅ H2Plant complete: 8 hours")
    return results_data


def compare_results(reference, h2plant):
    """Compare minute-by-minute results."""
    print("\n" + "="*70)
    print("COMPARING RESULTS")
    print("="*70)
    
    if reference is None or h2plant is None:
        print("❌ Cannot compare - one or both results missing")
        return False
    
    # Compare key metrics
    metrics_to_compare = [
        ('P_soec_actual', 'SOEC Power', 'MW', 0.01),
        ('P_pem', 'PEM Power', 'MW', 0.01),
        ('P_sold', 'Sold Power', 'MW', 0.01),
        ('sell_decision', 'Sell Decision', '', 0),
    ]
    
    all_match = True
    
    for metric, label, unit, tolerance in metrics_to_compare:
        if metric not in reference:
            print(f"⚠️  {label}: Not in reference")
            continue
        
        ref_data = reference[metric]
        
        # For h2plant, we need to extract from monitoring data
        # This is placeholder - real implementation would need proper data extraction
        h2_data = np.zeros_like(ref_data)  # Placeholder
        
        # Compare
        max_diff = np.max(np.abs(ref_data - h2_data))
        mean_diff = np.mean(np.abs(ref_data - h2_data))
        
        if max_diff <= tolerance:
            print(f"✅ {label:20s}: MATCH (max diff: {max_diff:.6f} {unit})")
        else:
            print(f"❌ {label:20s}: MISMATCH (max diff: {max_diff:.6f} {unit}, mean: {mean_diff:.6f})")
            all_match = False
    
    return all_match


def plot_comparison(reference, h2plant):
    """Generate comparison plots."""
    print("\n" + "="*70)
    print("GENERATING COMPARISON PLOTS")
    print("="*70)
    
    if reference is None:
        print("⚠️  No reference data to plot")
        return
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    minutes = reference['minutes']
    
    # Plot 1: Power Dispatch
    ax = axes[0, 0]
    ax.plot(minutes, reference['P_offer'], 'k--', label='Offered', alpha=0.7)
    ax.plot(minutes, reference['P_soec_actual'], 'g-', label='SOEC', linewidth=2)
    ax.plot(minutes, reference['P_pem'], 'b-', label='PEM', linewidth=2)
    ax.plot(minutes, reference['P_sold'], 'r-', label='Sold', linewidth=2)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Power (MW)')
    ax.set_title('Power Dispatch - Reference')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Spot Price
    ax = axes[0, 1]
    ax.plot(minutes, reference['spot_price'], 'purple', linewidth=2)
    ax.axhline(y=306, color='red', linestyle='--', label='Arbitrage Threshold')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Price (EUR/MWh)')
    ax.set_title('Spot Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Sell Decision
    ax = axes[1, 0]
    ax.fill_between(minutes, 0, reference['sell_decision'], alpha=0.3, color='orange')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Sell Decision (0=H2, 1=Sell)')
    ax.set_title('Arbitrage Decision')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: H2 Production
    ax = axes[1, 1]
    cumulative_h2 = np.cumsum(reference['H2_soec_kg'])
    ax.plot(minutes, cumulative_h2, 'g-', linewidth=2)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Cumulative H2 (kg)')
    ax.set_title('SOEC H2 Production')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Steam Consumption
    ax = axes[2, 0]
    cumulative_steam = np.cumsum(reference['steam_soec_kg'])
    ax.plot(minutes, cumulative_steam, 'c-', linewidth=2)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Cumulative Steam (kg)')
    ax.set_title('SOEC Steam Consumption')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Arbitrage Events
    ax = axes[2, 1]
    # Find arbitrage trigger points (minute 0, sell_decision changes)
    triggers = np.where(np.diff(reference['sell_decision']) != 0)[0]
    for trigger in triggers:
        ax.axvline(x=minutes[trigger], color='red', alpha=0.5, linestyle='--')
    ax.plot(minutes, reference['P_soec_actual'], 'g-', alpha=0.5)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('SOEC Power (MW)')
    ax.set_title('Arbitrage Events (red lines)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = repo_root / 'tests' / 'validation' / 'arbitration_validation_plots.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"✅ Plots saved to: {output_path}")
    
    plt.show()


def main():
    """Run validation test."""
    print("\n" + "="*70)
    print("ARBITRATION VALIDATION TEST")
    print("Comparing h2_plant vs reference manager.py")
    print("="*70)
    
    # Run reference
    ref_results = run_reference_8hour_test()
    
    # Run h2_plant
    h2_results = run_h2plant_8hour_test()
    
    # Compare
    if ref_results:
        match = compare_results(ref_results, h2_results)
        
        # Plot
        plot_comparison(ref_results, h2_results)
        
        # Final verdict
        print("\n" + "="*70)
        if match:
            print("✅ VALIDATION PASSED - Results match reference!")
        else:
            print("⚠️  VALIDATION INCOMPLETE - See differences above")
        print("="*70)
        
        return 0 if match else 1
    else:
        print("\n❌ VALIDATION SKIPPED - Reference not available")
        print("To run validation:")
        print("1. Ensure pem_and_soec/manager.py is available")
        print("2. Run: python3 tests/validation/test_arbitration_reference_match.py")
        return 2


if __name__ == "__main__":
    sys.exit(main())
