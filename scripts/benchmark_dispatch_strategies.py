"""
Benchmark dispatch strategies on identical input scenarios.

Generates side-by-side comparison of:
- SOEC_ONLY: Single SOEC electrolyzer dispatch
- REFERENCE_HYBRID: Hybrid SOEC/PEM with arbitrage
- ECONOMIC_SPOT: Economic spot purchase for non-RFNBO H2

Usage:
    python scripts/benchmark_dispatch_strategies.py

Output:
    - dispatch_strategy_comparison.csv: Comparison table
    - Console summary with key metrics
"""
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from h2_plant.control.dispatch import (
    DispatchInput,
    DispatchState,
    SoecOnlyStrategy,
    ReferenceHybridStrategy,
    EconomicSpotDispatchStrategy
)


def generate_test_scenario(hours: int = 24) -> tuple:
    """Generate a realistic test scenario with varying prices and wind.
    
    Returns:
        tuple: (minutes, wind_mw, prices_eur_mwh)
    """
    np.random.seed(42)  # Reproducibility
    
    steps_per_hour = 60
    total_minutes = hours * steps_per_hour
    
    # Generate wind profile (capacity factor varies 30-90%)
    base_wind = 100.0  # MW installed capacity
    wind_mw = base_wind * (0.3 + 0.6 * np.sin(np.linspace(0, 4*np.pi, total_minutes)) + 
                           0.1 * np.random.randn(total_minutes))
    wind_mw = np.clip(wind_mw, 0, base_wind)
    
    # Generate price profile (typical day-ahead pattern)
    base_price = 50.0  # EUR/MWh average
    price_variation = 30.0 * np.sin(np.linspace(0, 2*np.pi, total_minutes))  # Daily cycle
    price_noise = 10.0 * np.random.randn(total_minutes)
    
    # Add some very cheap periods (negative prices during oversupply)
    oversupply_hours = np.random.choice(hours, size=hours//6, replace=False)
    for h in oversupply_hours:
        start_min = h * steps_per_hour
        end_min = start_min + steps_per_hour
        price_variation[start_min:end_min] -= 40.0
    
    prices_eur_mwh = base_price + price_variation + price_noise
    prices_eur_mwh = np.clip(prices_eur_mwh, -10.0, 150.0)  # Allow negative prices
    
    minutes = np.arange(total_minutes)
    
    return minutes, wind_mw, prices_eur_mwh


def run_strategy_benchmark(strategy, minutes: np.ndarray, wind_mw: np.ndarray, 
                           prices_eur_mwh: np.ndarray, config: dict) -> dict:
    """Run a dispatch strategy through the test scenario.
    
    Args:
        strategy: Dispatch strategy instance
        minutes: Minute indices
        wind_mw: Wind power offer (MW)
        prices_eur_mwh: Spot prices (EUR/MWh)
        config: Plant configuration
    
    Returns:
        dict: Results metrics
    """
    state = DispatchState()
    
    total_h2_kg = 0.0
    total_p_soec_mwh = 0.0
    total_p_pem_mwh = 0.0
    total_p_sold_mwh = 0.0
    total_revenue_eur = 0.0
    total_cost_eur = 0.0
    
    dt = 1.0 / 60.0  # 1 minute in hours
    
    for i, minute in enumerate(minutes):
        # Build input
        inputs = DispatchInput(
            minute=int(minute),
            P_offer=wind_mw[i],
            P_future_offer=wind_mw[min(i + 60, len(wind_mw) - 1)],
            current_price=prices_eur_mwh[i],
            soec_capacity_mw=config['soec_capacity_mw'],
            pem_max_power_mw=config['pem_max_power_mw'],
            soec_h2_kwh_kg=config['soec_h2_kwh_kg'],
            pem_h2_kwh_kg=config['pem_h2_kwh_kg'],
            ppa_price_eur_mwh=config['ppa_price_eur_mwh'],
            h2_price_eur_kg=config['h2_price_eur_kg'],
            h2_non_rfnbo_price_eur_kg=config['h2_non_rfnbo_price_eur_kg'],
            p_grid_max_mw=config['p_grid_max_mw']
        )
        
        # Dispatch decision
        result = strategy.decide(inputs, state)
        
        # Update state for next iteration
        state = DispatchState(
            P_soec_prev=result.P_soec,
            force_sell=result.state_update.get('force_sell', False)
        )
        
        # Accumulate metrics
        total_p_soec_mwh += result.P_soec * dt
        total_p_pem_mwh += result.P_pem * dt
        total_p_sold_mwh += result.P_sold * dt
        
        # H2 production
        h2_soec = (result.P_soec * dt * 1000) / config['soec_h2_kwh_kg'] if result.P_soec > 0 else 0
        h2_pem = (result.P_pem * dt * 1000) / config['pem_h2_kwh_kg'] if result.P_pem > 0 else 0
        total_h2_kg += h2_soec + h2_pem
        
        # Economics
        # Revenue: H2 sales (RFNBO at full price, non-RFNBO at lower price) + electricity sales
        h2_rfnbo_kg = result.state_update.get('h2_rfnbo_kg', h2_soec + h2_pem)
        h2_non_rfnbo_kg = result.state_update.get('h2_non_rfnbo_kg', 0.0)
        
        total_revenue_eur += h2_rfnbo_kg * config['h2_price_eur_kg']
        total_revenue_eur += h2_non_rfnbo_kg * config['h2_non_rfnbo_price_eur_kg']
        total_revenue_eur += result.P_sold * dt * prices_eur_mwh[i]
        
        # Cost: PPA for renewable, spot price for grid purchase
        renewable_used = result.P_soec + result.P_pem - result.state_update.get('spot_purchased_mw', 0.0)
        spot_used = result.state_update.get('spot_purchased_mw', 0.0)
        
        total_cost_eur += renewable_used * dt * config['ppa_price_eur_mwh']
        total_cost_eur += spot_used * dt * prices_eur_mwh[i]
    
    # RFNBO metrics (for Economic Spot strategy)
    h2_rfnbo_kg = getattr(strategy, 'h2_rfnbo_kg', total_h2_kg)
    h2_non_rfnbo_kg = getattr(strategy, 'h2_non_rfnbo_kg', 0.0)
    rfnbo_pct = (h2_rfnbo_kg / (h2_rfnbo_kg + h2_non_rfnbo_kg) * 100) if (h2_rfnbo_kg + h2_non_rfnbo_kg) > 0 else 100.0
    
    return {
        'total_h2_kg': total_h2_kg,
        'h2_rfnbo_kg': h2_rfnbo_kg,
        'h2_non_rfnbo_kg': h2_non_rfnbo_kg,
        'rfnbo_compliance_pct': rfnbo_pct,
        'total_p_soec_mwh': total_p_soec_mwh,
        'total_p_pem_mwh': total_p_pem_mwh,
        'total_p_sold_mwh': total_p_sold_mwh,
        'revenue_eur': total_revenue_eur,
        'cost_eur': total_cost_eur,
        'profit_eur': total_revenue_eur - total_cost_eur
    }


def main():
    """Run benchmark comparison."""
    print("=" * 70)
    print("DISPATCH STRATEGY BENCHMARK COMPARISON")
    print("=" * 70)
    
    # Plant configuration
    config = {
        'soec_capacity_mw': 80.0,
        'pem_max_power_mw': 30.0,
        'soec_h2_kwh_kg': 37.5,
        'pem_h2_kwh_kg': 50.0,
        'ppa_price_eur_mwh': 50.0,
        'h2_price_eur_kg': 9.6,  # RFNBO certified price
        'h2_non_rfnbo_price_eur_kg': 2.0,  # Non-certified price
        'p_grid_max_mw': 30.0
    }
    
    # Generate test scenario
    hours = 24
    print(f"\nGenerating {hours}-hour test scenario...")
    minutes, wind_mw, prices = generate_test_scenario(hours)
    
    print(f"  Wind power: {wind_mw.mean():.1f} MW avg (min: {wind_mw.min():.1f}, max: {wind_mw.max():.1f})")
    print(f"  Spot price: {prices.mean():.1f} EUR/MWh avg (min: {prices.min():.1f}, max: {prices.max():.1f})")
    
    # Run each strategy
    strategies = {
        'SOEC_ONLY': SoecOnlyStrategy(),
        'REFERENCE_HYBRID': ReferenceHybridStrategy(),
        'ECONOMIC_SPOT': EconomicSpotDispatchStrategy()
    }
    
    results = {}
    for name, strategy in strategies.items():
        print(f"\nRunning {name}...")
        results[name] = run_strategy_benchmark(strategy, minutes, wind_mw, prices, config)
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    
    metrics = ['total_h2_kg', 'h2_rfnbo_kg', 'h2_non_rfnbo_kg', 'rfnbo_compliance_pct',
               'total_p_soec_mwh', 'total_p_pem_mwh', 'total_p_sold_mwh',
               'revenue_eur', 'cost_eur', 'profit_eur']
    
    # Header
    print(f"\n{'Metric':<30} {'SOEC_ONLY':>15} {'REF_HYBRID':>15} {'ECON_SPOT':>15}")
    print("-" * 75)
    
    for metric in metrics:
        values = [results[name].get(metric, 0.0) for name in ['SOEC_ONLY', 'REFERENCE_HYBRID', 'ECONOMIC_SPOT']]
        
        if 'pct' in metric:
            print(f"{metric:<30} {values[0]:>14.1f}% {values[1]:>14.1f}% {values[2]:>14.1f}%")
        elif 'eur' in metric.lower():
            print(f"{metric:<30} {values[0]:>14,.0f} € {values[1]:>14,.0f} € {values[2]:>14,.0f} €")
        else:
            print(f"{metric:<30} {values[0]:>15,.1f} {values[1]:>15,.1f} {values[2]:>15,.1f}")
    
    # Export to CSV
    try:
        import pandas as pd
        df = pd.DataFrame(results).T
        output_path = Path('dispatch_strategy_comparison.csv')
        df.to_csv(output_path)
        print(f"\n✓ Comparison saved to: {output_path}")
    except ImportError:
        print("\n(pandas not available - CSV export skipped)")
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
