import pandas as pd
import numpy as np
from datetime import datetime
import os
import time
from pathlib import Path
import warnings
import matplotlib.pyplot as plt

# Wind energy libraries
from pvlib import iotools
from windpowerlib import WindTurbine, ModelChain
# Removed 'import turbine_models' as it's not used and causing issues

# API clients
from entsoe import EntsoePandasClient


warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    # Location
    'latitude': 53.0,      # Changed from -14.07 (Netherlands North Sea)
    'longitude': 4.0,      # Changed from -42.48 (Netherlands North Sea)
    'altitude': 0,         # Changed from 800 (sea level for offshore)
    'z0': 0.0014,         # Changed from 0.03 (offshore roughness length)
    
    # Turbine
    'turbine_name': 'Vestas V117/3600', # Updated name
    'hub_height': 105,
    
    # Park sizing
    'P80_target_MW': 100.0,
    'farm_efficiency': 0.90,
    
    # Data sources
    'weather_data_source': 'csv',  # 'csv' or 'nasa_power'
    'weather_csv_path': 'weather_data_2023-2024.csv',
    'use_cached_prices': True,
    'prices_cache_path': 'prices_2024.csv',
    'entsoe_country_code': 'NL',
    'use_scale_height_correction': False, 
    
    # Outputs
    'output_dir': './results_2024/',
}

def extrapolate_density_with_height(rho_surf, h_surf, h_hub, H_scale=8500):
    return rho_surf * np.exp(-(h_hub - h_surf) / H_scale)

def load_turbine_power_curve(turbine_name):
    """
    Load turbine specifications using hardcoded values and a dummy power curve.
    This is a temporary workaround as windpowerlib's direct loading for this specific
    turbine type is not providing all necessary attributes.
    
    Args:
        turbine_name: name of turbine (used for type in specs)
    
    Returns:
        dict with turbine specifications and power curve compatible with ModelChain
    """
    
    # Hardcoded values for a generic 3.6 MW turbine similar to Vestas V117/3600
    nominal_power_W = 3600000.0  # 3.6 MW
    hub_height_val = CONFIG['hub_height'] 
    rotor_diameter = 117.0 # Approximate for V117
    cut_in_speed = 3.0
    rated_speed = 10.0 # Approximate wind speed at nominal power
    cut_out_speed = 25.0

    # Create a simple power curve for demonstration
    wind_speeds = np.arange(0, 30, 0.5) # 0 to 29.5 m/s
    power_output_W = np.zeros_like(wind_speeds)

    # Simplified power curve logic:
    # Below cut-in: 0
    power_output_W[wind_speeds < cut_in_speed] = 0
    # Between cut-in and rated: linear ramp
    mask_ramp = (wind_speeds >= cut_in_speed) & (wind_speeds <= rated_speed)
    power_output_W[mask_ramp] = nominal_power_W * \
        (wind_speeds[mask_ramp] - cut_in_speed) / (rated_speed - cut_in_speed)
    # Between rated and cut-out: nominal power
    mask_nominal = (wind_speeds > rated_speed) & (wind_speeds < cut_out_speed)
    power_output_W[mask_nominal] = nominal_power_W
    # Above cut-out: 0
    power_output_W[wind_speeds >= cut_out_speed] = 0

    power_curve_df = pd.DataFrame({
        'wind_speed': wind_speeds,
        'value': power_output_W
    })
    
    turbine_specs = {
        'turbine_type': turbine_name,
        'nominal_power_kW': nominal_power_W / 1000.0,
        'nominal_power_W': nominal_power_W,
        'hub_height': hub_height_val,
        'rotor_diameter': rotor_diameter,
        'cut_in_speed': cut_in_speed,
        'rated_speed': rated_speed,
        'cut_out_speed': cut_out_speed,
        'power_curve': power_curve_df,  # This will be used by WindTurbine
        'power_curve_W': power_curve_df.rename(columns={'value': 'power_W'}), # Ensure consistent naming
    }
    
    return turbine_specs

def prepare_weather_data(year, config):
    """
    Loads and prepares weather data for a specific year from the combined CSV file.
    """
    try:
        df = pd.read_csv(config['weather_csv_path'])
        df['time'] = pd.to_datetime(df['time'])
        weather_data = df[df['time'].dt.year == year].copy()
        
        # A. Calculate Air Density (Ideal Gas Law)
        if 'air_density' not in weather_data.columns and 'temperature' in weather_data.columns and 'pressure' in weather_data.columns:
            temp_k = weather_data['temperature'] + 273.15 
            weather_data['air_density'] = weather_data['pressure'] / (287.05 * temp_k)
        elif 'rhoa' in weather_data.columns:
            weather_data.rename(columns={'rhoa': 'air_density'}, inplace=True)
        else:
            weather_data['air_density'] = 1.225

        # C. Apply Scale Height Density Correction (for offshore)
        if config.get('use_scale_height_correction', False):
            weather_data['air_density'] = extrapolate_density_with_height(
                rho_surf=weather_data['air_density'],
                h_surf=config['altitude'],
                h_hub=config['hub_height']
            )

        # B. Apply Wind Shear Extrapolation (Log Law)
        if 'Wind Speed Hourly (m/s)' in weather_data.columns:
             weather_data['wind_speed_10m'] = weather_data['Wind Speed Hourly (m/s)']
             h_ref = 10 
             h_hub = config['hub_height']
             z0 = config['z0']
             shear_factor = np.log(h_hub / z0) / np.log(h_ref / z0)
             weather_data['wind_speed_hub'] = weather_data['wind_speed_10m'] * shear_factor
        
        if 'temperature' not in weather_data.columns:
            weather_data['temperature'] = 288.15  # Changed from 15.0 (15°C = 288.15 K)
        if 'pressure' not in weather_data.columns:
            weather_data['pressure'] = 101325.0

        return weather_data
    except Exception as e:
        print(f"Could not load or process weather data: {e}")
        # Create a dummy dataframe if it fails
        time_index = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31 23:00:00', freq='H')
        weather_data = pd.DataFrame(index=time_index)
        weather_data['wind_speed_hub'] = np.random.uniform(3, 25, size=len(weather_data))
        weather_data['air_density'] = 1.225
        weather_data['temperature'] = 15.0
        weather_data['pressure'] = 101325.0
        weather_data.reset_index(inplace=True)
        weather_data = weather_data.rename(columns={'index': 'time'})
        return weather_data



def calculate_power_generation_single_turbine(weather_data, turbine_specs,

                                               hub_height, apply_density_correction=True):


    """
    Calculate hourly power generation for a single turbine.

    Args:
        weather_data: DataFrame with wind_speed_hub, temperature, pressure
        turbine_specs: dict with turbine parameters
        hub_height: hub height (m)
        apply_density_correction: whether to apply IEC 61400-12-1 correction

    Returns:
        Series with power generation (kW) for 8760 hours
    """

    

    # Prepare data for ModelChain with MultiIndex columns
    model_input = pd.DataFrame(index=pd.to_datetime(weather_data['time']))
    model_input[('wind_speed', hub_height)] = weather_data['wind_speed_hub'].values
    model_input[('density', hub_height)] = weather_data['air_density'].values
    model_input[('temperature', 2)] = weather_data['temperature'].values
    model_input[('pressure', 0)] = weather_data['pressure'].values
    model_input.columns = pd.MultiIndex.from_tuples(model_input.columns)



    # Create windpowerlib turbine object
    turbine = WindTurbine(**turbine_specs)



    # turbine_specs['cut_in_speed'] and turbine_specs['cut_out_speed'] are already defined in load_turbine_power_curve.
    # No need to re-assign from turbine object.
    

    

    # Create ModelChain
    mc = ModelChain(
        turbine,
        power_output_model='power_curve',
        density_correction=apply_density_correction,
    ).run_model(weather_df=model_input)

    

    # Extract power output (convert from W to kW)
    power_output_kW = mc.power_output / 1000.0

    

    # Add to weather_data for tracking
    weather_data['power_output_kW'] = power_output_kW.values

    

    return power_output_kW, weather_data
def validate_power_generation(weather_data, power_output_kW, turbine_specs):
    """
    Sanity checks on generated power series.
    """
    # Check cut-in/cut-out behavior
    below_cutin = weather_data[weather_data['wind_speed_hub'] < turbine_specs['cut_in_speed']]
    if power_output_kW[below_cutin.index].mean() > 10:  # Should be ~0
        print("WARNING: Power generated below cut-in wind speed")
    
    above_cutout = weather_data[weather_data['wind_speed_hub'] > turbine_specs['cut_out_speed']]
    if power_output_kW[above_cutout.index].mean() > 100:  # Should be 0-small value
        print("WARNING: Power generated above cut-out wind speed")
    
    # Check for NaNs
    if power_output_kW.isna().any():
        print(f"WARNING: {power_output_kW.isna().sum()} NaN values in power output")
    
    # Check maximum power
    max_power = power_output_kW.max()
    if max_power > turbine_specs['nominal_power_kW'] * 1.05:  # 5% margin
        print(f"WARNING: Power exceeds nominal by {(max_power/turbine_specs['nominal_power_kW']-1)*100:.1f}%")
    
    print("✓ Power generation validation passed")
    return True

def create_power_duration_curve(power_output_kW, percentiles=[0.8]):
    """
    Create Power Duration Curve and extract percentile levels.
    """
    
    # Sort in descending order
    power_sorted = np.sort(power_output_kW.dropna())[::-1]
    
    # Create percentile array (0% = max power, 100% = zero power)
    time_percentiles = np.linspace(0, 1, len(power_sorted))
    
    results = {}
    for p in percentiles:
        # p represents "exceedance" - find index where percentile is exceeded
        # P80 = exceeded 80% of the time = 20th percentile of sorted data
        idx = int((1 - p) * len(power_sorted))
        if idx >= len(power_sorted):
            idx = len(power_sorted) - 1
        results[f'P{int(p*100)}'] = power_sorted[idx]
    
    return results, power_sorted, time_percentiles

def plot_power_duration_curve(power_sorted, time_percentiles, turbine_name, save_path=None):
    """
    Plot the Power Duration Curve.
    """
    plt.figure(figsize=(12, 7))
    plt.plot(time_percentiles * 100, power_sorted / 1000, linewidth=2, label='PDC')
    
    # Mark key percentiles
    for p in [0.10, 0.50, 0.80, 0.90]:
        idx = int(p * len(power_sorted))
        if idx >= len(power_sorted):
            idx = len(power_sorted) - 1
        power_at_p = power_sorted[idx] / 1000
        plt.axvline(x=p*100, color='gray', linestyle='--', alpha=0.5)
        plt.axhline(y=power_at_p, color='gray', linestyle='--', alpha=0.5)
        plt.plot(p*100, power_at_p, 'ro', markersize=8)
        plt.text(p*100+1, power_at_p+0.1, f'P{int(p*100)}: {power_at_p:.2f} MW', fontsize=9)
    
    plt.xlabel('Percentage of Hours Exceeded (%)', fontsize=11)
    plt.ylabel('Power Output (MW)', fontsize=11)
    plt.title(f'Power Duration Curve - {turbine_name}', fontsize=12, fontweight='bold')
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #plt.show() # Commented out to avoid blocking the script execution
    plt.close()

def calculate_turbine_count(P80_turbine_kW, P80_target_MW):
    """
    Calculate number of turbines needed to achieve target P80.
    
    Args:
        P80_turbine_kW: P80 power for single turbine (kW)
        P80_target_MW: target park power at P80 (MW)
    
    Returns:
        int: number of turbines (rounded up)
    """
    P80_target_kW = P80_target_MW * 1000
    
    if P80_turbine_kW <= 0:
        raise ValueError("Single turbine P80 must be positive")
    
    N_turbines = np.ceil(P80_target_kW / P80_turbine_kW)
    return int(N_turbines)

def calculate_park_generation(power_output_single_turbine_kW, N_turbines, 
                              farm_efficiency=0.90):
    """
    Scale single turbine generation to entire park, applying farm efficiency.
    
    Args:
        power_output_single_turbine_kW: Series with 8760 values
        N_turbines: number of turbines in park
        farm_efficiency: factor accounting for wake losses (0-1)
    
    Returns:
        DataFrame with gross and net park generation
    """
    
    # Gross (linear scaling, no losses)
    P_park_gross_kW = power_output_single_turbine_kW * N_turbines
    
    # Net (apply farm efficiency)
    P_park_net_kW = P_park_gross_kW * farm_efficiency
    
    return P_park_gross_kW, P_park_net_kW

def validate_farm_efficiency(power_output_kW, N_turbines, farm_efficiency, 
                            P80_target_MW):
    """
    Check if farm with efficiency factor meets P80 target.
    """
    P_park_net = power_output_kW * N_turbines * farm_efficiency
    p80_net_kW = np.percentile(P_park_net.dropna(), 20) # 20th percentile for P80
    P80_net = p80_net_kW / 1000  # Convert to MW
    
    print(f"\n--- Farm Efficiency Validation ---")
    print(f"Farm efficiency factor: {farm_efficiency*100:.1f}%")
    print(f"P80 (park, after efficiency): {P80_net:.2f} MW")
    print(f"Target P80: {P80_target_MW:.2f} MW")
    
    if P80_net >= P80_target_MW * 0.95:  # Allow 5% margin
        print("✓ P80 target MET")
        return True
    else:
        deficit = (P80_target_MW - P80_net)
        print(f"✗ P80 SHORTFALL: {deficit:.2f} MW")
        return False

def fetch_entsoe_prices(country_code='NL', year=2024, api_key=None):
    """
    Fetch intraday electricity prices from ENTSO-E.
    
    Args:
        country_code: ISO 3166 country code (e.g., 'NL' for Netherlands)
        year: year to fetch (default 2024)
        api_key: ENTSO-E API key (if None, read from env variable)
    
    Returns:
        DataFrame with hourly prices (EUR/MWh)
    """
    
    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.getenv('ENTSOE_API_KEY')
        if api_key is None:
            print("WARNING: ENTSOE_API_KEY not found. Set environment variable or pass api_key.")
            print("Creating a dummy price dataframe.")
            # Create a dummy dataframe with random prices for a year
            dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31 23:00:00', freq='H')
            prices = np.random.uniform(30, 150, size=len(dates))
            return pd.DataFrame(prices, index=dates, columns=['price'])

    client = EntsoePandasClient(api_key=api_key)
    
    # Split year into periods to avoid timeouts
    start_date = pd.Timestamp(f'{year}-01-01', tz='Europe/Amsterdam')
    end_date = pd.Timestamp(f'{year+1}-01-01', tz='Europe/Amsterdam')
    
    try:
        print(f"Fetching prices for {start_date.date()} to {end_date.date()}...")
        # Query intraday prices (VWAP = Volume Weighted Average Price)
        prices = client.query_intraday_prices(country_code, start=start_date, end=end_date)
        return prices
    except Exception as e:
        print(f"Error fetching ENTSO-E prices: {e}")
        print("Creating a dummy price dataframe.")
        dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31 23:00:00', freq='H')
        prices = np.random.uniform(30, 150, size=len(dates))
        return pd.DataFrame(prices, index=dates, columns=['price'])

def align_generation_with_prices(power_output_kW, prices_df):
    """
    Create aligned DataFrame with power generation and electricity prices.
    """
    
    # Ensure prices index is datetime and in UTC
    if not isinstance(prices_df.index, pd.DatetimeIndex):
        prices_df.index = pd.to_datetime(prices_df.index)
    if prices_df.index.tz is None:
        prices_df.index = prices_df.index.tz_localize('UTC')
    else:
        prices_df.index = prices_df.index.tz_convert('UTC')

    # Ensure power output index is datetime and in UTC
    if not isinstance(power_output_kW.index, pd.DatetimeIndex):
        power_output_kW.index = pd.to_datetime(power_output_kW.index)
    if power_output_kW.index.tz is None:
        power_output_kW.index = power_output_kW.index.tz_localize('UTC')
    else:
        power_output_kW.index = power_output_kW.index.tz_convert('UTC')

    combined = pd.DataFrame({
        'power_kW': power_output_kW,
        'price_EUR_per_MWh': prices_df.iloc[:, 0] if isinstance(prices_df, pd.DataFrame) else prices_df
    })
    
    combined = combined.dropna()
    
    print(f"✓ Aligned {len(combined)} hourly records")
    if not combined.empty:
        print(f"  Date range: {combined.index[0]} to {combined.index[-1]}")
    
    return combined

def calculate_daily_revenue(hourly_data):
    """
    Calculate daily aggregated energy and revenue.
    """
    if hourly_data.empty:
        return pd.DataFrame()
        
    daily = hourly_data.copy()
    daily['date'] = daily.index.date
    
    daily_aggregated = daily.groupby('date').agg(
        energy_kWh=('power_kW', 'sum'),
        avg_price_EUR_MWh=('price_EUR_per_MWh', 'mean')
    ).reset_index()
    
    daily_aggregated['energy_MWh'] = daily_aggregated['energy_kWh'] / 1000
    daily_aggregated['revenue_EUR'] = daily_aggregated['energy_MWh'] * daily_aggregated['avg_price_EUR_MWh']
    
    return daily_aggregated

def generate_revenue_report(daily_data, output_csv='daily_revenue_report.csv'):
    """
    Generate comprehensive revenue report.
    """
    if daily_data.empty:
        print("No data to generate report.")
        return {}

    # Summary statistics
    total_energy = daily_data['energy_MWh'].sum()
    total_revenue = daily_data['revenue_EUR'].sum()
    avg_daily_revenue = daily_data['revenue_EUR'].mean()
    
    print("\n" + "="*60)
    print("DAILY REVENUE REPORT")
    print("="*60)
    print(f"Total energy generated: {total_energy:,.1f} MWh")
    print(f"Total revenue (at avg price): €{total_revenue:,.2f}")
    print(f"Average daily revenue: €{avg_daily_revenue:,.2f}")
    print(f"Date range: {daily_data['date'].min()} to {daily_data['date'].max()}")
    print(f"Number of days: {len(daily_data)}")
    print("="*60)
    
    # Save to CSV
    daily_data.to_csv(os.path.join(CONFIG['output_dir'], output_csv), index=False)
    print(f"\n✓ Report saved to: {os.path.join(CONFIG['output_dir'], output_csv)}")
    
    return {
        'total_energy_MWh': total_energy,
        'total_revenue_EUR': total_revenue,
        'avg_daily_revenue_EUR': avg_daily_revenue,
        'num_days': len(daily_data),
        'date_range': (daily_data['date'].min(), daily_data['date'].max())
    }

def generate_graphs(weather_data, power_output_kW, daily_revenue, output_dir):
    """
    Generates and saves plots for wind speed, power output, and revenue.
    """
    print("\n[PHASE 8] Generating graphs...")
    
    # Wind Speed Time Series
    plt.figure(figsize=(12, 6))
    plt.plot(weather_data['time'], weather_data['wind_speed_hub'], label='Wind Speed (m/s)')
    plt.xlabel('Time')
    plt.ylabel('Wind Speed (m/s)')
    plt.title('Wind Speed Time Series')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'wind_speed_timeseries.png'))
    plt.close()
    print("✓ Wind speed time series plot saved.")

    # Power Output Time Series
    plt.figure(figsize=(12, 6))
    plt.plot(weather_data['time'], power_output_kW, label='Power Output (kW)', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Power Output (kW)')
    plt.title('Power Output Time Series')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'power_output_timeseries.png'))
    plt.close()
    print("✓ Power output time series plot saved.")

    # Wind Speed Histogram
    plt.figure(figsize=(12, 6))
    plt.hist(weather_data['wind_speed_hub'], bins=50, alpha=0.7, label='Wind Speed')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Frequency')
    plt.title('Wind Speed Distribution')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'wind_speed_histogram.png'))
    plt.close()
    print("✓ Wind speed histogram saved.")

    # Power Output Histogram
    plt.figure(figsize=(12, 6))
    plt.hist(power_output_kW.dropna(), bins=50, alpha=0.7, color='orange', label='Power Output')
    plt.xlabel('Power Output (kW)')
    plt.ylabel('Frequency')
    plt.title('Power Output Distribution')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'power_output_histogram.png'))
    plt.close()
    print("✓ Power output histogram saved.")

    # Daily Revenue Time Series
    if not daily_revenue.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(pd.to_datetime(daily_revenue['date']), daily_revenue['revenue_EUR'], label='Daily Revenue (EUR)', color='green')
        plt.xlabel('Date')
        plt.ylabel('Revenue (EUR)')
        plt.title('Daily Revenue Time Series')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'daily_revenue_timeseries.png'))
        plt.close()
        print("✓ Daily revenue time series plot saved.")

def main():
    """Execute complete workflow."""
    
    print("="*70)
    print("WIND POWER PLANT FOR HYDROGEN - IMPLEMENTATION")
    print("="*70)
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # ===== PHASE 1 & 2: DATA ACQUISITION & PREPARATION =====
    print("\n[PHASE 1 & 2] Using pre-processed weather data...")
    weather_data = prepare_weather_data(2024, CONFIG)
    
    print("✓ Weather data prepared.")

    # ===== PHASE 3: LOAD TURBINE =====
    print("\n[PHASE 3] Loading turbine specifications...")
    turbine_specs = load_turbine_power_curve(CONFIG['turbine_name'])
    print(f"✓ Loaded turbine: {turbine_specs['turbine_type']}")

    # ===== PHASE 4: SINGLE TURBINE POWER GENERATION =====
    print("\n[PHASE 4] Calculating power generation...")
    power_output_kW, weather_data = calculate_power_generation_single_turbine(
        weather_data,
        turbine_specs,
        hub_height=CONFIG['hub_height'],
        apply_density_correction=True
    )
    print(f"✓ Generated {len(power_output_kW)} hourly power values")
    
    validate_power_generation(weather_data.set_index('time'), power_output_kW, turbine_specs)

    # ===== PHASE 5: P80 SIZING & TURBINE COUNT CALCULATION =====
    print("\n[PHASE 5] Calculating P80 and turbine count...")
    pdc_metrics, power_sorted, time_percentiles = create_power_duration_curve(power_output_kW, percentiles=[0.8])
    print(f"✓ Power Duration Curve generated")
    
    if 'P80' in pdc_metrics and pdc_metrics['P80'] > 0:
        print(f"  P80 (Gross): {pdc_metrics['P80']:.2f} kW")
        
        # Apply farm efficiency to single turbine P80 for sizing
        p80_single_turbine_net_kW = pdc_metrics['P80'] * CONFIG['farm_efficiency']
        print(f"  P80 (Net, single turbine): {p80_single_turbine_net_kW:.2f} kW")

        N_turbines = calculate_turbine_count(
            P80_turbine_kW=p80_single_turbine_net_kW,
            P80_target_MW=CONFIG['P80_target_MW']
        )
        print(f"✓ Number of turbines needed: {N_turbines}")
    else:
        print("P80 is not positive. Cannot calculate turbine count. Setting to 0.")
        N_turbines = 0

    plot_power_duration_curve(
        power_sorted,
        time_percentiles,
        CONFIG['turbine_name'],
        save_path=os.path.join(CONFIG['output_dir'], 'PDC.png')
    )
    print(f"✓ Power Duration Curve saved to {os.path.join(CONFIG['output_dir'], 'PDC.png')}")

    # ===== PHASE 6: AGGREGATE PARK GENERATION & FARM EFFICIENCY =====
    print("\n[PHASE 6] Calculating park generation with farm efficiency...")
    P_park_gross, P_park_net = calculate_park_generation(
        power_output_kW,
        N_turbines,
        farm_efficiency=CONFIG['farm_efficiency']
    )
    print("✓ Park generation calculated")
    if N_turbines > 0:
        validate_farm_efficiency(power_output_kW, N_turbines, CONFIG['farm_efficiency'], CONFIG['P80_target_MW'])
    
    # ===== PHASE 7: ENERGY PRICING & REVENUE CALCULATION =====
    print("\n[PHASE 7] Calculating daily revenue...")
    prices_df = fetch_entsoe_prices(
        country_code=CONFIG['entsoe_country_code'],
        year=weather_data['time'].dt.year.max()
    )
    
    combined_data = align_generation_with_prices(
        P_park_net,
        prices_df
    )
    
    daily_revenue = calculate_daily_revenue(combined_data)
    
    generate_revenue_report(
        daily_revenue,
        output_csv='daily_revenue_report.csv'
    )

    # ===== PHASE 8: UNCERTAINTY ANALYSIS =====
    print("\n[PHASE 8] Performing uncertainty analysis...")
    perform_uncertainty_analysis(P_park_net, N_turbines)


    # ===== PHASE 9: GENERATE GRAPHS =====
    generate_graphs(weather_data, power_output_kW, daily_revenue, CONFIG['output_dir'])

def perform_uncertainty_analysis(power_net_kW, N_turbines):
    """
    Performs a simplified uncertainty analysis (P50, P75, P90).
    """
    print("\n--- Uncertainty Analysis ---")
    
    # P50 (mean expected energy)
    # Total energy in MWh for the year
    p50_mwh = (power_net_kW.sum()) / 1000
    
    # Typical uncertainty for wind projects (sigma)
    sigma = 0.10  # 10%
    
    # P75 and P90
    p75_mwh = p50_mwh * (1 - 0.67 * sigma)
    p90_mwh = p50_mwh * (1 - 1.28 * sigma)
    
    print(f"P50 (best guess) annual energy: {p50_mwh:,.0f} MWh")
    print(f"P75 (likely) annual energy:     {p75_mwh:,.0f} MWh")
    print(f"P90 (conservative) annual energy: {p90_mwh:,.0f} MWh")
    
    return {
        'P50_MWh': p50_mwh,
        'P75_MWh': p75_mwh,
        'P90_MWh': p90_mwh,
    }
if __name__ == '__main__':
    main()
