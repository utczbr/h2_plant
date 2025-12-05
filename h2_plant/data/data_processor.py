import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging
from datetime import datetime

# External libraries
try:
    from windpowerlib import WindTurbine, ModelChain
    from entsoe import EntsoePandasClient
except ImportError:
    logging.warning("windpowerlib or entsoe-py not installed. DataProcessor will fail.")

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Processes environmental data for the H2 Plant simulation.
    - Calculates wind power availability based on weather data.
    - Fetches electricity prices from ENTSO-E.
    - Generates a unified CSV file for the simulation.
    """
    
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent
        self.weather_file = Path("/home/stuart/Documentos/Planta Hidrogenio/Energy_by_Wind/weather_data_2023-2024.csv")
        self.api_key = '21161037-2eab-4624-9c2c-04408915773a'
        
        # Turbine Configuration (Vestas V117/3600)
        self.turbine_specs = {
            'turbine_type': 'Vestas V117/3600',
            'hub_height': 105,
        }
        self.nominal_power_mw = 3.6
        
    def process_environment_data(self, year: int = 2024, force_refresh: bool = False) -> str:
        """
        Main method to generate the environment data file.
        Returns the path to the generated file.
        """
        output_file = self.output_dir / f"environment_data_{year}.csv"
        
        if output_file.exists() and not force_refresh:
            logger.info(f"Environment data for {year} already exists at {output_file}")
            return str(output_file)
            
        logger.info(f"Generating environment data for {year}...")
        
        # 1. Process Wind Data
        wind_df = self._process_wind_data(year)
        
        # 2. Fetch Price Data
        price_df = self._fetch_price_data(year)
        
        # 3. Merge and Save
        merged_df = self._merge_data(wind_df, price_df)
        merged_df.to_csv(output_file)
        logger.info(f"Saved environment data to {output_file}")
        
        # 4. Save simple price file for PlantBuilder (EnergyPriceTracker)
        # PlantBuilder uses np.loadtxt which expects simple array
        # EnergyPriceTracker expects prices in EUR/MWh (it converts to EUR/kWh internally)
        simple_price_file = self.output_dir / f"prices_{year}.csv"
        # Save price_eur_mwh (NOT price_eur_kwh)
        np.savetxt(simple_price_file, merged_df['price_eur_mwh'].values, delimiter=',')
        logger.info(f"Saved simple price file to {simple_price_file}")
        
        return str(output_file)
        
    def _process_wind_data(self, year: int) -> pd.DataFrame:
        """Load weather data and calculate wind power coefficient."""
        logger.info("Processing wind data...")
        
        if not self.weather_file.exists():
            raise FileNotFoundError(f"Weather file not found: {self.weather_file}")
            
        # Load and filter
        df = pd.read_csv(self.weather_file)
        df['time'] = pd.to_datetime(df['time'])
        df = df[df['time'].dt.year == year].copy()
        df.set_index('time', inplace=True)
        
        if df.empty:
            raise ValueError(f"No weather data found for year {year}")

        # Prepare for windpowerlib
        # Map columns to windpowerlib format (MultiIndex)
        # We need: wind_speed, temperature, pressure, roughness_length (optional)
        
        # Logic adapted from wind_power_plant_final.py
        hub_height = self.turbine_specs['hub_height']
        
        # Calculate air density if missing
        if 'rhoa' in df.columns:
            df['density'] = df['rhoa']
        else:
            df['density'] = 1.225 # Fallback
            
        # Wind speed at hub height (simple shear if needed, but windpowerlib handles it if we give height)
        # The input file has 'Wind Speed Hourly (m/s)'. Let's assume it's at 10m or similar.
        # wind_power_plant_final.py assumes 10m and does shear.
        # windpowerlib can do this if we specify height.
        
        # Let's construct the model input dataframe
        model_input = pd.DataFrame(index=df.index)
        
        # Assume input wind speed is at 10m
        model_input[('wind_speed', 10)] = df['Wind Speed Hourly (m/s)']
        model_input[('temperature', 2)] = 288.15 # Default if missing, or use column if exists
        model_input[('density', 10)] = df['density'] # Assume density at surface/10m
        model_input[('pressure', 0)] = 101325.0
        model_input[('roughness_length', 0)] = 0.15 # Onshore default
        
        model_input.columns = pd.MultiIndex.from_tuples(model_input.columns)
        
        # Initialize Turbine
        # Note: 'Vestas V117/3600' might not be in windpowerlib database by default name, 
        # but wind_power_plant_final.py used a custom power curve loader.
        # For simplicity, we will use a generic turbine or try to define it.
        # wind_power_plant_final.py defined the power curve manually.
        # Let's replicate that manual definition to be safe.
        
        turbine_data = {
            'turbine_type': 'Vestas V117/3600',
            'hub_height': hub_height,
            'rotor_diameter': 117,
            'nominal_power': self.nominal_power_mw * 1e6, # Watts
            'power_curve': self._get_power_curve()
        }
        
        turbine = WindTurbine(**turbine_data)
        
        # Run Model
        mc = ModelChain(turbine, density_correction=True)
        mc.run_model(model_input)
        
        # Calculate Coefficient (0.0 to 1.0)
        power_output = mc.power_output # in Watts
        power_coeff = power_output / (self.nominal_power_mw * 1e6)
        
        result = pd.DataFrame({
            'wind_power_coefficient': power_coeff,
            'wind_speed': df['Wind Speed Hourly (m/s)'], # Raw wind speed at 10m (or whatever input was)
            'air_density': df['density']
        }, index=df.index)
        
        return result

    def _get_power_curve(self) -> pd.DataFrame:
        """Define power curve for Vestas V117/3600."""
        # Logic from wind_power_plant_final.py
        wind_speeds = np.arange(0, 30, 0.5)
        power_values = []
        
        cut_in = 3.0
        rated = 10.0
        cut_out = 25.0
        nominal = self.nominal_power_mw * 1e6
        
        for v in wind_speeds:
            if v < cut_in or v >= cut_out:
                p = 0.0
            elif v >= rated:
                p = nominal
            else:
                # Linear ramp (simplified)
                p = nominal * (v - cut_in) / (rated - cut_in)
            power_values.append(p)
            
        return pd.DataFrame({
            'wind_speed': wind_speeds,
            'value': power_values
        })

    def _fetch_price_data(self, year: int) -> pd.DataFrame:
        """Fetch intraday prices from ENTSO-E."""
        logger.info("Fetching price data from ENTSO-E...")
        
        client = EntsoePandasClient(api_key=self.api_key)
        country_code = 'NL'
        
        start = pd.Timestamp(f'{year}-01-01', tz='Europe/Amsterdam')
        end = pd.Timestamp(f'{year}-12-31 23:00', tz='Europe/Amsterdam')
        
        try:
            # Fetch in chunks to avoid timeout
            # First half
            end_h1 = pd.Timestamp(f'{year}-06-30', tz='Europe/Amsterdam')
            df1 = client.query_day_ahead_prices(country_code, start=start, end=end_h1)
            
            # Second half
            df2 = client.query_day_ahead_prices(country_code, start=end_h1, end=end)
            
            df = pd.concat([df1, df2])
            
            # Resample to hourly if needed (day-ahead is usually hourly)
            df = df.resample('h').mean()
            
            # Handle timezones - convert to naive UTC or keep consistent
            df.index = df.index.tz_convert(None) 
            
            return pd.DataFrame({'price_eur_mwh': df})
            
        except Exception as e:
            logger.error(f"Error fetching prices: {e}")
            # Fallback: Generate dummy data
            logger.warning("Using fallback price data.")
            dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31 23:00', freq='H')
            return pd.DataFrame({
                'price_eur_mwh': np.random.uniform(30, 150, size=len(dates))
            }, index=dates)

    def _merge_data(self, wind_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        """Merge wind and price data."""
        # Align indices
        combined = wind_df.join(price_df, how='outer')
        
        # Fill missing
        combined['wind_power_coefficient'] = combined['wind_power_coefficient'].fillna(0.0)
        combined['wind_speed'] = combined['wind_speed'].fillna(0.0)
        combined['air_density'] = combined['air_density'].fillna(1.225)
        
        combined['price_eur_mwh'] = combined['price_eur_mwh'].fillna(method='ffill').fillna(50.0)
        
        # Calculate price in EUR/kWh
        combined['price_eur_kwh'] = combined['price_eur_mwh'] / 1000.0
        
        return combined

if __name__ == "__main__":
    # Test run
    logging.basicConfig(level=logging.INFO)
    processor = DataProcessor()
    processor.process_environment_data(2024, force_refresh=True)
