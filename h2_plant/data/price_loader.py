import pandas as pd
import os
import logging
from typing import Tuple
import numpy as np

logger = logging.getLogger(__name__)

class EnergyPriceLoader:
    """
    Loads energy prices and wind data from CSV files.
    """
    def __init__(self, scenarios_dir: str):
        self.scenarios_dir = scenarios_dir

    def load_data(self, price_file: str, wind_file: str, duration_hours: int, timestep_hours: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads and resamples data to match simulation timestep.
        """
        try:
            # Construct paths
            p_path = os.path.join(self.scenarios_dir, price_file)
            w_path = os.path.join(self.scenarios_dir, wind_file)
            
            # Load Prices
            df_prices = pd.read_csv(p_path, index_col=0)
            df_prices.index = pd.to_datetime(df_prices.index, utc=True).tz_localize(None)
            df_prices.sort_index(inplace=True)
            
            # Find price column
            col_price = [c for c in df_prices.columns if 'price' in c.lower() or 'eur' in c.lower()][0]
            if df_prices[col_price].dtype == 'object':
                df_prices[col_price] = df_prices[col_price].astype(str).str.replace(',', '.')
            df_prices[col_price] = pd.to_numeric(df_prices[col_price], errors='coerce')

            # Load Wind
            df_wind = pd.read_csv(w_path, sep=',', decimal='.')
            if len(df_wind.columns) < 2:
                df_wind = pd.read_csv(w_path, sep=';', decimal=',')
                
            if 'data_hora' not in df_wind.columns:
                 df_wind.reset_index(inplace=True)
                 for col in df_wind.columns:
                     if 'data' in col.lower() or 'date' in col.lower():
                         df_wind.rename(columns={col: 'data_hora'}, inplace=True)
                         break
            
            df_wind['data_hora'] = pd.to_datetime(df_wind['data_hora'])
            df_wind = df_wind.set_index('data_hora').sort_index()
            
            col_wind = 'potencia_2_turbinas_MW'
            if df_wind[col_wind].dtype == 'object':
                df_wind[col_wind] = df_wind[col_wind].astype(str).str.replace(',', '.')
            df_wind[col_wind] = pd.to_numeric(df_wind[col_wind], errors='coerce')

            # Resample
            start_time = df_prices.index[0]
            end_time = start_time + pd.Timedelta(hours=duration_hours)
            
            freq = f"{int(timestep_hours * 60)}min"
            idx = pd.date_range(start=start_time, end=end_time, freq=freq, inclusive='left')
            
            prices = df_prices[col_price].reindex(idx).ffill().bfill().values
            wind = df_wind[col_wind].reindex(idx).interpolate(method='linear').ffill().bfill().values
            
            return prices, wind

        except Exception as e:
            logger.error(f"Error loading external data: {e}")
            raise
