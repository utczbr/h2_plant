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
        
        Handles different data resolutions:
        - Prices: typically 15-minute resolution
        - Wind: typically hourly resolution  
        - Output: resampled to simulation timestep (typically 1-minute)
        """
        try:
            # Construct paths
            p_path = os.path.join(self.scenarios_dir, price_file)
            w_path = os.path.join(self.scenarios_dir, wind_file)
            
            # Load Prices
            df_prices = pd.read_csv(p_path, index_col=0)
            # Parse timezone-aware timestamps, keeping LOCAL time from the string
            # The timestamps like "2024-01-01 00:00:00+01:00" should become "2024-01-01 00:00:00"
            # We do this by parsing as UTC, converting to local tz, then stripping tz
            df_prices.index = pd.to_datetime(df_prices.index, utc=True)
            # Convert to Europe/Amsterdam (the timezone in the file) then drop tz
            df_prices.index = df_prices.index.tz_convert('Europe/Amsterdam').tz_localize(None)
            # Remove duplicates caused by DST changes (keep first occurrence)
            df_prices = df_prices[~df_prices.index.duplicated(keep='first')]
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
            
            # Parse wind timestamps as naive (no timezone)
            df_wind['data_hora'] = pd.to_datetime(df_wind['data_hora'])
            # Strip timezone if present
            if df_wind['data_hora'].dt.tz is not None:
                df_wind['data_hora'] = df_wind['data_hora'].dt.tz_localize(None)
            df_wind = df_wind.set_index('data_hora').sort_index()
            
            col_wind = 'potencia_2_turbinas_MW'
            if col_wind not in df_wind.columns:
                # Fallback: try to find power column
                for c in df_wind.columns:
                    if 'potencia' in c.lower() or 'power' in c.lower() or 'mw' in c.lower():
                        col_wind = c
                        break
            if df_wind[col_wind].dtype == 'object':
                df_wind[col_wind] = df_wind[col_wind].astype(str).str.replace(',', '.')
            df_wind[col_wind] = pd.to_numeric(df_wind[col_wind], errors='coerce')

            # Use wind data start time (naive) as common reference
            # This ensures wind and simulation time indexes align
            start_time = df_wind.index[0]
            end_time = start_time + pd.Timedelta(hours=duration_hours)
            
            # === DATA WRAPPING LOGIC ===
            # If requested duration exceeds available data, tile the data annually
            def extend_dataframe(df, target_end_time):
                if df.empty: return df
                
                current_end = df.index[-1]
                if target_end_time <= current_end:
                    return df
                
                frames = [df]
                # Calculate how many years we need to cover
                # Using a while loop is robust against short files or long durations
                temp_end = current_end
                year_offset = 1
                
                while temp_end < target_end_time: # Corrected variable name in loop
                    # Copy original dataframe
                    df_copy = df.copy()
                    # Shift index by N years
                    # DateOffset accounts for leap years correctly (maintains month-day alignment)
                    df_copy.index = df_copy.index + pd.DateOffset(years=year_offset)
                    frames.append(df_copy)
                    
                    temp_end = df_copy.index[-1]
                    year_offset += 1
                
                # Verify we aren't creating duplicates at boundaries (e.g. if original ends Dec 31 23:00 and copy starts Jan 01 00:00, it's fine)
                # But if original covers full year including Jan 1 00:00 next year, we might overlap.
                # Simplest safe approach: concat and drop duplicates on index
                extended_df = pd.concat(frames)
                extended_df = extended_df[~extended_df.index.duplicated(keep='first')]
                return extended_df.sort_index()

            # Extend source dataframes if necessary
            df_wind = extend_dataframe(df_wind, end_time)
            
            # Align prices to wind before extension (critical for offset logic)
            price_start = df_prices.index[0]
            wind_start = df_wind.index[0] 
            # Note: wind_start is now the ORIGINAL start (Extend function didn't shift start)
            
            if price_start.date() != wind_start.date():
                 offset = wind_start.normalize() - price_start.normalize()
                 df_prices.index = df_prices.index + offset
            
            df_prices = extend_dataframe(df_prices, end_time)

            freq = f"{int(round(timestep_hours * 60))}min"
            idx = pd.date_range(start=start_time, end=end_time, freq=freq, inclusive='left')
            
            # Resample to simulation timestep (forward-fill hourly data)
            # Reindex handles truncation if we extended too far
            wind = df_wind[col_wind].reindex(idx, method='ffill').ffill().bfill().values
            prices = df_prices[col_price].reindex(idx, method='ffill').ffill().bfill().values
            
            logger.info(f"Loaded {len(prices)} price points, {len(wind)} wind points "
                       f"(range: {wind.min():.2f}-{wind.max():.2f} MW)")
            
            return prices, wind

        except Exception as e:
            logger.error(f"Error loading external data: {e}")
            raise
