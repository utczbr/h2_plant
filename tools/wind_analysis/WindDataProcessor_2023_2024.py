import xarray as xr
import pandas as pd
import numpy as np
import os

# --- PATHS AND DEFINITIONS ---
# Path to the NetCDF file containing wind data (u and v components).
NETCDF_PATH = r"data.nc"
# Path to the CSV file containing air density data ('rhoa').
RHO_CSV_PATH = r"ninja_weather_density_air.csv"
# Target latitude for data extraction.
TARGET_LATITUDE = 53.0
# Target longitude for data extraction.
TARGET_LONGITUDE = 4.0

class WindDataProcessor:
    # Initialize the processor with file paths and target coordinates.
    def __init__(self, netcdf_path: str, csv_path: str, latitude: float, longitude: float):
        self.netcdf_path = netcdf_path
        self.csv_path = csv_path
        self.latitude = latitude
        self.longitude = longitude
        # DataFrames to store final processed data for 2023 and 2024.
        self.df_2023_final = pd.DataFrame()
        self.df_2024_final = pd.DataFrame()
        
    # Loads NetCDF wind data and filters it for the nearest point to the target lat/lon.
    def _load_and_filter_data(self) -> xr.Dataset:
        # Check if NetCDF file exists.
        if not os.path.exists(self.netcdf_path):
            raise FileNotFoundError(f"ERROR: NetCDF file not found at: {self.netcdf_path}")
            
        with xr.open_dataset(self.netcdf_path) as ds:
            # Select the data for the nearest point to the target coordinates.
            ds_point = ds.sel(latitude=self.latitude, longitude=self.longitude, method='nearest')
            # Load data into memory.
            return ds_point.load()

    # Prepares the air density DataFrame (rhoa) for a specific target year.
    def _prepare_rho_dataframe(self, target_year: int) -> pd.DataFrame:
        # Check if CSV file exists.
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"ERROR: CSV file not found at: {self.csv_path}")

        # Read the CSV, skipping initial metadata rows.
        df_rho = pd.read_csv(self.csv_path, skiprows=4)
        # Rename columns to meaningful names.
        df_rho.columns = ['time', 'local_time', 'rhoa']

        # Convert 'time' column to datetime objects.
        df_rho['time'] = pd.to_datetime(df_rho['time'], format='%Y-%m-%d %H:%M')
        # Replace the year in the timestamp with the target year for alignment.
        df_rho['time'] = df_rho['time'].apply(lambda dt: dt.replace(year=target_year))
        
        # Ensure 'time' is explicitly a datetime type (redundant but safe).
        df_rho['time'] = pd.to_datetime(df_rho['time'])
        
        # Drop the 'local_time' column and set 'time' as the index for merging.
        df_rho = df_rho.drop(columns=['local_time'], errors='ignore').set_index('time')
        # Ensure the index name is 'time'.
        df_rho.index.name = 'time'
        return df_rho

    # NEW FUNCTION: Imputation using Backfill
    def _impute_bfill(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills NaN values in 'rhoa' with the next non-NaN value (backfill)."""
        df['rhoa'] = df['rhoa'].fillna(method='bfill') 
        return df

    # Main method to process wind and density data and perform the merge.
    def process_data(self):
        # --- PHASE 1: NETCDF PROCESSING (WIND) ---
        # Load and filter wind data.
        ds_point = self._load_and_filter_data()
        # Calculate wind speed from u (zonal) and v (meridional) components.
        wind_speed_data = np.sqrt(ds_point['u100']**2 + ds_point['v100']**2)
        
        # Convert the xarray DataArray to a Pandas DataFrame.
        df_hourly = wind_speed_data.to_dataframe(name="Wind Speed Hourly (m/s)")
        
        # CORRECT FINAL STEP: Drop extra index levels (like lat/lon) to avoid merge errors.
        # Identify index names that are not 'time' and not None.
        index_names_to_drop = [name for name in df_hourly.index.names if name not in ['time', None]]
        
        if index_names_to_drop:
            # Drop the identified index levels.
            df_hourly = df_hourly.droplevel(index_names_to_drop)
        
        # Reaffirm the name of the main index.
        df_hourly.index.name = 'time'
        
        # Separate data by year.
        df_2023 = df_hourly.loc['2023'].copy()
        df_2024 = df_hourly.loc['2024'].copy()
        
        # --- PHASE 2: RHOA PREPARATION AND MERGE ---
        
        # Merge 2023 Data
        df_rho_2023 = self._prepare_rho_dataframe(2023)
        self.df_2023_final = df_2023.merge(
            df_rho_2023,
            left_index=True,  
            right_index=True, # Merge based on the index ('time')
            how='left'
        )
        
        # Apply backfill imputation and reset the 'time' index to a column.
        self.df_2023_final = self._impute_bfill(self.df_2023_final)
        self.df_2023_final = self.df_2023_final.reset_index(names=['time'])

        # Merge 2024 Data
        df_rho_2024 = self._prepare_rho_dataframe(2024)
        self.df_2024_final = df_2024.merge(
            df_rho_2024,
            left_index=True,  
            right_index=True, # Merge based on the index ('time')
            how='left'
        )
        
        # Apply backfill imputation and reset the 'time' index to a column.
        self.df_2024_final = self._impute_bfill(self.df_2024_final)
        self.df_2024_final = self.df_2024_final.reset_index(names=['time'])


if __name__ == "__main__":
    try:
        # Create an instance of the data processor.
        processor = WindDataProcessor(NETCDF_PATH, RHO_CSV_PATH, TARGET_LATITUDE, TARGET_LONGITUDE)
        
        # Execute the data processing workflow.
        processor.process_data()
        
        # Combine the 2023 and 2024 dataframes
        combined_df = pd.concat([processor.df_2023_final, processor.df_2024_final], ignore_index=True)
        
        # Save the combined dataframe to a CSV file
        output_filename = 'weather_data_2023-2024.csv'
        combined_df.to_csv(output_filename, index=False)
        
        print(f"✓ Successfully processed data and saved to {output_filename}")
        
    # Handle specific file not found error.
    except FileNotFoundError as fe:
        print(f"\n❌ FILE ERROR: {fe}")
    # Handle other general exceptions.
    except Exception as e:
        print(f"\n❌ ERROR: An issue occurred during script execution. Details: {e}")