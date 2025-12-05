"""
Environment Manager for external environmental data.

Manages time-series data for:
- Wind power availability
- Energy prices
- Weather conditions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from h2_plant.core.component import Component

class EnvironmentManager(Component):
    """
    Manages external environmental data for simulation.
    Provides time-series data for wind, energy prices, and weather.
    """
    
    def __init__(self, 
                 wind_data_path: Optional[str] = None,
                 price_data_path: Optional[str] = None,
                 use_default_data: bool = True,
                 installed_wind_capacity_mw: float = 20.0):
        super().__init__()
        
        # Default paths (relative to h2_plant/data/)
        if wind_data_path is None and use_default_data:
            wind_data_path = str(Path(__file__).parent.parent.parent / 'h2_plant' / 'data' / 'wind_data.csv')
        if price_data_path is None and use_default_data:
            price_data_path = str(Path(__file__).parent.parent.parent / 'h2_plant' / 'data' / 'EnergyPriceAverage2023-24.csv')
        
        self.wind_data_path = wind_data_path
        self.price_data_path = price_data_path
        self.installed_wind_capacity_mw = installed_wind_capacity_mw
        
        # Data storage
        self.wind_data: Optional[pd.DataFrame] = None
        self.price_data: Optional[pd.DataFrame] = None
        
        # State
        self.current_timestep = 0
        self.current_minute = 0  # NEW: Track minutes for arbitration logic
        self.wind_power_coefficient = 0.5
        self.air_density_kg_m3 = 1.225
        self.energy_price_eur_kwh = 0.06
        
        # Resolution tracking
        self.price_resolution_minutes = 15  # Default: 15-minute intervals
        self.wind_resolution_hours = 1      # Default: hourly
        
        # Direct power input support
        self.use_direct_power = False
        self.current_power_mw = 0.0
        
    def initialize(self, dt: float, registry) -> None:
        """Load environmental data files."""
        super().initialize(dt, registry)
        self._load_data()
    
    def _load_data(self) -> None:
        """Load wind and price data, generating it if necessary."""
        try:
            # 1. Load specific files FIRST (higher priority than unified)
            
            # Wind Data
            if self.wind_data_path and Path(self.wind_data_path).exists():
                try:
                    # Try reading with different separators/decimals
                    try:
                        self.wind_data = pd.read_csv(self.wind_data_path, sep=';', decimal=',')
                        if len(self.wind_data.columns) < 2: # Fallback if separator was wrong
                             self.wind_data = pd.read_csv(self.wind_data_path, sep=',', decimal='.')
                    except:
                        self.wind_data = pd.read_csv(self.wind_data_path, sep=',', decimal='.')

                    print(f"EnvironmentManager: Loaded wind data from {self.wind_data_path} ({len(self.wind_data)} rows)")
                    
                    # Normalize columns (strip whitespace)
                    self.wind_data.columns = self.wind_data.columns.str.strip()
                    
                    # Check for direct power input (aliases)
                    power_aliases = ['Power_MW', 'potencia_2_turbinas_MW', 'ActivePower', 'Power']
                    found_alias = None
                    for alias in power_aliases:
                        if alias in self.wind_data.columns:
                            found_alias = alias
                            break
                    
                    if found_alias:
                        self.use_direct_power = True
                        # Standardize column name
                        self.wind_data['Power_MW'] = pd.to_numeric(self.wind_data[found_alias].astype(str).str.replace(',', '.'), errors='coerce')
                        print(f"EnvironmentManager: Detected '{found_alias}' column - using direct power input mode")
                    else:
                        print(f"DEBUG EnvironmentManager: Power column NOT found in {self.wind_data.columns.tolist()}")
                        
                except Exception as e:
                    print(f"EnvironmentManager: Error reading wind file: {e}")
            
            # Price Data
            if self.price_data_path and Path(self.price_data_path).exists():
                try:
                    # Robust read for prices
                    df_precos = pd.read_csv(self.price_data_path, index_col=0)
                    
                    # Identify price column
                    col_valor_preco = [c for c in df_precos.columns if 'price' in c.lower() or 'eur' in c.lower()]
                    if col_valor_preco:
                        col_name = col_valor_preco[0]
                        # Clean data
                        if df_precos[col_name].dtype == 'object':
                            df_precos[col_name] = df_precos[col_name].astype(str).str.replace(',', '.')
                        
                        df_precos[col_name] = pd.to_numeric(df_precos[col_name], errors='coerce')
                        
                        # Store as standard DataFrame
                        self.price_data = df_precos
                        # Ensure 'price_eur_mwh' exists
                        self.price_data['price_eur_mwh'] = df_precos[col_name]
                        
                        print(f"EnvironmentManager: Loaded price data from {self.price_data_path} ({len(self.price_data)} rows)")
                    else:
                         # Fallback for headerless or simple files
                         self.price_data = pd.read_csv(self.price_data_path)
                         print(f"EnvironmentManager: Loaded raw price data (columns: {self.price_data.columns.tolist()})")

                except Exception as e:
                    print(f"EnvironmentManager: Error reading price file: {e}")
            
            # 2. Check for unified environment file (FALLBACK ONLY if not loaded above)
            if self.wind_data is None or self.price_data is None:
                data_dir = Path(__file__).parent.parent.parent / 'data'
                env_file = data_dir / 'environment_data_2024.csv'
                
                if env_file.exists():
                    try:
                        self.unified_data = pd.read_csv(env_file)
                        if 'time' in self.unified_data.columns:
                             self.unified_data['time'] = pd.to_datetime(self.unified_data['time'])
                        
                        print(f"EnvironmentManager: Loaded unified data from {env_file}")
                        
                        if self.wind_data is None:
                            self.wind_data = self.unified_data
                        if self.price_data is None:
                            self.price_data = self.unified_data
                    except Exception as e:
                        print(f"EnvironmentManager: Error loading unified file: {e}")
            
            # 3. Generate defaults if still nothing loaded
            if self.wind_data is None:
                print(f"EnvironmentManager: No wind data found, using defaults")
                self._create_default_wind_data()
                
            if self.price_data is None:
                print(f"EnvironmentManager: No price data found, using defaults")
                self._create_default_price_data()
            
            # 3. Post-processing and Validation
            # Ensure price_data has 'price_eur_kwh'
            if self.price_data is not None:
                if 'price_eur_kwh' not in self.price_data.columns:
                    if 'price_eur_mwh' in self.price_data.columns:
                        self.price_data['price_eur_kwh'] = self.price_data['price_eur_mwh'] / 1000.0
                    elif 'price' in self.price_data.columns:
                         # Assume EUR/MWh if not specified? Or EUR/kWh?
                         # Safe bet: check magnitude. If > 10, likely MWh.
                         mean_price = self.price_data['price'].mean()
                         if mean_price > 10:
                             self.price_data['price_eur_kwh'] = self.price_data['price'] / 1000.0
                         else:
                             self.price_data['price_eur_kwh'] = self.price_data['price']
                    else:
                        # Fallback if we can't identify column
                        col0 = self.price_data.columns[0]
                        if self.price_data[col0].dtype in [float, int]:
                            if self.price_data[col0].mean() > 10:
                                self.price_data['price_eur_kwh'] = self.price_data[col0] / 1000.0
                            else:
                                self.price_data['price_eur_kwh'] = self.price_data[col0]

            # Detect resolution
            self.price_resolution_factor = 1.0
            if self.price_data is not None:
                rows = len(self.price_data)
                if rows > 8800: # Likely > hourly
                    self.price_resolution_factor = rows / 8760.0
                    print(f"EnvironmentManager: Detected high resolution price data ({rows} rows). Factor: {self.price_resolution_factor:.2f}")

        except Exception as e:
            print(f"EnvironmentManager: Error loading data ({e}), using defaults")
            self._create_default_wind_data()
            self._create_default_price_data()
    
    def _create_default_wind_data(self) -> None:
        """Create default wind data if file not available."""
        # 8760 hours with varying wind coefficient (0.0 to 1.0)
        hours = 8760
        wind_coeff = 0.3 + 0.4 * np.sin(np.arange(hours) * 2 * np.pi / 24)  # Daily cycle
        self.wind_data = pd.DataFrame({
            'hour': np.arange(hours),
            'wind_power_coefficient': np.clip(wind_coeff, 0.0, 1.0),
            'air_density': 1.225
        })
    
    def _create_default_price_data(self) -> None:
        """Create default price data if file not available."""
        # 8760 hours with varying prices (day/night pattern)
        hours = 8760
        base_price = 0.06  # EUR/kWh
        # Higher during day (6am-10pm), lower at night
        hour_of_day = np.arange(hours) % 24
        price_multiplier = np.where((hour_of_day >= 6) & (hour_of_day < 22), 1.3, 0.7)
        
        self.price_data = pd.DataFrame({
            'hour': np.arange(hours),
            'price_eur_kwh': base_price * price_multiplier
        })
        self.price_resolution_factor = 1.0
    
    def step(self, t: float) -> None:
        """
        Update environmental conditions for current timestep.
        
        Supports both hourly and minute-level timesteps.
        For minute-level: handles 15-minute price slots and hourly wind data.
        """
        super().step(t)
        
        # Convert time (hours) to minutes
        minute_absolute = int(t * 60)  # Total minutes since start
        self.current_minute = minute_absolute
        
        # Calculate hour for wind data (hourly resolution)
        hour_index = int(t) % 8760
        self.current_timestep = hour_index
        
        # Update wind data (hourly resolution)
        # Note: If we have minute-level power data, we should handle it here.
        # For now, assuming wind_data is hourly or we map minute to row index if resolution matches.
        # But self.wind_resolution_hours is 1.
        
        if self.wind_data is not None:
            # Use pre-converted numpy arrays if available (Optimization)
            if not hasattr(self, '_wind_coeffs'):
                 self._wind_coeffs = self.wind_data['wind_power_coefficient'].values if 'wind_power_coefficient' in self.wind_data else np.full(len(self.wind_data), 0.5)
                 self._wind_power_mw = self.wind_data['Power_MW'].values if 'Power_MW' in self.wind_data else np.zeros(len(self.wind_data))
                 self._wind_air_density = self.wind_data['air_density'].values if 'air_density' in self.wind_data else np.full(len(self.wind_data), 1.225)
                 self._wind_rows = len(self.wind_data)

            wind_rows = self._wind_rows
            
            # Index logic
            if self.use_direct_power and wind_rows >= 525600: 
                 wind_idx = minute_absolute % wind_rows
            elif self.use_direct_power and (wind_rows == 480 or wind_rows == 10080): 
                 wind_idx = min(minute_absolute, wind_rows - 1)
            else:
                 wind_idx = min(hour_index, wind_rows - 1)
            
            # Fast access via numpy
            if self.use_direct_power:
                self.current_power_mw = self._wind_power_mw[wind_idx]
                self.wind_power_coefficient = self.current_power_mw / self.installed_wind_capacity_mw
            else:
                self.wind_power_coefficient = self._wind_coeffs[wind_idx]
            
            self.air_density_kg_m3 = self._wind_air_density[wind_idx]
        
        # Update price data (15-minute resolution)
        if self.price_data is not None:
            if not hasattr(self, '_price_values'):
                 self._price_values = self.price_data['price_eur_kwh'].values if 'price_eur_kwh' in self.price_data else np.full(len(self.price_data), 0.06)
                 self._price_rows = len(self.price_data)

            minute_of_hour = minute_absolute % 60
            price_slot = minute_of_hour // 15 
            price_index = (hour_index * 4) + price_slot
            
            price_idx = price_index % self._price_rows
            
            self.energy_price_eur_kwh = self._price_values[price_idx]
    
    def get_wind_power_availability(self, installed_capacity_kw: float) -> float:
        """
        Calculate available wind power based on current conditions.
        
        Args:
            installed_capacity_kw: Installed wind turbine capacity in kW
            
        Returns:
            Available power in kW
        """
        if self.use_direct_power:
            return self.current_power_mw * 1000.0 # Convert MW to kW
            
        return installed_capacity_kw * self.wind_power_coefficient
    
    def get_current_energy_price(self) -> float:
        """Get current electricity price in EUR/kWh."""
        return self.energy_price_eur_kwh
    
    def get_current_conditions(self) -> Dict[str, float]:
        """Get all current environmental conditions."""
        return {
            'timestep': self.current_timestep,
            'wind_power_coefficient': self.wind_power_coefficient,
            'energy_price_eur_kwh': self.energy_price_eur_kwh,
            'air_density_kg_m3': self.air_density_kg_m3,
            'current_wind_power_mw': self.current_wind_power_mw,
            'current_energy_price_eur_mwh': self.current_energy_price_eur_mwh
        }
    
    def get_state(self) -> Dict:
        """Return current state."""
        return {
            **super().get_state(),
            **self.get_current_conditions()
        }

    @property
    def current_wind_power_mw(self) -> float:
        """Get current available wind power in MW."""
        if self.use_direct_power:
            return self.current_power_mw
            
        return self.wind_power_coefficient * self.installed_wind_capacity_mw

    @property
    def current_energy_price_eur_mwh(self) -> float:
        """Get current energy price in EUR/MWh."""
        return self.energy_price_eur_kwh * 1000.0

    def get_minute_of_hour(self, t: float) -> int:
        """Get minute within current hour (0-59) for arbitration logic."""
        return int(t * 60) % 60
    
    def get_future_price(self, minutes_ahead: int) -> float:
        """
        Get price at a future time (minutes ahead).
        
        Used by coordinator for ramp-down anticipation.
        """
        future_minute = self.current_minute + minutes_ahead
        future_hour = future_minute // 60
        future_minute_of_hour = future_minute % 60
        future_price_slot = future_minute_of_hour // 15
        future_price_index = (future_hour * 4) + future_price_slot
        
        if self.price_data is not None:
            idx = future_price_index % len(self.price_data)
            row = self.price_data.iloc[idx]
            return row.get('price_eur_kwh', 0.06) * 1000.0  # Convert to EUR/MWh
        return 60.0  # Default fallback
    
    def get_future_power(self, minutes_ahead: int = 60) -> float:
        """
        Get wind power availability at a future time (minutes ahead).
        
        Wind data is hourly, so we look ahead in hour increments.
        
        Args:
            minutes_ahead: Number of minutes into the future
            
        Returns:
            Forecasted wind power in MW
        """
        hours_ahead = minutes_ahead / 60.0
        
        # Handle direct power input (minute resolution)
        if self.use_direct_power and self.wind_data is not None:
            wind_rows = len(self.wind_data)
            future_minute_absolute = self.current_minute + minutes_ahead
            
            # Logic matching step() for 8-hour test or general minute data
            # Logic matching step() for 8-hour test or general minute data
            # Simplified: If using direct power, assume minute-level resolution for these test files
            # This covers 480 (8h), 10080 (7d), and 525600 (1y)
            future_idx = min(future_minute_absolute, wind_rows - 1)

            if future_idx < len(self.wind_data):
                row = self.wind_data.iloc[future_idx]
                return row.get('Power_MW', 0.0)

        # Use wind resolution logic for standard hourly data
        wind_rows = len(self.wind_data) if self.wind_data is not None else 8760
        wind_factor = wind_rows / 8760.0
        
        # Approximate future index
        future_idx = int((self.current_timestep + hours_ahead) * wind_factor) % wind_rows
        
        future_coeff = 0.0
        if self.wind_data is not None and future_idx < len(self.wind_data):
            row = self.wind_data.iloc[future_idx]
            future_coeff = row.get('wind_power_coefficient', 0.0)
            
        return future_coeff * self.installed_wind_capacity_mw
