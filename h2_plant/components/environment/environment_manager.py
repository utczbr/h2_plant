"""
Environment Manager Component.

This module manages external environmental data including wind power availability,
electricity prices, and weather conditions. It serves as the interface between
time-series data files and the simulation engine.

Data Management:
    - **Wind Data**: Supports both capacity factor (0-1) and direct power (MW)
      input formats. Resolution can be hourly or minute-level.
    - **Price Data**: Electricity market prices in EUR/MWh, supporting 15-minute
      intraday markets or hourly day-ahead prices.
    - **Weather Data**: Air density and temperature for equipment performance
      derating calculations (future extension).

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Loads and validates time-series data files.
    - `step()`: Advances environmental conditions to current simulation time.
    - `get_state()`: Returns current wind, price, and weather conditions.

Data Resolution Handling:
    The manager correctly maps simulation time to data indices across different
    resolutions. For minute-level simulation with hourly wind data, interpolation
    or sample-and-hold is applied as appropriate.

File Formats:
    - Wind: CSV with 'wind_power_coefficient' or 'Power_MW' columns.
    - Prices: CSV with 'price_eur_mwh' or 'price_eur_kwh' columns.
    - Unified: Combined environment_data_YYYY.csv with all series.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from h2_plant.core.component import Component


class EnvironmentManager(Component):
    """
    External environmental data manager for simulation.

    Provides time-series data for wind power availability, electricity prices,
    and weather conditions. Supports multiple data file formats and resolutions.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Loads CSV data files and preprocesses for fast lookup.
        - `step()`: Updates current conditions based on simulation time.
        - `get_state()`: Returns all current environmental parameters.

    The manager uses NumPy array caching for O(1) lookups during simulation,
    avoiding repeated DataFrame access overhead.

    Attributes:
        wind_power_coefficient (float): Current wind capacity factor (0-1).
        energy_price_eur_kwh (float): Current electricity price (EUR/kWh).
        current_wind_power_mw (float): Calculated available wind power (MW).
        use_direct_power (bool): True if using direct MW input vs. coefficient.

    Example:
        >>> env = EnvironmentManager(
        ...     wind_data_path='wind_2024.csv',
        ...     price_data_path='prices_2024.csv',
        ...     installed_wind_capacity_mw=20.0
        ... )
        >>> env.initialize(dt=1/60, registry=registry)
        >>> env.step(t=0.0)
        >>> power = env.get_wind_power_availability(20000)  # kW
        >>> price = env.get_current_energy_price()  # EUR/kWh
    """

    def __init__(
        self,
        wind_data_path: Optional[str] = None,
        price_data_path: Optional[str] = None,
        use_default_data: bool = True,
        installed_wind_capacity_mw: float = 20.0
    ):
        """
        Initialize the environment manager.

        Args:
            wind_data_path (str, optional): Path to wind data CSV file.
                If None and use_default_data is True, uses package default.
            price_data_path (str, optional): Path to price data CSV file.
                If None and use_default_data is True, uses package default.
            use_default_data (bool): Whether to use default data files when
                explicit paths are not provided. Default: True.
            installed_wind_capacity_mw (float): Installed wind generation
                capacity in MW. Used to convert coefficients to absolute power.
                Default: 20.0 MW.
        """
        super().__init__()

        # Resolve default file paths relative to package data directory
        if wind_data_path is None and use_default_data:
            wind_data_path = str(Path(__file__).parent.parent.parent / 'h2_plant' / 'data' / 'wind_data.csv')
        if price_data_path is None and use_default_data:
            price_data_path = str(Path(__file__).parent.parent.parent / 'h2_plant' / 'data' / 'EnergyPriceAverage2023-24.csv')

        self.wind_data_path = wind_data_path
        self.price_data_path = price_data_path
        self.installed_wind_capacity_mw = installed_wind_capacity_mw

        # DataFrame storage for loaded data
        self.wind_data: Optional[pd.DataFrame] = None
        self.price_data: Optional[pd.DataFrame] = None

        # Current state variables
        self.current_timestep = 0
        self.current_minute = 0
        self.wind_power_coefficient = 0.5
        self.air_density_kg_m3 = 1.225
        self.energy_price_eur_kwh = 0.06

        # Data resolution tracking
        self.price_resolution_minutes = 15
        self.wind_resolution_hours = 1

        # Direct power input mode (vs. coefficient mode)
        self.use_direct_power = False
        self.current_power_mw = 0.0

    def initialize(self, dt: float, registry) -> None:
        """
        Prepare the manager for simulation execution.

        Fulfills the Component Lifecycle Contract initialization phase by
        loading environmental data files and preprocessing for fast access.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)
        self._load_data()

    def _load_data(self) -> None:
        """
        Load and preprocess environmental data files.

        Attempts to load wind and price data from specified paths, falling
        back to unified data files or synthetic defaults if unavailable.
        Automatically detects data format and column naming conventions.
        """
        try:
            # ================================================================
            # Wind Data Loading
            # ================================================================
            if self.wind_data_path and Path(self.wind_data_path).exists():
                try:
                    # Handle European CSV format (semicolon separator, comma decimal)
                    try:
                        self.wind_data = pd.read_csv(self.wind_data_path, sep=';', decimal=',')
                        if len(self.wind_data.columns) < 2:
                            self.wind_data = pd.read_csv(self.wind_data_path, sep=',', decimal='.')
                    except Exception:
                        self.wind_data = pd.read_csv(self.wind_data_path, sep=',', decimal='.')

                    print(f"EnvironmentManager: Loaded wind data from {self.wind_data_path} "
                          f"({len(self.wind_data)} rows)")

                    self.wind_data.columns = self.wind_data.columns.str.strip()

                    # Detect direct power input column
                    power_aliases = ['Power_MW', 'potencia_2_turbinas_MW', 'ActivePower', 'Power']
                    found_alias = None
                    for alias in power_aliases:
                        if alias in self.wind_data.columns:
                            found_alias = alias
                            break

                    if found_alias:
                        self.use_direct_power = True
                        self.wind_data['Power_MW'] = pd.to_numeric(
                            self.wind_data[found_alias].astype(str).str.replace(',', '.'),
                            errors='coerce'
                        )
                        print(f"EnvironmentManager: Detected '{found_alias}' column - "
                              "using direct power input mode")

                except Exception as e:
                    print(f"EnvironmentManager: Error reading wind file: {e}")

            # ================================================================
            # Price Data Loading
            # ================================================================
            if self.price_data_path and Path(self.price_data_path).exists():
                try:
                    df_precos = pd.read_csv(self.price_data_path, index_col=0)

                    # Identify price column by name pattern
                    col_valor_preco = [c for c in df_precos.columns
                                       if 'price' in c.lower() or 'eur' in c.lower()]
                    if col_valor_preco:
                        col_name = col_valor_preco[0]
                        if df_precos[col_name].dtype == 'object':
                            df_precos[col_name] = df_precos[col_name].astype(str).str.replace(',', '.')

                        df_precos[col_name] = pd.to_numeric(df_precos[col_name], errors='coerce')

                        self.price_data = df_precos
                        self.price_data['price_eur_mwh'] = df_precos[col_name]

                        print(f"EnvironmentManager: Loaded price data from {self.price_data_path} "
                              f"({len(self.price_data)} rows)")
                    else:
                        self.price_data = pd.read_csv(self.price_data_path)
                        print(f"EnvironmentManager: Loaded raw price data "
                              f"(columns: {self.price_data.columns.tolist()})")

                except Exception as e:
                    print(f"EnvironmentManager: Error reading price file: {e}")

            # ================================================================
            # Unified File Fallback
            # ================================================================
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

            # ================================================================
            # Generate Synthetic Defaults
            # ================================================================
            if self.wind_data is None:
                print("EnvironmentManager: No wind data found, using synthetic defaults")
                self._create_default_wind_data()

            if self.price_data is None:
                print("EnvironmentManager: No price data found, using synthetic defaults")
                self._create_default_price_data()

            # ================================================================
            # Post-Processing
            # ================================================================
            # Ensure price_eur_kwh column exists for unified interface
            if self.price_data is not None:
                if 'price_eur_kwh' not in self.price_data.columns:
                    if 'price_eur_mwh' in self.price_data.columns:
                        self.price_data['price_eur_kwh'] = self.price_data['price_eur_mwh'] / 1000.0
                    elif 'price' in self.price_data.columns:
                        mean_price = self.price_data['price'].mean()
                        if mean_price > 10:
                            self.price_data['price_eur_kwh'] = self.price_data['price'] / 1000.0
                        else:
                            self.price_data['price_eur_kwh'] = self.price_data['price']
                    else:
                        col0 = self.price_data.columns[0]
                        if self.price_data[col0].dtype in [float, int]:
                            if self.price_data[col0].mean() > 10:
                                self.price_data['price_eur_kwh'] = self.price_data[col0] / 1000.0
                            else:
                                self.price_data['price_eur_kwh'] = self.price_data[col0]

            # Detect price data resolution for proper indexing
            self.price_resolution_factor = 1.0
            if self.price_data is not None:
                rows = len(self.price_data)
                if rows > 8800:
                    self.price_resolution_factor = rows / 8760.0
                    print(f"EnvironmentManager: Detected high resolution price data "
                          f"({rows} rows). Factor: {self.price_resolution_factor:.2f}")

        except Exception as e:
            print(f"EnvironmentManager: Error loading data ({e}), using defaults")
            self._create_default_wind_data()
            self._create_default_price_data()

    def _create_default_wind_data(self) -> None:
        """
        Generate synthetic wind data when external files are unavailable.

        Creates one year (8760 hours) of sinusoidal wind patterns with
        daily variation. Coefficient ranges from 0.0 to 1.0.
        """
        hours = 8760
        # Daily sinusoidal pattern with mean of 0.5
        wind_coeff = 0.3 + 0.4 * np.sin(np.arange(hours) * 2 * np.pi / 24)
        self.wind_data = pd.DataFrame({
            'hour': np.arange(hours),
            'wind_power_coefficient': np.clip(wind_coeff, 0.0, 1.0),
            'air_density': 1.225
        })

    def _create_default_price_data(self) -> None:
        """
        Generate synthetic price data when external files are unavailable.

        Creates one year (8760 hours) of day/night price patterns with
        higher prices during peak hours (6am-10pm).
        """
        hours = 8760
        base_price = 0.06  # EUR/kWh baseline
        hour_of_day = np.arange(hours) % 24
        price_multiplier = np.where((hour_of_day >= 6) & (hour_of_day < 22), 1.3, 0.7)

        self.price_data = pd.DataFrame({
            'hour': np.arange(hours),
            'price_eur_kwh': base_price * price_multiplier
        })
        self.price_resolution_factor = 1.0

    def step(self, t: float) -> None:
        """
        Update environmental conditions for current simulation time.

        Maps simulation time to appropriate data indices, handling different
        resolutions for wind (hourly) and price (15-minute) data.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        # Convert to absolute minutes for price slot calculation
        minute_absolute = int(t * 60)
        self.current_minute = minute_absolute

        # Hour index for annual data wrap-around
        hour_index = int(t) % 8760
        self.current_timestep = hour_index

        # ====================================================================
        # Wind Data Update
        # ====================================================================
        if self.wind_data is not None:
            # Cache numpy arrays for O(1) access
            if not hasattr(self, '_wind_coeffs'):
                self._wind_coeffs = (self.wind_data['wind_power_coefficient'].values
                                     if 'wind_power_coefficient' in self.wind_data else
                                     np.full(len(self.wind_data), 0.5))
                self._wind_power_mw = (self.wind_data['Power_MW'].values
                                       if 'Power_MW' in self.wind_data else
                                       np.zeros(len(self.wind_data)))
                self._wind_air_density = (self.wind_data['air_density'].values
                                         if 'air_density' in self.wind_data else
                                         np.full(len(self.wind_data), 1.225))
                self._wind_rows = len(self.wind_data)

            wind_rows = self._wind_rows

            # Index selection depends on data resolution and mode
            if self.use_direct_power and wind_rows >= 525600:
                wind_idx = minute_absolute % wind_rows
            elif self.use_direct_power and (wind_rows == 480 or wind_rows == 10080):
                wind_idx = min(minute_absolute, wind_rows - 1)
            else:
                wind_idx = min(hour_index, wind_rows - 1)

            if self.use_direct_power:
                self.current_power_mw = self._wind_power_mw[wind_idx]
                self.wind_power_coefficient = self.current_power_mw / self.installed_wind_capacity_mw
            else:
                self.wind_power_coefficient = self._wind_coeffs[wind_idx]

            self.air_density_kg_m3 = self._wind_air_density[wind_idx]

        # ====================================================================
        # Price Data Update
        # ====================================================================
        if self.price_data is not None:
            if not hasattr(self, '_price_values'):
                self._price_values = (self.price_data['price_eur_kwh'].values
                                     if 'price_eur_kwh' in self.price_data else
                                     np.full(len(self.price_data), 0.06))
                self._price_rows = len(self.price_data)

            # Map to 15-minute price slots
            minute_of_hour = minute_absolute % 60
            price_slot = minute_of_hour // 15
            price_index = (hour_index * 4) + price_slot

            price_idx = price_index % self._price_rows

            self.energy_price_eur_kwh = self._price_values[price_idx]

    def get_wind_power_availability(self, installed_capacity_kw: float) -> float:
        """
        Calculate available wind power at current conditions.

        Args:
            installed_capacity_kw (float): Installed wind capacity in kW.

        Returns:
            float: Available power in kW.
        """
        if self.use_direct_power:
            return self.current_power_mw * 1000.0

        return installed_capacity_kw * self.wind_power_coefficient

    def get_current_energy_price(self) -> float:
        """
        Get current electricity price.

        Returns:
            float: Current price in EUR/kWh.
        """
        return self.energy_price_eur_kwh

    def get_current_conditions(self) -> Dict[str, float]:
        """
        Get all current environmental parameters.

        Returns:
            Dict[str, float]: Dictionary with timestep, wind coefficient,
                price, air density, wind power (MW), and price (EUR/MWh).
        """
        return {
            'timestep': self.current_timestep,
            'wind_power_coefficient': self.wind_power_coefficient,
            'energy_price_eur_kwh': self.energy_price_eur_kwh,
            'air_density_kg_m3': self.air_density_kg_m3,
            'current_wind_power_mw': self.current_wind_power_mw,
            'current_energy_price_eur_mwh': self.current_energy_price_eur_mwh
        }

    def get_state(self) -> Dict:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access.

        Returns:
            Dict: Complete state dictionary including environmental conditions.
        """
        return {
            **super().get_state(),
            **self.get_current_conditions()
        }

    @property
    def current_wind_power_mw(self) -> float:
        """
        Get current available wind power in MW.

        Returns:
            float: Wind power in MW.
        """
        if self.use_direct_power:
            return self.current_power_mw

        return self.wind_power_coefficient * self.installed_wind_capacity_mw

    @property
    def current_energy_price_eur_mwh(self) -> float:
        """
        Get current energy price in EUR/MWh.

        Returns:
            float: Price in EUR/MWh (converted from kWh internally).
        """
        return self.energy_price_eur_kwh * 1000.0

    def get_minute_of_hour(self, t: float) -> int:
        """
        Get minute within current hour for intra-hour arbitration.

        Args:
            t (float): Simulation time in hours.

        Returns:
            int: Minute of hour (0-59).
        """
        return int(t * 60) % 60

    def get_future_price(self, minutes_ahead: int) -> float:
        """
        Get electricity price at a future time.

        Used by dispatch coordinator for ramping decisions based on
        anticipated price changes.

        Args:
            minutes_ahead (int): Lookahead time in minutes.

        Returns:
            float: Forecasted price in EUR/MWh.
        """
        future_minute = self.current_minute + minutes_ahead
        future_hour = future_minute // 60
        future_minute_of_hour = future_minute % 60
        future_price_slot = future_minute_of_hour // 15
        future_price_index = (future_hour * 4) + future_price_slot

        if self.price_data is not None:
            idx = future_price_index % len(self.price_data)
            row = self.price_data.iloc[idx]
            return row.get('price_eur_kwh', 0.06) * 1000.0
        return 60.0

    def get_future_power(self, minutes_ahead: int = 60) -> float:
        """
        Get forecasted wind power at a future time.

        Used by dispatch coordinator for ramping and storage decisions.

        Args:
            minutes_ahead (int): Lookahead time in minutes. Default: 60.

        Returns:
            float: Forecasted wind power in MW.
        """
        hours_ahead = minutes_ahead / 60.0

        # Direct power mode: minute-level resolution available
        if self.use_direct_power and self.wind_data is not None:
            wind_rows = len(self.wind_data)
            future_minute_absolute = self.current_minute + minutes_ahead

            future_idx = min(future_minute_absolute, wind_rows - 1)

            if future_idx < len(self.wind_data):
                row = self.wind_data.iloc[future_idx]
                return row.get('Power_MW', 0.0)

        # Coefficient mode: hourly resolution
        wind_rows = len(self.wind_data) if self.wind_data is not None else 8760
        wind_factor = wind_rows / 8760.0

        future_idx = int((self.current_timestep + hours_ahead) * wind_factor) % wind_rows

        future_coeff = 0.0
        if self.wind_data is not None and future_idx < len(self.wind_data):
            row = self.wind_data.iloc[future_idx]
            future_coeff = row.get('wind_power_coefficient', 0.0)

        return future_coeff * self.installed_wind_capacity_mw
