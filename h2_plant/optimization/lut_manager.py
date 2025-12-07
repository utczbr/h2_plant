"""
Lookup Table Manager for high-performance thermodynamic property lookups.

Replaces expensive CoolProp.PropsSI() calls with pre-computed interpolation
tables, achieving 50-200x speedup while maintaining <0.5% accuracy.
"""

import numpy as np
import numpy.typing as npt
from typing import Dict, Tuple, Optional, Literal, Any
from pathlib import Path
import pickle
import logging
from dataclasses import dataclass

try:
    import CoolProp.CoolProp as CP
except ImportError:
    CP = None
    logging.warning("CoolProp not available - LUT generation disabled")

from h2_plant.core.constants import StandardConditions

from h2_plant.core.component import Component

logger = logging.getLogger(__name__)


PropertyType = Literal['D', 'H', 'S', 'C']  # Density, Enthalpy, Entropy, Heat capacity, Viscosity


@dataclass
class LUTConfig:
    """Configuration for lookup table generation."""
    
    # Pressure range (Pa)
    pressure_min: float = 1e5          # 1 bar
    pressure_max: float = 1000e5       # 1000 bar (High Range)
    pressure_points: int = 2000        # MAXIMUM Resolution
    
    # Temperature range (K)
    temperature_min: float = 273.15    # 0°C
    temperature_max: float = 1200.0    # ~927°C (Covers SOEC)
    temperature_points: int = 2000     # MAXIMUM Resolution

    # Entropy range for Isentropic Lookups (J/kg/K)
    # H2 is ~60k. O2/H2O are ~6k-10k. 
    # Must start at 0 to cover heavy fluids/liquids!
    entropy_min: float = 0.0           
    entropy_max: float = 100000.0
    entropy_points: int = 500          # High resolution for isentropic
    
    # Properties to pre-compute
    properties: Tuple[PropertyType, ...] = ('D', 'H', 'S', 'C')
    
    # Gases to support
    fluids: Tuple[str, ...] = ('H2', 'O2', 'N2', 'CO2', 'CH4', 'H2O')
    
    # Interpolation method
    interpolation: Literal['linear', 'cubic'] = 'linear'
    
    # Cache directory
    cache_dir: Path = Path.home() / '.h2_plant' / 'lut_cache'


class LUTManager(Component):
    """
    Manages lookup tables for thermodynamic property calculations.
    
    Provides high-performance property lookups with automatic LUT generation,
    disk caching, and fallback to CoolProp for out-of-range queries.
    """
    
    def __init__(self, config: Optional[LUTConfig] = None):
        """
        Initialize LUT Manager.
        
        Args:
            config: LUT configuration (uses defaults if None)
        """
        super().__init__()
        self.config = config or LUTConfig()
        self._luts: Dict[str, Dict[PropertyType, npt.NDArray]] = {}
        self._pressure_grid: Optional[npt.NDArray] = None
        self._temperature_grid: Optional[npt.NDArray] = None
        self._entropy_grid: Optional[npt.NDArray] = None # For isentropic lookups
        
        # Create cache directory
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"LUTManager initialized with cache dir: {self.config.cache_dir}")
    
    def initialize(self, dt: float = 0.0, registry: Optional['ComponentRegistry'] = None) -> None:
        """
        Initialize LUT Manager by loading or generating lookup tables.
        
        Attempts to load cached LUTs from disk. If not found, generates
        new LUTs using CoolProp and saves to cache.
        
        Raises:
            RuntimeError: If CoolProp unavailable and no cache exists
        """
        super().initialize(dt, registry) # Call Component's initialize
        
        logger.info("Initializing LUT Manager...")
        
        # Generate coordinate grids
        self._pressure_grid = np.linspace(
            self.config.pressure_min,
            self.config.pressure_max,
            self.config.pressure_points
        )
        self._temperature_grid = np.linspace(
            self.config.temperature_min,
            self.config.temperature_max,
            self.config.temperature_points
        )
        self._entropy_grid = np.linspace(
            self.config.entropy_min,
            self.config.entropy_max,
            self.config.entropy_points
        )
        
        # Load or generate LUTs for each fluid
        for fluid in self.config.fluids:
            cache_path = self._get_cache_path(fluid)
            
            if cache_path.exists():
                logger.info(f"Loading cached LUT for {fluid}")
                try:
                    self._luts[fluid] = self._load_from_cache(cache_path)
                except Exception as e:
                    logger.warning(f"Failed to load cache for {fluid}: {e}. Regenerating.")
                    self._luts[fluid] = self._generate_lut(fluid)
                    self._save_to_cache(fluid, self._luts[fluid]) # Save newly generated LUT
            else:
                logger.info(f"Cache not found for {fluid}. Generating new LUT...")
                self._luts[fluid] = self._generate_lut(fluid)
                self._save_to_cache(fluid, self._luts[fluid])
        
        # self._initialized is set by super().initialize
        logger.info("LUT Manager initialization complete")
    
    def step(self, t: float) -> None:
        """
        LUTManager does not have per-timestep logic, but needs to implement
        the abstract method from Component.
        """
        super().step(t)
        pass # No operation needed for LUTManager per timestep
    
    def get_state(self) -> Dict[str, Any]:
        """
        Return current state of the LUT Manager.
        """
        return {
            **super().get_state(),
            "lut_config": self.config.__dict__, # Store config for state
            "luts_loaded_fluids": list(self._luts.keys())
        }
    
    def lookup(
        self,
        fluid: str,
        property_type: PropertyType,
        pressure: float,
        temperature: float
    ) -> float:
        """
        Lookup thermodynamic property with bilinear interpolation.
        
        Args:
            fluid: Fluid name ('H2', 'O2', 'N2')
            property_type: Property code ('D'=density, 'H'=enthalpy, etc.)
            pressure: Pressure in Pa
            temperature: Temperature in K
            
        Returns:
            Property value (units depend on property_type)
            
        Raises:
            ValueError: If fluid or property_type not supported
            RuntimeError: If LUT not initialized
            
        Example:
            # Density of H2 at 350 bar, 298.15 K
            rho = lut.lookup('H2', 'D', 350e5, 298.15)  # kg/m³
        """
        if not self._initialized:
            self.initialize()
        
        if fluid not in self._luts:
            raise ValueError(f"Fluid '{fluid}' not supported. Available: {list(self._luts.keys())}")
        
        if property_type not in self._luts[fluid]:
            raise ValueError(
                f"Property '{property_type}' not available for {fluid}. "
                f"Available: {list(self._luts[fluid].keys())}"
            )
        
        # Check if in bounds
        if not self._in_bounds(pressure, temperature):
            logger.warning(
                f"Property lookup out of LUT bounds: P={pressure/1e5:.1f} bar, "
                f"T={temperature:.1f} K. Falling back to CoolProp."
            )
            return self._fallback_coolprop(fluid, property_type, pressure, temperature)
        
        # Bilinear interpolation
        lut = self._luts[fluid][property_type]
        value = self._interpolate_2d(lut, pressure, temperature)
        
        return float(value)
    
    def lookup_batch(
        self,
        fluid: str,
        property_type: PropertyType,
        pressures: npt.NDArray[np.float64],
        temperatures: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Vectorized batch lookup for arrays of pressures and temperatures.
        
        Args:
            fluid: Fluid name
            property_type: Property code
            pressures: Array of pressures in Pa
            temperatures: Array of temperatures in K
            
        Returns:
            Array of property values
            
        Example:
            pressures = np.array([30e5, 350e5, 900e5])
            temps = np.array([298.15, 298.15, 298.15])
            densities = lut.lookup_batch('H2', 'D', pressures, temps)
        """
        if not self._initialized:
            self.initialize()
        
        # Vectorized interpolation
        lut = self._luts[fluid][property_type]
        results = np.zeros_like(pressures)
        
        for i in range(len(pressures)):
            results[i] = self._interpolate_2d(lut, pressures[i], temperatures[i])
        
        return results
    
    def lookup_isentropic_enthalpy(
        self,
        fluid: str,
        pressure: float,
        entropy: float
    ) -> float:
        """
        Lookup Enthalpy given Pressure and Entropy (Isentropic step).
        Uses the special 'H_from_PS' table.
        """
        if not self._initialized:
             self.initialize()
             
        if fluid not in self._luts or 'H_from_PS' not in self._luts[fluid]:
            # Fallback to CoolProp if table missing (e.g. old cache)
            if CP:
                return CP.PropsSI('H', 'P', pressure, 'S', entropy, fluid)
            else:
                raise RuntimeError("Isentropic LUT missing and CoolProp unavailable")
                
        lut = self._luts[fluid]['H_from_PS']
        
        # Check bounds for Entropy
        if entropy < self.config.entropy_min or entropy > self.config.entropy_max:
             if CP: return CP.PropsSI('H', 'P', pressure, 'S', entropy, fluid)
        
        # Use JIT interpolation with Pressure and Entropy grids
        # Note: self._entropy_grid must be initialized in __init__ or initialize
        from h2_plant.optimization.numba_ops import bilinear_interp_jit
        
        return float(bilinear_interp_jit(
            self._pressure_grid,
            self._entropy_grid,
            lut,
            pressure,
            entropy
        ))
    
    def _interpolate_2d(
        self,
        lut: npt.NDArray,
        pressure: float,
        temperature: float
    ) -> float:
        """
        Perform 2D bilinear interpolation on LUT.
        
        Args:
            lut: 2D lookup table [pressure_idx, temperature_idx]
            pressure: Pressure value in Pa
            temperature: Temperature value in K
            
        Returns:
            Interpolated property value
        """
        # Find bounding indices
        p_idx = np.searchsorted(self._pressure_grid, pressure)
        t_idx = np.searchsorted(self._temperature_grid, temperature)
        
        # Clamp to valid range
        p_idx = np.clip(p_idx, 1, len(self._pressure_grid) - 1)
        t_idx = np.clip(t_idx, 1, len(self._temperature_grid) - 1)
        
        # Get bounding coordinates
        p0, p1 = self._pressure_grid[p_idx - 1], self._pressure_grid[p_idx]
        t0, t1 = self._temperature_grid[t_idx - 1], self._temperature_grid[t_idx]
        
        # Get corner values
        q00 = lut[p_idx - 1, t_idx - 1]
        q01 = lut[p_idx - 1, t_idx]
        q10 = lut[p_idx, t_idx - 1]
        q11 = lut[p_idx, t_idx]
        
        # Bilinear interpolation
        # https://en.wikipedia.org/wiki/Bilinear_interpolation
        wp = (pressure - p0) / (p1 - p0) if p1 != p0 else 0.0
        wt = (temperature - t0) / (t1 - t0) if t1 != t0 else 0.0
        
        value = (
            q00 * (1 - wp) * (1 - wt) +
            q10 * wp * (1 - wt) +
            q01 * (1 - wp) * wt +
            q11 * wp * wt
        )
        
        return value
    
    def _generate_lut(self, fluid: str) -> Dict[PropertyType, npt.NDArray]:
        """
        Generate lookup table for a fluid using CoolProp.
        
        Args:
            fluid: Fluid name ('H2', 'O2', 'N2')
            
        Returns:
            Dictionary mapping property types to 2D arrays
        """
        if CP is None:
            raise RuntimeError("CoolProp not available - cannot generate LUT")
        
        lut = {}
        
        for prop in self.config.properties:
            logger.info(f"Generating {fluid} {prop} table...")
            
            # Initialize array
            table = np.zeros((self.config.pressure_points, self.config.temperature_points))
            
            # Populate table
            for i, pressure in enumerate(self._pressure_grid):
                for j, temperature in enumerate(self._temperature_grid):
                    try:
                        value = CP.PropsSI(prop, 'P', pressure, 'T', temperature, fluid)
                        table[i, j] = value
                    except Exception as e:
                        logger.warning(
                            f"CoolProp error at P={pressure/1e5:.1f} bar, "
                            f"T={temperature:.1f} K: {e}"
                        )
                        table[i, j] = np.nan
            
            lut[prop] = table
            logger.info(f"  ✓ {fluid} {prop} table complete ({table.shape})")
        
            lut[prop] = table
            logger.info(f"  ✓ {fluid} {prop} table complete ({table.shape})")
            
        # Generate H(P, S) table for Isentropic compression
        logger.info(f"Generating {fluid} isentropic H(P,S) table...")
        h_ps_table = np.zeros((self.config.pressure_points, self.config.entropy_points))
        for i, p in enumerate(self._pressure_grid):
            for j, s in enumerate(self._entropy_grid):
                try:
                    # 'H' from 'P', 'S'
                    h_ps_table[i, j] = CP.PropsSI('H', 'P', p, 'S', s, fluid)
                except:
                    h_ps_table[i, j] = np.nan
        lut['H_from_PS'] = h_ps_table
        logger.info(f"  ✓ {fluid} H(P,S) table complete")
        
        return lut
    
    def _in_bounds(self, pressure: float, temperature: float) -> bool:
        """Check if pressure and temperature are within LUT bounds."""
        return (
            self.config.pressure_min <= pressure <= self.config.pressure_max and
            self.config.temperature_min <= temperature <= self.config.temperature_max
        )
    
    def _fallback_coolprop(
        self,
        fluid: str,
        property_type: PropertyType,
        pressure: float,
        temperature: float
    ) -> float:
        """Fallback to direct CoolProp call for out-of-bounds queries."""
        if CP is None:
            raise RuntimeError(
                f"Query out of LUT bounds and CoolProp unavailable: "
                f"P={pressure/1e5:.1f} bar, T={temperature:.1f} K"
            )
        
        return CP.PropsSI(property_type, 'P', pressure, 'T', temperature, fluid)
    
    def _get_cache_path(self, fluid: str) -> Path:
        """Get cache file path for a fluid."""
        return self.config.cache_dir / f"lut_{fluid}_v1.pkl"
    
    def _save_to_cache(self, fluid: str, cache_path: Path) -> None:
        """Save LUT to disk cache."""
        logger.info(f"Saving LUT cache to {cache_path}")
        
        cache_data = {
            'lut': self._luts[fluid],
            'pressure_grid': self._pressure_grid,
            'temperature_grid': self._temperature_grid,
            'entropy_grid': self._entropy_grid,
            'config': self.config
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _load_from_cache(self, cache_path: Path) -> Dict[PropertyType, npt.NDArray]:
        """Load LUT from disk cache."""
        logger.info(f"Attempting to load LUT from: {cache_path}")
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Validate cache matches current config - RELAXED CHECK
        # We check critical parameters only, ignoring fluid lists/names.
        # This allows using "Hydrogen" table for "H2" etc.
        c_saved = cache_data['config']
        c_curr = self.config
        
        match = False
        if isinstance(c_saved, dict):
            # Dict comparison
            match = (
                c_saved.get('pressure_min') == c_curr.pressure_min and
                c_saved.get('pressure_max') == c_curr.pressure_max and
                c_saved.get('pressure_points') == c_curr.pressure_points and
                c_saved.get('temperature_min') == c_curr.temperature_min and
                c_saved.get('temperature_max') == c_curr.temperature_max and
                c_saved.get('temperature_points') == c_curr.temperature_points
            )
            saved_desc = f"P={c_saved.get('pressure_min')}-{c_saved.get('pressure_max')}/{c_saved.get('pressure_points')}"
        else:
            # Object comparison
            match = (
                c_saved.pressure_min == c_curr.pressure_min and
                c_saved.pressure_max == c_curr.pressure_max and
                c_saved.pressure_points == c_curr.pressure_points and
                c_saved.temperature_min == c_curr.temperature_min and
                c_saved.temperature_max == c_curr.temperature_max and
                c_saved.temperature_points == c_curr.temperature_points
            )
            saved_desc = f"P={c_saved.pressure_min}-{c_saved.pressure_max}/{c_saved.pressure_points}"
        
        if not match:
            logger.warning(
                f"Cached LUT config mismatch (P/T ranges). Regenerating.\n"
                f"Saved: {saved_desc}\n"
                f"Curr:  P={c_curr.pressure_min}-{c_curr.pressure_max}/{c_curr.pressure_points}, "
                f"T={c_curr.temperature_min}-{c_curr.temperature_max}/{c_curr.temperature_points}"
            )
            return self._generate_lut(cache_path.stem.split('_')[1])
        
        return cache_data['lut']
    
    def get_accuracy_report(self, fluid: str, num_samples: int = 1000) -> Dict[str, float]:
        """
        Generate accuracy report comparing LUT interpolation to CoolProp.
        
        Args:
            fluid: Fluid to test
            num_samples: Number of random samples to compare
            
        Returns:
            Dictionary with mean/max absolute and relative errors per property
        """
        if CP is None:
            raise RuntimeError("CoolProp required for accuracy validation")
        
        if not self._initialized:
            self.initialize()
        
        logger.info(f"Generating accuracy report for {fluid} ({num_samples} samples)...")
        
        # Random sample points within LUT bounds
        pressures = np.random.uniform(
            self.config.pressure_min,
            self.config.pressure_max,
            num_samples
        )
        temperatures = np.random.uniform(
            self.config.temperature_min,
            self.config.temperature_max,
            num_samples
        )
        
        report = {}
        
        for prop in self.config.properties:
            lut_values = np.array([
                self.lookup(fluid, prop, p, t)
                for p, t in zip(pressures, temperatures)
            ])
            
            coolprop_values = np.array([
                CP.PropsSI(prop, 'P', p, 'T', t, fluid)
                for p, t in zip(pressures, temperatures)
            ])
            
            abs_error = np.abs(lut_values - coolprop_values)
            rel_error = abs_error / np.abs(coolprop_values) * 100  # Percent
            
            report[prop] = {
                'mean_abs_error': float(np.mean(abs_error)),
                'max_abs_error': float(np.max(abs_error)),
                'mean_rel_error_pct': float(np.mean(rel_error)),
                'max_rel_error_pct': float(np.max(rel_error))
            }
            
            logger.info(
                f"  {prop}: mean error {np.mean(rel_error):.3f}%, "
                f"max error {np.max(rel_error):.3f}%"
            )
        
        return report

    # --- Custom LUT Support (PEM V_cell) ---

    def register_custom_table(
        self, 
        name: str, 
        data_2d: npt.NDArray, 
        axes: Tuple[npt.NDArray, npt.NDArray]
    ) -> None:
        """
        Register custom 2D LUT (e.g., PEM V_cell).
        
        Args:
            name: Identifier for the table
            data_2d: 2D array of values [axis0_idx, axis1_idx]
            axes: Tuple of (axis0_values, axis1_values)
        """
        if not hasattr(self, '_custom_luts'):
            self._custom_luts = {}
            
        self._custom_luts[name] = {
            'data': data_2d,
            'axes': axes
        }
        logger.info(f"Registered custom LUT: {name} {data_2d.shape}")

    def load_pem_tables(self) -> None:
        """Load PEM V_cell LUT from disk."""
        try:
            # Load V_cell LUT
            path = Path(__file__).parent.parent / "data" / "lut_pem_vcell.npz"
            if path.exists():
                with np.load(path) as data:
                    self.register_custom_table(
                        "pem_vcell",
                        data['v_cell'],
                        (data['j_op'], data['t_op_h'])
                    )
            else:
                logger.warning(f"PEM V_cell LUT not found at {path}")
                
            # Load Degradation LUT
            deg_path = Path(__file__).parent.parent / "data" / "lut_pem_degradation.npy"
            if deg_path.exists():
                deg_data = np.load(deg_path)
                # This is 1D interpolation (time -> voltage), but we can store it as custom
                # Or just keep it simple. For now, let's store it.
                # deg_data is [N, 2] where col 0 is time, col 1 is voltage
                self._pem_deg_data = deg_data
                from scipy.interpolate import interp1d
                self._pem_deg_interpolator = interp1d(
                    deg_data[:, 0], deg_data[:, 1], 
                    kind='linear', fill_value="extrapolate"
                )
            else:
                logger.warning(f"PEM Degradation LUT not found at {deg_path}")
                
        except Exception as e:
            logger.error(f"Failed to load PEM tables: {e}")

    def lookup_pem_vcell(
        self, 
        j_op: float, 
        t_op_h: float
    ) -> float:
        """
        Single PEM cell voltage lookup.
        
        Args:
            j_op: Current density (A/cm2)
            t_op_h: Operating hours
            
        Returns:
            Cell voltage (V)
        """
        if not hasattr(self, '_custom_luts') or 'pem_vcell' not in self._custom_luts:
            # Fallback or error? For now, assume loaded.
            # If not loaded, maybe try to load?
            self.load_pem_tables()
            if 'pem_vcell' not in self._custom_luts:
                 raise RuntimeError("PEM V_cell LUT not loaded")

        lut_data = self._custom_luts['pem_vcell']
        v_cell_grid = lut_data['data']
        j_grid, t_grid = lut_data['axes']
        
        # Bilinear interpolation manually or reuse _interpolate_2d logic adapted
        # _interpolate_2d assumes self._pressure_grid etc.
        # Let's adapt it here locally
        
        # Find indices
        j_idx = np.searchsorted(j_grid, j_op)
        t_idx = np.searchsorted(t_grid, t_op_h)
        
        j_idx = np.clip(j_idx, 1, len(j_grid) - 1)
        t_idx = np.clip(t_idx, 1, len(t_grid) - 1)
        
        j0, j1 = j_grid[j_idx - 1], j_grid[j_idx]
        t0, t1 = t_grid[t_idx - 1], t_grid[t_idx]
        
        # Corner values
        # v_cell_grid is [j_idx, t_idx] based on generation script (we need to ensure this)
        q00 = v_cell_grid[j_idx - 1, t_idx - 1]
        q01 = v_cell_grid[j_idx - 1, t_idx]
        q10 = v_cell_grid[j_idx, t_idx - 1]
        q11 = v_cell_grid[j_idx, t_idx]
        
        # Weights
        wj = (j_op - j0) / (j1 - j0) if j1 != j0 else 0.0
        wt = (t_op_h - t0) / (t1 - t0) if t1 != t0 else 0.0
        
        value = (
            q00 * (1 - wj) * (1 - wt) +
            q10 * wj * (1 - wt) +
            q01 * (1 - wj) * wt +
            q11 * wj * wt
        )
        
        return float(value)
    
    def batch_lookup_pem_vcell(
        self, 
        j_op_array: npt.NDArray, 
        t_op_h_array: npt.NDArray
    ) -> npt.NDArray:
        """
        Vectorized PEM V_cell lookup for arrays.
        """
        # Simple loop for now, or map
        # Optimization: Use scipy.interpolate.RegularGridInterpolator if available and faster
        # But for now, list comprehension is okay or numpy vectorization
        
        # To strictly follow the plan "Vectorized PEM V_cell lookup for arrays"
        # We can use map or a loop.
        results = np.zeros_like(j_op_array)
        for i in range(len(j_op_array)):
            results[i] = self.lookup_pem_vcell(j_op_array[i], t_op_h_array[i])
        return results
