"""
Visualization Utilities Module.

Provides centralized helper functions for:
- Column name resolution (topology abstraction)
- Data downsampling (performance optimization)
- Unit conversion (K→C, Pa→Bar standardization)
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union
import logging

import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHED_CONFIG = None

def load_viz_config(config_path: Path = None) -> Dict[str, Any]:
    """Load visualization config once and cache it."""
    global _CACHED_CONFIG
    if _CACHED_CONFIG is None:
        if config_path is None:
            # Assuming this file is in h2_plant/visualization/, config is in scenarios/
            # Original path logic from user plan was Path(__file__).parent.parent / 'visualization_config.yaml' -> h2_plant/visualization_config.yaml?
            # User system has it in /home/stuart/Documentos/Planta Hidrogenio/scenarios/visualization_config.yaml
            # So from h2_plant/visualization/utils.py:
            #   parent -> visualization
            #   parent -> h2_plant
            #   parent -> Planta Hidrogenio
            #   then -> scenarios/visualization_config.yaml
            base_dir = Path(__file__).resolve().parent.parent.parent
            config_path = base_dir / 'scenarios' / 'visualization_config.yaml'
            
        try:
            with open(config_path) as f:
                _CACHED_CONFIG = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load visualization config from {config_path}: {e}")
            _CACHED_CONFIG = {}
            
    return _CACHED_CONFIG

def get_viz_config(key: str, default: Any = None) -> Any:
    """
    Get visualization config value with dot notation.
    
    Examples:
        >>> get_viz_config('styling.colors.pem', '#2196F3')
        >>> get_viz_config('performance.max_points_default', 2000)
    """
    config = load_viz_config()
    keys = key.split('.')
    value = config
    for k in keys:
        if isinstance(value, dict):
            value = value.get(k)
        else:
            return default
    return value if value is not None else default

def get_library_preference(graphid: str = None, category: str = None) -> List[str]:
    """Returns ['matplotlib', 'plotly'] or subset based on config."""
    config = load_viz_config()
    dual = config.get('dual_generation', {})
    
    prefs = []
    
    # 1. Graph specific override
    if graphid and graphid in dual.get('graphs', {}):
        p = dual['graphs'][graphid]
        prefs = p.split() if isinstance(p, str) else []
    # 2. Category override
    elif category and category in dual.get('categories', {}):
        p = dual['categories'][category]
        prefs = p.split() if isinstance(p, str) else []
    # 3. Global preference
    else:
        p = dual.get('preference', 'both')
        prefs = p.split() if isinstance(p, str) else []
    
    # Resolve 'both', 'auto' keywords to actual libraries
    libs = []
    if 'matplotlib' in prefs or 'both' in prefs or 'auto' in prefs:
        libs.append('matplotlib')
    if 'plotly' in prefs or 'both' in prefs or 'auto' in prefs:
        libs.append('plotly')
        
    return libs


# ==============================================================================
# COLUMN NAME RESOLUTION
# ==============================================================================

def find_column(df: pd.DataFrame, component_id: str, metric: str, 
                fallback_patterns: Optional[List[str]] = None) -> Optional[str]:
    """
    Find the DataFrame column name for a given component and metric.
    
    Resolves the topology-abstraction problem by searching multiple
    naming conventions used across the codebase.
    
    Args:
        df: Simulation history DataFrame.
        component_id: Component identifier (e.g., 'KOD_1', 'SOEC_Cluster').
        metric: Metric name (e.g., 'temperature_c', 'water_removed_kg_h').
        fallback_patterns: Optional list of additional patterns to try.
        
    Returns:
        Column name if found, None otherwise.
        
    Example:
        >>> col = find_column(df, 'KOD_1', 'outlet_temp_c')
        >>> if col: ax.plot(df[col])
    """
    # Primary pattern: ComponentID_metric
    primary = f"{component_id}_{metric}"
    if primary in df.columns:
        return primary
    
    # Alternative patterns
    patterns = [
        f"{component_id.lower()}_{metric}",
        f"{component_id}_{metric.lower()}",
        f"{component_id.replace('_', '')}_{metric}",
    ]
    
    if fallback_patterns:
        patterns.extend([f"{component_id}_{fp}" for fp in fallback_patterns])
    
    for pattern in patterns:
        if pattern in df.columns:
            return pattern
    
    # Fuzzy match: find columns containing both component and metric
    for col in df.columns:
        if component_id in col and metric.split('_')[0] in col:
            return col
    
    return None


def find_columns_by_type(df: pd.DataFrame, component_type: str, 
                          metric: str) -> Dict[str, str]:
    """
    Find all columns matching a component type and metric pattern.
    
    Args:
        df: History DataFrame.
        component_type: Component type prefix (e.g., 'Chiller', 'KOD', 'Coalescer').
        metric: Metric suffix (e.g., 'cooling_load_kw', 'water_removed_kg_h').
        
    Returns:
        Dict mapping component_id to column name.
        
    Example:
        >>> cols = find_columns_by_type(df, 'Chiller', 'cooling_load_kw')
        >>> # Returns {'Chiller_1': 'Chiller_1_cooling_load_kw', ...}
    """
    results = {}
    
    for col in df.columns:
        if metric in col.lower():
            # Extract component ID from column name
            parts = col.rsplit('_', metric.count('_') + 1)
            if len(parts) > 1:
                potential_id = parts[0]
                if component_type in potential_id:
                    results[potential_id] = col
    
    return results


def get_component_stream_type(df: pd.DataFrame, component_id: str) -> str:
    """
    Detect if a component processes primarily H2 or O2 based on composition.
    
    Args:
        df: History DataFrame.
        component_id: Component name (e.g., 'KOD_1').
        
    Returns:
        'H2', 'O2', or 'Unknown'.
    """
    # Check naming convention first
    if component_id.startswith('O2_'):
        return 'O2'
    if 'H2' in component_id or any(x in component_id for x in ['Deoxo', 'PSA']):
        return 'H2'
    
    # Check composition columns
    h2_col = find_column(df, component_id, 'outlet_h2_mass_frac')
    o2_col = find_column(df, component_id, 'outlet_o2_mass_frac')
    
    if h2_col and o2_col:
        h2_mean = df[h2_col].mean() if h2_col in df.columns else 0
        o2_mean = df[o2_col].mean() if o2_col in df.columns else 0
        return 'H2' if h2_mean > o2_mean else 'O2'
    
    return 'Unknown'


# ==============================================================================
# DOWNSAMPLING
# ==============================================================================

def downsample_dataframe(df: pd.DataFrame, max_points: int = 2000, 
                          method: str = 'stride') -> pd.DataFrame:
    """
    Downsample a DataFrame for faster plotting.
    
    Args:
        df: Input DataFrame.
        max_points: Maximum number of rows in output.
        method: 'stride' (simple striding) or 'minmax' (preserves peaks).
        
    Returns:
        Downsampled DataFrame.
    """
    n = len(df)
    if n <= max_points:
        return df
    
    if method == 'stride':
        stride = max(1, n // max_points)
        return df.iloc[::stride].copy()
    
    elif method == 'minmax':
        # Preserves local minima and maxima for visual accuracy
        stride = max(1, n // (max_points // 2))
        indices = set()
        
        for col in df.select_dtypes(include=[np.number]).columns:
            # Add indices of local extrema
            for i in range(0, n, stride):
                chunk = df[col].iloc[i:i+stride]
                if len(chunk) > 0:
                    indices.add(i + chunk.idxmin() - df.index[0])
                    indices.add(i + chunk.idxmax() - df.index[0])
        
        return df.iloc[sorted(indices)].copy()
    
    else:
        logger.warning(f"Unknown downsampling method: {method}, using stride")
        return downsample_dataframe(df, max_points, 'stride')


def calculate_stride(n_points: int, max_points: int = 2000) -> int:
    """
    Calculate stride value for downsampling.
    
    Args:
        n_points: Total number of data points.
        max_points: Maximum desired points.
        
    Returns:
        Stride value (minimum 1).
    """
    return max(1, n_points // max_points)


def downsample_list(data: Union[List[Any], np.ndarray], max_points: int = 2000) -> List[Any]:
    """
    Downsample a list or numpy array for generic usage (e.g. JSON export).
    
    Args:
        data: Input list or array.
        max_points: Maximum number of points to return.
        
    Returns:
        Downsampled list.
    """
    if data is None:
        return []
        
    n = len(data)
    if n <= max_points:
        return list(data) if isinstance(data, (np.ndarray, pd.Series)) else data
        
    stride = max(1, n // max_points)
    
    # Handle list vs numpy/pandas
    if isinstance(data, (np.ndarray, pd.Series)):
        return data[::stride].tolist()
    else:
        return data[::stride]


# ==============================================================================
# UNIT CONVERSION
# ==============================================================================

def kelvin_to_celsius(value: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """Convert temperature from Kelvin to Celsius."""
    return value - 273.15


def celsius_to_kelvin(value: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """Convert temperature from Celsius to Kelvin."""
    return value + 273.15


def pascal_to_bar(value: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """Convert pressure from Pascal to Bar."""
    return value / 1e5


def bar_to_pascal(value: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """Convert pressure from Bar to Pascal."""
    return value * 1e5


def auto_convert_temperature(df: pd.DataFrame, column: str, 
                              target_unit: str = 'celsius') -> pd.Series:
    """
    Auto-detect temperature unit and convert to target.
    
    Uses heuristic: values > 200 are assumed Kelvin.
    
    Args:
        df: DataFrame containing the column.
        column: Column name.
        target_unit: 'celsius' or 'kelvin'.
        
    Returns:
        Converted Series.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found")
    
    data = df[column]
    mean_val = data.mean()
    
    # Heuristic detection
    is_kelvin = mean_val > 200 or '_k' in column.lower()
    is_celsius = mean_val < 200 or '_c' in column.lower()
    
    if target_unit == 'celsius':
        return kelvin_to_celsius(data) if is_kelvin else data
    else:
        return celsius_to_kelvin(data) if is_celsius else data


def auto_convert_pressure(df: pd.DataFrame, column: str, 
                           target_unit: str = 'bar') -> pd.Series:
    """
    Auto-detect pressure unit and convert to target.
    
    Uses heuristic: values > 1000 are assumed Pascal.
    
    Args:
        df: DataFrame containing the column.
        column: Column name.
        target_unit: 'bar' or 'pascal'.
        
    Returns:
        Converted Series.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found")
    
    data = df[column]
    mean_val = data.mean()
    
    # Heuristic detection
    is_pascal = mean_val > 1000 or '_pa' in column.lower()
    is_bar = mean_val < 1000 or '_bar' in column.lower()
    
    if target_unit == 'bar':
        return pascal_to_bar(data) if is_pascal else data
    else:
        return bar_to_pascal(data) if is_bar else data


# ==============================================================================
# TIME BASIS CONTRACT
# ==============================================================================

def get_dt_hours(df: pd.DataFrame) -> float:
    """
    Get loop time step in hours.
    
    Standardizes the conversion from discrete steps to energy (MWh).
    Derives from metadata if available, defaults to 1 minute (1/60 h).
    """
    dt_seconds = df.attrs.get('dt_seconds', 60.0)
    return dt_seconds / 3600.0

def get_time_axis_hours(df: pd.DataFrame) -> np.ndarray:
    """
    Get standardized time axis in hours.
    
    Unified source of truth for x-axis.
    """
    if 'minute' in df.columns:
        return df['minute'].values / 60.0
    elif hasattr(df.index, 'is_numeric') and df.index.is_numeric():
         # Fallback to index assuming minute steps if not specified
         dt_h = get_dt_hours(df)
         # If index is integers, multiply by dt_h * 60? No, assume index is "steps"
         # But usually index IS minutes in this sim.
         return df.index.values * (dt_h * 60.0) 
    else:
        # Fallback for non-numeric index or raw arrays
        return np.arange(len(df)) * get_dt_hours(df)


# ==============================================================================
# CONFIGURATION HELPERS
# ==============================================================================

def get_config_value(df: pd.DataFrame, key: str, default: Any = None) -> Any:
    """
    Retrieve configuration value from df.attrs['config'].
    
    Args:
        df: DataFrame with config stored in attrs.
        key: Configuration key.
        default: Default value if not found.
        
    Returns:
        Configuration value or default.
    """
    config = df.attrs.get('config', {})
    return config.get(key, default)
