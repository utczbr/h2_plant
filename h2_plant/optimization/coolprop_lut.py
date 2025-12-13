"""
CoolProp Lookup Table with In-Memory Caching.

This module provides a caching wrapper around CoolProp.PropsSI() to reduce
redundant thermodynamic property calculations during simulation.

Performance Optimization:
    CoolProp property evaluations are computationally expensive (~0.1-1ms each).
    For simulations calling the same conditions repeatedly (e.g., during
    Newton-Raphson iterations), caching provides significant speedup.

Cache Strategy:
    - Input values are rounded to 3 significant figures to increase hit rate.
    - Cache is stored as a class-level dictionary for persistence across calls.
    - Same physical conditions with minor numerical differences share cache entries.

Usage:
    Replace `CP.PropsSI(...)` with `CoolPropLUT.PropsSI(...)` for automatic caching.
"""

import math
import CoolProp.CoolProp as CP
from typing import Dict, Tuple, Optional


class CoolPropLUT:
    """
    Caching wrapper for CoolProp thermodynamic property calculations.

    Stores previously calculated results in a dictionary, returning cached
    values when inputs match within 3 significant figures.

    Cache Key Structure:
        (output_property, input1_name, input1_value, input2_name, input2_value, fluid)

    Attributes:
        _cache (Dict): Class-level cache dictionary.

    Example:
        >>> # First call computes and caches
        >>> h = CoolPropLUT.PropsSI('H', 'P', 101325, 'T', 298.15, 'Water')
        >>> # Second call returns cached value
        >>> h2 = CoolPropLUT.PropsSI('H', 'P', 101325, 'T', 298.15, 'Water')
    """
    _cache: Dict[Tuple[str, str, float, str, float, str], float] = {}

    @staticmethod
    def _round_sig(x: float, sig: int = 3) -> float:
        """
        Round a number to specified significant figures.

        Rounding increases cache hit rate by mapping similar values to
        identical keys, at the cost of minor accuracy loss.

        Args:
            x (float): Value to round.
            sig (int): Number of significant figures. Default: 3.

        Returns:
            float: Rounded value.
        """
        if x == 0:
            return 0.0
        return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)

    @staticmethod
    def PropsSI(output: str, name1: str, value1: float, name2: str, value2: float, fluid: str) -> float:
        """
        Cached version of CoolProp.PropsSI.

        Returns cached result if available, otherwise computes via CoolProp
        and stores for future use.

        Args:
            output (str): Property to calculate ('H', 'D', 'S', 'C', etc.).
            name1 (str): First input property name ('P', 'T', 'S', etc.).
            value1 (float): First input value (SI units).
            name2 (str): Second input property name.
            value2 (float): Second input value (SI units).
            fluid (str): Fluid name ('H2', 'Water', 'O2', etc.).

        Returns:
            float: Property value (SI units), or 0.0 if calculation fails.

        Example:
            >>> density = CoolPropLUT.PropsSI('D', 'P', 350e5, 'T', 298.15, 'H2')
        """
        v1_rounded = CoolPropLUT._round_sig(value1, 3)
        v2_rounded = CoolPropLUT._round_sig(value2, 3)

        key = (output, name1, v1_rounded, name2, v2_rounded, fluid)

        if key in CoolPropLUT._cache:
            return CoolPropLUT._cache[key]

        try:
            val = CP.PropsSI(output, name1, value1, name2, value2, fluid)
            CoolPropLUT._cache[key] = val
            return val
        except Exception:
            return 0.0

    @staticmethod
    def clear_cache() -> None:
        """
        Clear all cached property values.

        Call when simulation parameters change significantly or to
        free memory after long simulations.
        """
        CoolPropLUT._cache.clear()
