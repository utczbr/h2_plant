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

import CoolProp.CoolProp as CP
from typing import Dict, Tuple, Optional

# Per-variable quantization functions: single arithmetic op, no log10/floor
_QUANT_P = 500.0       # 500 Pa bins (~0.005 bar)
_QUANT_T = 10.0        # 0.1 K bins (round to nearest 0.1 K)
_QUANT_GENERIC = 100.0  # Generic fallback


def _quantize(name: str, value: float) -> float:
    """Quantize a value based on its property name."""
    if name == 'P':
        return round(value / _QUANT_P) * _QUANT_P
    if name == 'T':
        return round(value * _QUANT_T) / _QUANT_T
    return round(value / _QUANT_GENERIC) * _QUANT_GENERIC


class CoolPropLUT:
    """
    Caching wrapper for CoolProp thermodynamic property calculations.

    Stores previously calculated results in a dictionary, returning cached
    values when inputs match within per-variable quantization bins.

    Cache Key Structure:
        (output_property, input1_name, input1_value_q, input2_name, input2_value_q, fluid)
    """
    _cache: Dict[Tuple[str, str, float, str, float, str], float] = {}

    @staticmethod
    def PropsSI(output: str, name1: str, value1: float, name2: str, value2: float, fluid: str) -> float:
        """
        Cached version of CoolProp.PropsSI.

        Returns cached result if available, otherwise computes via CoolProp
        and stores for future use.
        """
        v1_q = _quantize(name1, value1)
        v2_q = _quantize(name2, value2)

        key = (output, name1, v1_q, name2, v2_q, fluid)

        cache = CoolPropLUT._cache
        if key in cache:
            return cache[key]

        try:
            val = CP.PropsSI(output, name1, value1, name2, value2, fluid)
            cache[key] = val
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
