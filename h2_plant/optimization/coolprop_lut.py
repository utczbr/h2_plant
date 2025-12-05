import math
import CoolProp.CoolProp as CP
from typing import Dict, Tuple, Optional

class CoolPropLUT:
    """
    A simple caching wrapper for CoolProp calls to improve performance.
    Uses a dictionary to store previously calculated results.
    """
    _cache: Dict[Tuple[str, str, float, str, float, str], float] = {}
    
    @staticmethod
    def _round_sig(x: float, sig: int = 3) -> float:
        if x == 0: return 0.0
        return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)

    @staticmethod
    def PropsSI(output: str, name1: str, value1: float, name2: str, value2: float, fluid: str) -> float:
        """
        Cached version of CoolProp.PropsSI.
        Rounds inputs to 3 significant digits to reduce cache misses.
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
    def clear_cache():
        CoolPropLUT._cache.clear()
