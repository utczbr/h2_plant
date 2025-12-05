"""
Legacy import adapters for backward compatibility.

Provides deprecated wrappers for old import paths during transition period.
Will be removed in v3.0.
"""

import warnings
from h2_plant.components.production.atr_source import ATRProductionSource
from h2_plant.components.compression.filling_compressor import FillingCompressor


def _deprecated_import(old_name: str, new_name: str, new_class):
    """Create deprecated wrapper for old import."""
    warnings.warn(
        f"{old_name} is deprecated. Use {new_name} instead. "
        f"Legacy support will be removed in v3.0.",
        DeprecationWarning,
        stacklevel=3
    )
    return new_class


# Legacy class wrappers
class ATRModel(ATRProductionSource):
    """DEPRECATED: Use h2_plant.components.production.ATRProductionSource instead."""
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ATRModel is deprecated. Use ATRProductionSource instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)
