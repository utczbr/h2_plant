"""Mixing components for combining multiple fluid streams."""

# Import components with graceful fallback for optional dependencies
__all__ = []

try:
    from h2_plant.components.mixing.multicomponent_mixer import MultiComponentMixer
    __all__.append('MultiComponentMixer')
except ImportError:
    pass

try:
    from h2_plant.components.mixing.oxygen_mixer import OxygenMixer
    __all__.append('OxygenMixer')
except ImportError:
    pass

try:
    from h2_plant.components.mixing.h2_distribution import H2Distribution
    __all__.append('H2Distribution')
except ImportError:
    pass

try:
    from h2_plant.components.mixing.water_mixer import WaterMixer
    __all__.append('WaterMixer')
except ImportError:
    pass
