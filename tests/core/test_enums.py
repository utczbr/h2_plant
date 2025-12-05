import numpy as np
from h2_plant.core.enums import TankState, ProductionState

def test_enum_integer_values():
    """Test enums have integer values."""
    assert isinstance(TankState.IDLE.value, int)
    assert TankState.IDLE == 0
    assert TankState.FILLING == 1

def test_enum_numpy_compatibility():
    """Test enums work with NumPy arrays."""
    states = np.array([TankState.IDLE, TankState.FULL, TankState.IDLE], dtype=np.int32)
    
    # Vectorized comparison
    idle_mask = states == TankState.IDLE
    assert np.array_equal(idle_mask, [True, False, True])
    
    # Indexing
    idle_indices = np.where(states == TankState.IDLE)[0]
    assert np.array_equal(idle_indices, [0, 2])
