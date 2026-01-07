"""
Integer-based enumerations for high-performance simulation.

All enums use IntEnum for:
- NumPy array compatibility (dtype=np.int32)
- Numba JIT compilation support
- Memory efficiency (4 bytes vs 40+ bytes for strings)
- Fast comparisons and vectorization
"""

from enum import IntEnum


class TankState(IntEnum):
    """
    State of a hydrogen storage tank.
    
    Used for vectorized tank operations in TankArray and scheduling logic.
    
    Examples:
        # Single tank
        if tank.state == TankState.IDLE:
            tank.fill(mass)
        
        # Vectorized (NumPy array of states)
        states = np.array([TankState.IDLE, TankState.FILLING, TankState.FULL])
        available_indices = np.where(states == TankState.IDLE)[0]
    """
    IDLE = 0         # Tank ready for filling or discharging
    FILLING = 1      # Currently being filled
    DISCHARGING = 2  # Currently being discharged
    FULL = 3         # At capacity (>99% full)
    EMPTY = 4        # Depleted (<1% full)
    MAINTENANCE = 5  # Offline for maintenance


class ProductionState(IntEnum):
    """
    State of hydrogen production source (Electrolyzer or ATR).
    
    Examples:
        if electrolyzer.state == ProductionState.RUNNING:
            h2_output = electrolyzer.step(t)
    """
    OFFLINE = 0      # Not producing
    STARTING = 1     # Warm-up phase (ATR only, typically 30-60 min)
    RUNNING = 2      # Active production
    SHUTTING_DOWN = 3  # Cool-down phase
    MAINTENANCE = 4  # Scheduled maintenance
    FAULT = 5        # Error state requiring intervention


class CompressorMode(IntEnum):
    """
    Operating mode for compression equipment.
    
    Examples:
        if compressor.mode == CompressorMode.LP_TO_HP:
            compressor.transfer_mass(lp_tank, hp_tank)
    """
    IDLE = 0         # Not operating
    LP_TO_HP = 1     # Transferring from low-pressure to high-pressure storage
    HP_TO_DELIVERY = 2  # Boosting to delivery pressure (900 bar)
    RECIRCULATION = 3   # Internal pressure balancing


class AllocationStrategy(IntEnum):
    """
    Strategy for allocating hydrogen demand across production pathways.
    
    Used by DualPathCoordinator to split demand between electrolyzer and ATR.
    
    Examples:
        coordinator = DualPathCoordinator(strategy=AllocationStrategy.COST_OPTIMAL)
        elec_demand, atr_demand = coordinator.allocate(total_demand, t)
    """
    COST_OPTIMAL = 0     # Minimize total production cost based on energy prices
    PRIORITY_GRID = 1    # Maximize electrolyzer usage (grid-powered)
    PRIORITY_ATR = 2     # Maximize ATR usage (natural gas)
    BALANCED = 3         # 50/50 split between pathways
    EMISSIONS_OPTIMAL = 4  # Minimize CO2 emissions


class FlowDirection(IntEnum):
    """
    Direction of hydrogen flow in the system.
    
    Used for tracking mass flow through pipelines and validation.
    """
    NONE = 0         # No flow
    PRODUCTION_TO_LP = 1   # Source → Low-pressure storage
    LP_TO_HP = 2     # Low-pressure → High-pressure (compression)
    HP_TO_DELIVERY = 3     # High-pressure → Customer delivery
    RECYCLE = 4      # Return flow (pressure balancing)


class SystemMode(IntEnum):
    """
    Overall system operating mode.
    
    Controls high-level system behavior and safety interlocks.
    """
    STARTUP = 0      # Initial startup sequence
    NORMAL = 1       # Standard operation
    PEAK_DEMAND = 2  # High-demand mode (maximize output)
    LOW_DEMAND = 3   # Low-demand mode (optimize efficiency)
    EMERGENCY_STOP = 4  # Emergency shutdown
    MAINTENANCE = 5  # Maintenance mode


class DispatchStrategyEnum(IntEnum):
    """
    Strategy for dispatch power allocation between electrolyzers.
    
    Used by dispatch framework to select allocation logic.
    
    Examples:
        strategy = create_dispatch_strategy(DispatchStrategyEnum.ECONOMIC_SPOT)
        result = strategy.decide(inputs, state)
    """
    SOEC_ONLY = 0        # Single SOEC electrolyzer dispatch
    REFERENCE_HYBRID = 1  # Hybrid SOEC/PEM with arbitrage
    ECONOMIC_SPOT = 2    # Economic spot purchase for non-RFNBO H2
