"""
ATR Rate Limiter - Quasi-Steady State Input Dynamics

Implements velocity-limited ramping of the oxygen flow setpoint to model
the physical inertia of valves, thermal mass, and control loops. This 
ensures thermodynamic consistency at every timestep by evaluating lookup
tables at valid intermediate operating points.

Physics Basis:
- Operating Range: O₂ = 7.125 – 23.75 kmol/hr (30% – 100% capacity)
- Max Ramp Rate: Transition 30%→100% takes 10 minutes
- Rate Limit: (23.75 - 7.125) / 10 min = 1.6625 kmol/hr/min
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class ATRRateLimiterConfig:
    """Configuration for ATR rate limiter parameters."""
    min_o2_kmol_h: float = 7.125      # 30% capacity
    max_o2_kmol_h: float = 23.75      # 100% capacity  
    max_ramp_rate_kmol_h_min: float = 1.6625  # kmol/hr per minute
    idle_threshold_kmol_h: float = 0.1  # Below this, plant is considered idle


class ATRRateLimiter:
    """
    Rate limiter for ATR oxygen flow setpoint changes.
    
    Implements first-order lag with velocity constraint (slew rate limiter).
    The internal state F_O2_internal represents the actual operating point
    of the plant, which ramps towards the target setpoint at a limited rate.
    
    Algorithm:
        1. Calculate error: e = F_target - F_internal
        2. Determine max step: Δmax = rate_limit × dt
        3. If |e| ≤ Δmax: snap to target (F_internal = F_target)
        4. Else: step by Δmax in direction of error
        5. Clamp to operating bounds
    
    Attributes:
        f_o2_internal: Current rate-limited oxygen flow (kmol/hr)
        target_o2: Last requested target setpoint (kmol/hr)
        is_ramping: True if currently transitioning
    """
    
    def __init__(
        self, 
        initial_o2: Optional[float] = None,
        config: Optional[ATRRateLimiterConfig] = None
    ):
        """
        Initialize the rate limiter.
        
        Args:
            initial_o2: Starting oxygen flow (kmol/hr). Defaults to minimum.
            config: Rate limiter configuration. Uses defaults if None.
        """
        self.config = config or ATRRateLimiterConfig()
        self.f_o2_internal = initial_o2 if initial_o2 is not None else self.config.min_o2_kmol_h
        self.target_o2 = self.f_o2_internal
        self.is_ramping = False
        
        # Ensure initial state is within bounds
        self._clamp_internal()
    
    def _clamp_internal(self) -> None:
        """Clamp internal state to operating bounds."""
        self.f_o2_internal = np.clip(
            self.f_o2_internal,
            self.config.min_o2_kmol_h,
            self.config.max_o2_kmol_h
        )
    
    def update(self, target_o2: float, dt_seconds: float) -> float:
        """
        Update internal state towards target with rate limiting.
        
        This is the core rate-limiter equation:
            F_internal(t+dt) = F_internal(t) + sign(e) × min(|e|, Δmax)
        
        Args:
            target_o2: Desired oxygen flow setpoint (kmol/hr)
            dt_seconds: Simulation timestep in seconds
            
        Returns:
            Rate-limited oxygen flow value (kmol/hr) for this timestep
        """
        # Handle idle state: if target is below threshold, keep at minimum
        if target_o2 < self.config.idle_threshold_kmol_h:
            # Plant going to idle - but we don't ramp below minimum
            self.target_o2 = self.config.min_o2_kmol_h
            self.f_o2_internal = self.config.min_o2_kmol_h
            self.is_ramping = False
            return self.f_o2_internal
        
        # Clamp target to valid operating range
        clamped_target = np.clip(
            target_o2,
            self.config.min_o2_kmol_h,
            self.config.max_o2_kmol_h
        )
        self.target_o2 = clamped_target
        
        # Convert dt to minutes for rate calculation
        dt_min = dt_seconds / 60.0
        max_step = self.config.max_ramp_rate_kmol_h_min * dt_min
        
        # Calculate error (deviation from target)
        error = clamped_target - self.f_o2_internal
        
        # Apply rate limit
        if abs(error) <= max_step:
            # Within one step of target - snap to it
            self.f_o2_internal = clamped_target
            self.is_ramping = False
        else:
            # Rate limited - step towards target
            direction = 1.0 if error > 0 else -1.0
            self.f_o2_internal += direction * max_step
            self.is_ramping = True
        
        # Safety clamp (should be redundant but ensures robustness)
        self._clamp_internal()
        
        return self.f_o2_internal
    
    def get_ramp_time_remaining(self) -> float:
        """
        Calculate estimated time to reach current target.
        
        Returns:
            Remaining ramp time in minutes (0 if at target)
        """
        if not self.is_ramping:
            return 0.0
        
        distance = abs(self.target_o2 - self.f_o2_internal)
        return distance / self.config.max_ramp_rate_kmol_h_min
    
    def reset(self, o2_value: Optional[float] = None) -> None:
        """
        Reset the rate limiter state.
        
        Args:
            o2_value: Value to reset to. Defaults to minimum if None.
        """
        self.f_o2_internal = o2_value if o2_value is not None else self.config.min_o2_kmol_h
        self.target_o2 = self.f_o2_internal
        self.is_ramping = False
        self._clamp_internal()
    
    @property
    def ramp_progress(self) -> float:
        """
        Get ramp progress as fraction 0-1.
        
        Returns:
            1.0 if at target, 0.0 if just started ramping, 
            intermediate value during transition.
        """
        if not self.is_ramping:
            return 1.0
        
        # This is approximate - assumes target hasn't changed during ramp
        total_range = self.config.max_o2_kmol_h - self.config.min_o2_kmol_h
        if total_range < 1e-6:
            return 1.0
            
        distance_to_target = abs(self.target_o2 - self.f_o2_internal)
        return max(0.0, 1.0 - distance_to_target / total_range)
