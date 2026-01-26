"""
Dual-path production coordinator.

Coordinates two isolated production pathways with configurable
allocation strategies for demand splitting and economic optimization.
"""

from typing import Dict, Any, List, Optional
import logging

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.enums import AllocationStrategy
from h2_plant.core.exceptions import ComponentNotFoundError

logger = logging.getLogger(__name__)

# Add pem_and_soec to path for shadow prediction
import sys
from pathlib import Path
import copy

pem_soec_dir = Path(__file__).resolve().parent.parent / 'legacy' / 'pem_soec_reference'
if str(pem_soec_dir) not in sys.path:
    sys.path.insert(0, str(pem_soec_dir))

try:
    import soec_operator
    SOEC_AVAILABLE = True
except ImportError:
    SOEC_AVAILABLE = False
    logger.warning("soec_operator not available for shadow prediction")


class DualPathCoordinator(Component):
    """
    Coordinates dual-path hydrogen production system with Hybrid Dispatch Strategy.

    Manages:
    - Hybrid Dispatch (SOEC Base Load + PEM Balancing + Arbitrage)
    - Economic optimization based on Spot Price
    - Pathway health monitoring
    - Aggregate metrics
    """

    def __init__(
        self,
        pathway_ids: List[str],
        allocation_strategy: AllocationStrategy = AllocationStrategy.COST_OPTIMAL,
        demand_scheduler_id: str = 'demand_scheduler',
        h2_price_kg: float = 9.6,
        ppa_price_eur_mwh: float = 50.0
    ):
        super().__init__()

        self.pathway_ids = pathway_ids
        self.allocation_strategy = allocation_strategy
        self.demand_scheduler_id = demand_scheduler_id

        # Component references
        self._pathways: Dict[str, Any] = {}
        self._demand_scheduler: Optional[Component] = None
        self._environment: Optional[Component] = None
        self._soec_cluster: Optional[Component] = None
        self._pem_electrolyzer: Optional[Component] = None

        # Configuration (loaded from config)
        self.SOEC_MAX_CAPACITY = 11.52  # MW (6 * 1.92)
        self.MAX_PEM_POWER = 5.0  # MW
        self.MIN_PEM_POWER = 0.25  # MW (Legacy Alignment)
        self.PPA_PRICE = ppa_price_eur_mwh  # EUR/MWh
        self.H2_PRICE_KG = h2_price_kg  # EUR/kg
        self.SOEC_KWH_KG = 37.5  # kWh/kg
        self.ARBITRAGE_THRESHOLD = 306.0  # EUR/MWh (calculated)

        # State Variables (matching manager.py)
        self.force_sell_flag = False
        self.previous_soec_power_mw = 0.0
        self.sell_decision = 0  # 0: H2, 1: Sell
        
        # Phase 1 Enhancement: Sell window tracking
        self.arbitrage_trigger_minute = None  # Track when arbitrage started
        self.sell_window_remaining = 0  # Minutes left in 15-minute sell window

        # Current timestep outputs
        self.soec_setpoint_mw = 0.0
        self.pem_setpoint_mw = 0.0
        self.sold_power_mw = 0.0
        self.soec_actual_mw = 0.0

        # Cumulative metrics
        self.cumulative_production_kg = 0.0
        self.cumulative_sold_energy_mwh = 0.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize coordinator."""
        super().initialize(dt, registry)

        # Resolve components
        if registry.has('environment_manager'):
            self._environment = registry.get('environment_manager')
        if registry.has('demand_scheduler'):
            self._demand_scheduler = registry.get('demand_scheduler')
        if registry.has('soec_cluster'):
            self._soec_cluster = registry.get('soec_cluster')
        if registry.has('pem_electrolyzer_detailed'):
            self._pem_electrolyzer = registry.get('pem_electrolyzer_detailed')
            # Update MAX_PEM_POWER from component capacity if available
            if hasattr(self._pem_electrolyzer, 'max_power_mw'):
                self.MAX_PEM_POWER = float(self._pem_electrolyzer.max_power_mw)
                logger.info(f"Updated MAX_PEM_POWER from component: {self.MAX_PEM_POWER} MW")

        # Load config values if available
        # (Read from config.arbitration section)

        logger.info("DualPathCoordinator initialized with minute-level arbitration")

    def step(self, t: float) -> None:
        """Execute one minute of hybrid dispatch."""
        super().step(t)

        # Get inputs from environment
        P_offer_mw = 0.0
        spot_price_eur_mwh = 0.0
        minute_of_hour = 0

        if self._environment:
            P_offer_mw = self._environment.current_wind_power_mw
            spot_price_eur_mwh = self._environment.current_energy_price_eur_mwh
            P_offer_mw = self._environment.current_wind_power_mw
            spot_price_eur_mwh = self._environment.current_energy_price_eur_mwh
            minute_of_hour = self._environment.get_minute_of_hour(t)
            
        self.current_offer_mw = P_offer_mw # Store for aggregation

        # Get future offer (for minute 45 ramp-down check)
        P_future_offer = P_offer_mw
        if minute_of_hour >= 45 and self._environment:
            minutes_to_next_hour = 60 - minute_of_hour
            # Future wind power (must be in MW, matching current P_offer_mw)
            P_future_offer = self._environment.get_future_power(minutes_to_next_hour)

        # Execute arbitration logic (exact reference implementation)
        self._execute_dispatch_logic(
            t, P_offer_mw, P_future_offer, spot_price_eur_mwh, minute_of_hour
        )

        # Set component setpoints
        self._apply_setpoints()

        # Aggregate results
        self._aggregate_results()
        
    
    def _check_minute_zero_arbitrage(
        self, 
        t: float, 
        P_offer: float,
        current_spot_price: float, 
        minute_of_hour: int
    ) -> None:
        """
        Minute 0 exclusive arbitrage decision with defensive checks.
        
        Phase 1 Enhancement: Implements exact legacy logic from manager.py lines 241-258
        
        ONLY at minute 0:
        - Check if ramp up detected (offer > previous)
        - Compare sale profit vs H2 profit over 15-minute window  
        - Set force_sell_flag and track sell window
        
        Defensive checks:
        - Validate previous_soec_power_mw
        - Only trigger on ramp up
        - Track remaining sell window time
        """
        if minute_of_hour != 0:
            return
        
        # Defensive: Ensure we have valid previous power
        if self.previous_soec_power_mw < 0:
            logger.warning(f"t={t:.2f}: Previous power invalid, skipping arbitrage")
            return
        
        offer_prev_diff = P_offer - self.previous_soec_power_mw
        if offer_prev_diff <= 0.0:
            # Ramp down or flat: reset sell flag
            self.force_sell_flag = False
            self.sell_window_remaining = 0
            return
        
        # Calculate profits over 15-minute horizon (exactly as legacy)
        P_surplus = offer_prev_diff
        time_h_arbitrage = 15.0 / 60.0  # 15 minutes
        
        # Sales profit (sell surplus at spot price)
        sale_profit = P_surplus * time_h_arbitrage * (
            current_spot_price - self.PPA_PRICE
        )
        
        # H2 profit (produce surplus as H2)
        E_surplus_mwh = P_surplus * time_h_arbitrage
        mass_kg = E_surplus_mwh * (1000.0 / self.SOEC_KWH_KG)
        h2_profit = mass_kg * self.H2_PRICE_KG
        
        # Decision
        if sale_profit > h2_profit:
            self.force_sell_flag = True
            self.sell_window_remaining = 15  # 15 minutes
            self.arbitrage_trigger_minute = int(t * 60)  # Store trigger time
            logger.info(
                f"t={t:.1f}: MIN 0 ARBITRAGE: Sale={sale_profit:.2f}EUR > H2={h2_profit:.2f}EUR. "
                f"Selling {P_surplus:.2f}MW for 15 min"
            )
        else:
            self.force_sell_flag = False
            self.sell_window_remaining = 0
            logger.debug(
                f"t={t:.1f}: MIN 0: H2 production more profitable. "
                f"Sale={sale_profit:.2f} < H2={h2_profit:.2f}"
            )
    
    def _check_minute_45_constraints(
        self, 
        t: float, 
        minute_of_hour: int,
        P_offer: float, 
        P_future_offer: float
    ) -> None:
        """
        Minute 45 lookahead constraint with ramp-down preparation.
        
        Phase 1 Enhancement: Implements legacy logic from manager.py lines 267-275
        
        If next hour will ramp down significantly:
        - Reset sell flag to resume H2 production early
        - Allow modules time to prepare for shutdown sequence
        - Avoid aggressive ramps that could stress hardware
        """
        if minute_of_hour != 45:
            return
        
        ramp_down_expected = P_future_offer < P_offer
        if not ramp_down_expected:
            return
        
        # Phase 1.5 Alignment: Absolute comparison (Legacy Logic)
        P_soec_set_fut = min(P_future_offer, self.SOEC_MAX_CAPACITY)
        
        # If current power exceeds future capacity, we must reset to allow ramp down
        if self.previous_soec_power_mw > P_soec_set_fut:
            was_selling = self.force_sell_flag
            self.force_sell_flag = False
            self.sell_window_remaining = 0
            
            if was_selling:
                logger.info(
                    f"t={t:.1f}: MIN 45 CONSTRAINT: Future ramp={ramp_magnitude*100:.1f}%. "
                    f"Resetting sell flag early (offer={P_offer:.2f}MW → future={P_future_offer:.2f}MW)"
                )
            else:
                logger.debug(
                    f"t={t:.1f}: MIN 45: Significant ramp down detected but not selling"
                )
    
    def _execute_dispatch_logic(
        self, 
        t: float, 
        P_offer: float, 
        P_future_offer: float,
        current_spot_price: float,
        minute_of_hour: int
    ) -> None:
        """
        Core arbitration logic - EXACT implementation from manager.py.
        """

        # Calculate arbitrage limit
        h2_eq_price = (1000.0 / self.SOEC_KWH_KG) * self.H2_PRICE_KG
        arbitrage_limit = self.PPA_PRICE + h2_eq_price

        # --- PHASE 1 ENHANCEMENT: Call defensive arbitrage methods ---
        
        # 1. Minute 0 check (exclusive, with defensive validation)
        self._check_minute_zero_arbitrage(t, P_offer, current_spot_price, minute_of_hour)
        
        # 2. Minute 45 check (ramp-down lookahead constraint)
        self._check_minute_45_constraints(t, minute_of_hour, P_offer, P_future_offer)
        
        # 3. Decrement sell window if active - REMOVED FOR LEGACY ALIGNMENT
        # Legacy system does not have a 15-minute timeout; it persists until price drops or min 45.
        # if self.sell_window_remaining > 0:
        #     self.sell_window_remaining -= 1
        #     if self.sell_window_remaining == 0 and self.force_sell_flag:
        #         logger.info(f"t={t:.1f}: 15-minute sell window expired, resuming H2 production")
        #         self.force_sell_flag = False

        # --- LEGACY RESET LOGIC (maintains compatibility) ---
        if self.force_sell_flag and (current_spot_price <= arbitrage_limit):
            self.force_sell_flag = False
            logger.info(f"t={t:.2f}: Force sell reset - price dropped")

        # Set sell decision flag
        self.sell_decision = 1 if self.force_sell_flag else 0

        # --- BYPASS MODE (Priority 1: TOTAL SALE BYPASS OF SOEC) ---
        # Lines 226-248 from manager.py
        if self.sell_decision == 1:
            # BYPASS: Keep SOEC at previous power, sell surplus
            P_soec_set = self.previous_soec_power_mw
            P_soec_actual = self.previous_soec_power_mw  # Prediction
            P_pem = 0.0

            soec_offer_difference = P_offer - P_soec_actual
            P_sold = soec_offer_difference
            self.sold_power_mw = P_sold  # Set state for monitoring

        else:
            # --- NORMAL H2 PRODUCTION LOGIC ---
            # Lines 250-303 from manager.py

            # 1. SOEC Setpoint Calculation
            P_soec_set_real = P_offer
            if P_offer > self.SOEC_MAX_CAPACITY:
                P_soec_set_real = self.SOEC_MAX_CAPACITY

            # Ramp Down Check (Q4: minutes 45-60)
            if minute_of_hour >= 45 and minute_of_hour < 60:
                future_offer_difference = P_future_offer - P_offer
                if future_offer_difference < 0:
                    P_soec_set_fut = min(P_future_offer, self.SOEC_MAX_CAPACITY)
                    P_soec_set_real = min(P_soec_set_real, P_soec_set_fut)

            # 2. SOEC Setpoint and Prediction
            # Use P_soec_set_real directly (matching manager.py which doesn't use discrete allocation)
            P_soec_set = P_soec_set_real
            
            # Predict P_soec_actual using SHADOW EXECUTION
            # This ensures we match soec_operator's complex logic (ramps, startups, etc.) exactly.
            P_soec_actual_predicted = P_soec_set
            shadow_success = False

            if SOEC_AVAILABLE and self._soec_cluster and hasattr(self._soec_cluster, 'soec_state'):
                try:
                    # Get current state from component
                    real_state = self._soec_cluster.soec_state
                    if real_state:
                        # Deep copy to avoid side effects
                        state_copy = copy.deepcopy(real_state)
                        
                        # Run shadow step
                        # Returns: (P_actual, updated_state, h2, steam)
                        P_actual_shadow, _, _, _ = soec_operator.run_soec_step(
                            P_soec_set, state_copy
                        )
                        P_soec_actual_predicted = P_actual_shadow
                        shadow_success = True
                except Exception as e:
                    logger.warning(f"Shadow prediction failed: {e}")
            
            if not shadow_success:
                # Fallback to simple ramp prediction
                current_power = getattr(self._soec_cluster, 'P_total_mw', 0.0)
                max_ramp = 1.44 
                if P_soec_set > current_power + max_ramp:
                    P_soec_actual_predicted = current_power + max_ramp
                elif P_soec_set < current_power - max_ramp:
                    P_soec_actual_predicted = current_power - max_ramp
            
            # Use predicted actual for surplus calculation
            P_soec_actual = P_soec_actual_predicted
            
            P_sold_surplus = P_offer - P_soec_actual
            
            # Ensure non-negative
            if P_sold_surplus < 0:
                P_sold_surplus = 0.0
                
            self.sold_power_mw = P_sold_surplus # Set state for this step

            # 3. PEM AND SOLD DISPATCH
            # P_sold_surplus is the "remainder" that SOEC cannot take.
            # We can try to put this into PEM or sell it.
            
            soec_offer_difference = P_sold_surplus
            
            PEM_SELL_FLAG_LOCAL = False

            # Arbitrage check for surplus
            if soec_offer_difference > 0.0 and current_spot_price > arbitrage_limit:
                PEM_SELL_FLAG_LOCAL = True
            
            # Priority 1: Sell Surplus if Advantageous
            if PEM_SELL_FLAG_LOCAL:
                P_sold = soec_offer_difference
                P_pem = 0.0
                if self.sell_decision == 0:
                    self.sell_decision = 1

            # Priority 2: H2 Production (PEM) if not selling
            elif soec_offer_difference > 0.0:
                # Phase 1.5 Alignment: Minimum Power Check
                if soec_offer_difference > self.MIN_PEM_POWER:
                    if soec_offer_difference <= self.MAX_PEM_POWER:
                        P_pem = soec_offer_difference
                        P_sold = 0.0
                    else:
                        # PEM Saturated
                        P_pem = self.MAX_PEM_POWER
                        P_sold = soec_offer_difference - self.MAX_PEM_POWER
                else:
                    # Below minimum power: Sell everything
                    P_pem = 0.0
                    P_sold = soec_offer_difference
            else:
                P_pem = 0.0
                P_sold = 0.0

            # 4. Update sell_decision for History/Graph
            if self.sell_decision == 0 and PEM_SELL_FLAG_LOCAL:
                self.sell_decision = 1
                
            # Log the decision
            # print(f"  Dispatch: SOEC={P_soec_actual:.2f}, PEM={P_pem:.2f}, Sold={P_sold:.2f}")

        # Store setpoints for application
        self.soec_setpoint_mw = P_soec_set
        self.pem_setpoint_mw = P_pem
        self.sold_power_mw = P_sold
        self.soec_actual_mw = P_soec_actual

    def calculate_soec_allocation(self, P_available: float) -> tuple[float, float]:
        """
        Allocates available power to SOEC modules based on discrete steps.
        
        Returns:
            (P_assigned, P_sold)
            P_assigned: Total power assigned to SOEC modules.
            P_sold: Remainder power that could not be assigned.
        """
        # Get dynamic parameters from cluster if available
        N_total = 6
        if self._soec_cluster:
            N_total = self._soec_cluster.num_modules
            
        # Determine P_nom based on configured max capacity
        # If efficient limit is on, P_nom will be 1.92 (11.52 / 6)
        # If not, it will be 2.4 (14.4 / 6)
        P_nom = 2.4
        if self.SOEC_MAX_CAPACITY > 0:
            P_nom = self.SOEC_MAX_CAPACITY / N_total
            
        P_min = 0.12
        step_first = 0.12
        step_up = 0.24
        
        # 1. Compute maximum number of full modules
        full_modules = min(int(P_available // P_nom), N_total)
        
        assigned_power = full_modules * P_nom
        remainder = P_available - assigned_power
        
        # 2. Handle remainder
        if full_modules == N_total:
            # All modules occupied: sell remainder
            return assigned_power, remainder
            
        # There is at least one free module
        if remainder < P_min:
            # Remainder too small to open a module
            return assigned_power, remainder
            
        # Remainder >= P_min: compute largest allowed P_next <= remainder
        # Phase 1.5 Alignment: Legacy allows continuous power assignment above P_min
        # We do NOT discretize here. The SOEC operator handles internal limits.
        
        P_next = remainder
        
        # Ensure P_next does not exceed P_nom (crucial if P_nom is 1.92)
        if P_next > P_nom:
             P_next = P_nom # Should be covered by full_modules logic, but for safety
        
        assigned_power += P_next
        sold = remainder - P_next
        
        return assigned_power, sold

    def _apply_setpoints(self) -> None:
        """Apply calculated setpoints to components."""

        # Set SOEC power setpoint
        if self._soec_cluster and hasattr(self._soec_cluster, 'set_power_setpoint'):
            self._soec_cluster.set_power_setpoint(self.soec_setpoint_mw)

        # Set PEM power setpoint
        if self._pem_electrolyzer and hasattr(self._pem_electrolyzer, 'set_power_input_mw'):
            if self.pem_setpoint_mw > 0:
                self._pem_electrolyzer.set_power_input_mw(self.pem_setpoint_mw)
            else:
                # FIX: Add safeguard for shutdown method
                if hasattr(self._pem_electrolyzer, 'shutdown'):
                    self._pem_electrolyzer.shutdown()
                else:
                    self._pem_electrolyzer.set_power_input_mw(0.0)

    def _aggregate_results(self) -> None:
        """Aggregate results from components."""

        total_h2_kg = 0.0

        # Read from SOEC
        if self._soec_cluster:
            h2_soec = getattr(self._soec_cluster, 'h2_output_kg', 0.0)
            total_h2_kg += h2_soec

            # Update previous power for next step
            self.previous_soec_power_mw = getattr(
                self._soec_cluster, 'P_total_mw', self.previous_soec_power_mw
            )

        # Read from PEM
        if self._pem_electrolyzer:
            h2_pem = getattr(self._pem_electrolyzer, 'h2_output_kg', 0.0)
            total_h2_kg += h2_pem

        # Update cumulative
        self.cumulative_production_kg += total_h2_kg
        
        # NOTE: We rely on P_sold calculated in step() based on PREDICTED SOEC power.
        # Since we implemented Shadow Prediction, P_soec_actual_predicted matches P_soec_actual (Real).
        # So P_sold is correct and consistent with manager.py logic.
        # We do NOT recalculate here to avoid using stale data (if Coordinator runs first).

        self.cumulative_sold_energy_mwh += (self.sold_power_mw * self.dt)

    def get_state(self) -> Dict[str, Any]:
        """Return coordinator state for monitoring."""
        return {
            **super().get_state(),
            'allocation_strategy': self.allocation_strategy.name,
            'soec_setpoint_mw': float(self.soec_setpoint_mw),
            'pem_setpoint_mw': float(self.pem_setpoint_mw),
            'sold_power_mw': float(self.sold_power_mw),
            'sell_decision': int(self.sell_decision),
            'force_sell_flag': bool(self.force_sell_flag),
            'cumulative_production_kg': float(self.cumulative_production_kg),
            'cumulative_sold_energy_mwh': float(self.cumulative_sold_energy_mwh),
            'P_offer_mw': float(getattr(self, 'current_offer_mw', 0.0)),
            # Phase 1 enhancement: Sell window tracking
            'sell_window_remaining': int(self.sell_window_remaining),
            'arbitrage_trigger_minute': self.arbitrage_trigger_minute
        }
