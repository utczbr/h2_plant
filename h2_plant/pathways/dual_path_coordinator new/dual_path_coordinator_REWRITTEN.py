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
        demand_scheduler_id: str = 'demand_scheduler'
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
        self.MIN_PEM_POWER = 0.5  # MW (efficiency cutoff)
        self.PPA_PRICE = 50.0  # EUR/MWh
        self.H2_PRICE_KG = 9.6  # EUR/kg
        self.SOEC_KWH_KG = 37.5  # kWh/kg
        self.ARBITRAGE_THRESHOLD = 306.0  # EUR/MWh (calculated)

        # State Variables (matching manager.py)
        self.force_sell_flag = False
        self.previous_soec_power_mw = 0.0
        self.sell_decision = 0  # 0: H2, 1: Sell

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
            minute_of_hour = self._environment.get_minute_of_hour(t)

        # Get future offer (for minute 45 ramp-down check)
        P_future_offer = P_offer_mw
        if minute_of_hour >= 45 and self._environment:
            minutes_to_next_hour = 60 - minute_of_hour
            # Future wind power
            P_future_offer = self._environment.get_future_power(minutes_to_next_hour)

        # Execute arbitration logic (exact reference implementation)
        self._execute_dispatch_logic(
            t, P_offer_mw, P_future_offer, spot_price_eur_mwh, minute_of_hour
        )

        # Set component setpoints
        self._apply_setpoints()

        # Aggregate results
        self._aggregate_results()

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

        # --- ARBITRAGE STEP (ONLY at minute 0 AND with RAMP UP) ---
        # Lines 186-204 from manager.py
        if minute_of_hour == 0:
            offer_previous_difference = P_offer - self.previous_soec_power_mw

            if offer_previous_difference > 0.0:  # Ramp Up
                P_surplus = offer_previous_difference
                time_h_arbitrage = 15.0 / 60.0  # 15 minutes

                # Sales Profit
                sale_profit = P_surplus * time_h_arbitrage * (
                    current_spot_price - self.PPA_PRICE
                )

                # H2 Profit
                E_surplus_mwh = P_surplus * time_h_arbitrage
                mass_kg = E_surplus_mwh * (1000.0 / self.SOEC_KWH_KG)
                h2_profit = mass_kg * self.H2_PRICE_KG

                # Decision
                if sale_profit > h2_profit:
                    self.force_sell_flag = True
                    logger.info(
                        f"t={t:.2f}: Arbitrage triggered - "
                        f"Sale profit ({sale_profit:.2f}) > H2 profit ({h2_profit:.2f})"
                    )
                else:
                    self.force_sell_flag = False

        # --- ARBITRAGE CONTROL BLOCK (Continuous) ---
        # Lines 206-217 from manager.py

        # 1. Reset if price drops below arbitrage limit
        if self.force_sell_flag and (current_spot_price <= arbitrage_limit):
            self.force_sell_flag = False
            logger.info(f"t={t:.2f}: Force sell reset - price dropped")

        # 2. Reset at minute 45 if future ramp down (OPERATIONAL PRIORITY)
        if minute_of_hour == 45:
            future_offer_difference = P_future_offer - P_offer
            if future_offer_difference < 0:
                self.force_sell_flag = False
                logger.info(f"t={t:.2f}: Force sell reset - future ramp down")

        # 3. Set sell decision flag
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

            P_soec_set = P_soec_set_real

            # 2. Predict SOEC actual power (SOEC ramps at 0.24 MW/min)
            # In reality, soec_operator handles this, but we need prediction
            # for surplus calculation BEFORE soec_operator.step() is called
            P_soec_actual = self.previous_soec_power_mw
            if hasattr(self, '_soec_cluster') and self._soec_cluster:
                # Use actual SOEC state if available
                P_soec_actual = getattr(self._soec_cluster, 'P_total_mw', self.previous_soec_power_mw)

            # 3. PEM AND SOLD DISPATCH
            soec_offer_difference = P_offer - P_soec_actual

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
                # Check minimum power threshold
                if soec_offer_difference < self.MIN_PEM_POWER:
                    # Too low for efficient operation - sell instead
                    P_pem = 0.0
                    P_sold = soec_offer_difference
                elif soec_offer_difference <= self.MAX_PEM_POWER:
                    P_pem = soec_offer_difference
                    P_sold = 0.0
                else:
                    P_pem = self.MAX_PEM_POWER
                    P_sold = soec_offer_difference - self.MAX_PEM_POWER
            else:
                P_pem = 0.0
                P_sold = 0.0

        # Store setpoints for application
        self.soec_setpoint_mw = P_soec_set
        self.pem_setpoint_mw = P_pem
        self.sold_power_mw = P_sold
        self.soec_actual_mw = P_soec_actual

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
                self._pem_electrolyzer.shutdown()

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
            'cumulative_production_kg': float(self.cumulative_production_kg),
            'cumulative_sold_energy_mwh': float(self.cumulative_sold_energy_mwh)
        }
