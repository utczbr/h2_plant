"""
Dispatch Strategy Framework for Power Allocation.

This module implements dispatch strategies that determine how to allocate
available power between electrolyzers (SOEC, PEM) and grid sales based
on energy prices and arbitrage opportunities.

Arbitrage Model:
    The dispatch decision compares profitability of:
    - **Hydrogen production**: Revenue from H₂ sales.
    - **Grid sales**: Revenue from selling electricity.

    Arbitrage threshold calculation:
    **P_threshold = P_PPA + (1000 / η_H₂) × Price_H₂**

    Where:
    - P_PPA: Power Purchase Agreement price (EUR/MWh).
    - η_H₂: Electrolyzer efficiency (kWh/kg H₂).
    - Price_H₂: Hydrogen selling price (EUR/kg).

    When spot price > threshold, selling electricity is more profitable.

Strategy Pattern:
    Abstract base class DispatchStrategy defines the interface.
    Concrete implementations provide specific logic:
    - ReferenceHybridStrategy: SOEC priority with PEM backup.
    - SoecOnlyStrategy: Single electrolyzer dispatch.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class DispatchInput:
    """
    Inputs required for dispatch decision.

    Attributes:
        minute (int): Current minute within simulation.
        P_offer (float): Available power offer from wind/grid (MW).
        P_future_offer (float): Lookahead power offer for ramp planning (MW).
        current_price (float): Current spot electricity price (EUR/MWh).
        soec_capacity_mw (float): SOEC electrolyzer capacity (MW).
        pem_max_power_mw (float): PEM electrolyzer maximum power (MW).
        soec_h2_kwh_kg (float): SOEC energy efficiency (kWh/kg H₂).
        pem_h2_kwh_kg (float): PEM energy efficiency (kWh/kg H₂).
    """
    minute: int
    P_offer: float
    P_future_offer: float
    current_price: float
    soec_capacity_mw: float
    pem_max_power_mw: float
    soec_h2_kwh_kg: float = 37.5
    pem_h2_kwh_kg: float = 50.0
    ppa_price_eur_mwh: float = 50.0 # Default
    h2_price_eur_kg: float = 9.6    # Default
    arbitrage_threshold_eur_mwh: Optional[float] = None  # Explicit override


@dataclass
class DispatchState:
    """
    State maintained between dispatch steps.

    Attributes:
        P_soec_prev (float): Previous SOEC power setpoint (MW).
        force_sell (bool): Flag indicating arbitrage sell mode active.
    """
    P_soec_prev: float = 0.0
    force_sell: bool = False


@dataclass
class DispatchResult:
    """
    Result of the dispatch decision.

    Attributes:
        P_soec (float): Power allocated to SOEC (MW).
        P_pem (float): Power allocated to PEM (MW).
        P_sold (float): Power sold to grid (MW).
        state_update (Dict[str, Any]): Updates to DispatchState.
    """
    P_soec: float
    P_pem: float
    P_sold: float
    state_update: Dict[str, Any]


class DispatchStrategy(ABC):
    """
    Abstract base class for dispatch strategies.

    Defines the interface for power allocation decisions between
    electrolyzers and grid sales.
    """

    @abstractmethod
    def decide(self, inputs: DispatchInput, state: DispatchState) -> DispatchResult:
        """
        Calculate dispatch setpoints based on inputs and state.

        Args:
            inputs (DispatchInput): Current system inputs and constraints.
            state (DispatchState): Persistent state from previous steps.

        Returns:
            DispatchResult: Power allocation and state updates.
        """
        pass


class ReferenceHybridStrategy(DispatchStrategy):
    """
    Hybrid SOEC/PEM dispatch strategy with dynamic price arbitrage.

    Prioritizes SOEC for base load, uses PEM for surplus power,
    and implements arbitrage logic to sell when profitable.

    Strategy Logic:
        1. **Arbitrage Check (Start of Hour)**: Compare selling vs producing.
        2. **Continuous Check**: Exit arbitrage if price drops.
        3. **Ramp Anticipation (Minute 45)**: Prepare for power changes.
        4. **Dispatch**: Allocate to SOEC → PEM → Grid.

    Robust to missing components (capacity=0 handled gracefully).

    Example:
        >>> strategy = ReferenceHybridStrategy()
        >>> result = strategy.decide(inputs, state)
        >>> print(f"SOEC: {result.P_soec} MW, PEM: {result.P_pem} MW")
    """

    def decide(self, inputs: DispatchInput, state: DispatchState) -> DispatchResult:
        """
        Calculate dispatch setpoints for hybrid SOEC/PEM topology.

        Args:
            inputs (DispatchInput): Current power offer, prices, and constraints.
            state (DispatchState): Previous SOEC power and arbitrage state.

        Returns:
            DispatchResult: Power allocation to SOEC, PEM, and grid.
        """
        # Economic parameters
        PPA_PRICE = inputs.ppa_price_eur_mwh
        H2_PRICE_KG = inputs.h2_price_eur_kg

        # Select reference efficiency for arbitrage calculation
        ref_h2_kwh_kg = inputs.soec_h2_kwh_kg
        if inputs.soec_capacity_mw <= 0 and inputs.pem_max_power_mw > 0:
            ref_h2_kwh_kg = inputs.pem_h2_kwh_kg

        # Calculate arbitrage threshold price (or use explicit override)
        if inputs.arbitrage_threshold_eur_mwh is not None:
            arbitrage_limit = inputs.arbitrage_threshold_eur_mwh
        else:
            h2_eq_price = (1000.0 / ref_h2_kwh_kg) * H2_PRICE_KG
            arbitrage_limit = PPA_PRICE + h2_eq_price

        P_soec_prev = state.P_soec_prev
        force_sell = state.force_sell
        minute_of_hour = inputs.minute % 60

        # Arbitrage check at start of hour
        if minute_of_hour == 0 and (inputs.P_offer - P_soec_prev) > 0:
            sale_profit = (inputs.P_offer - P_soec_prev) * 0.25 * (inputs.current_price - PPA_PRICE)
            E_surplus = (inputs.P_offer - P_soec_prev) * 0.25
            h2_profit = E_surplus * (1000.0 / ref_h2_kwh_kg) * H2_PRICE_KG

            force_sell = sale_profit > h2_profit

        # Exit arbitrage if price drops below threshold
        if force_sell and (inputs.current_price <= arbitrage_limit):
            force_sell = False

        # Ramp down anticipation at minute 45
        if minute_of_hour == 45:
            P_soec_set_fut = min(inputs.P_future_offer, inputs.soec_capacity_mw)
            if P_soec_prev > P_soec_set_fut:
                force_sell = False

        # Dispatch calculation
        P_soec = 0.0
        P_pem = 0.0
        P_sold = 0.0

        if force_sell:
            # Arbitrage mode: maintain previous SOEC, sell remainder
            P_soec = P_soec_prev
            P_pem = 0.0
            P_sold = inputs.P_offer - P_soec
        else:
            # Normal operation: prioritize SOEC
            P_soec_target = min(inputs.P_offer, inputs.soec_capacity_mw)

            # Apply ramp-down in final 15 minutes of hour
            if minute_of_hour >= 45:
                P_soec_fut = min(inputs.P_future_offer, inputs.soec_capacity_mw)
                if P_soec_target > P_soec_fut:
                    P_soec_target = P_soec_fut

            P_soec = P_soec_target
            surplus = inputs.P_offer - P_soec

            if surplus > 0:
                if inputs.current_price > arbitrage_limit:
                    # Sell surplus at high price
                    P_sold = surplus
                    P_pem = 0.0
                else:
                    # Use surplus for PEM production
                    P_pem = min(surplus, inputs.pem_max_power_mw)
                    P_sold = surplus - P_pem
            else:
                P_pem = 0.0
                P_sold = 0.0

        return DispatchResult(
            P_soec=P_soec,
            P_pem=P_pem,
            P_sold=P_sold,
            state_update={'force_sell': force_sell}
        )


class SoecOnlyStrategy(DispatchStrategy):
    """
    SOEC-only dispatch strategy with dynamic price arbitrage.

    Allocates power to single SOEC electrolyzer with surplus
    sold to grid. No PEM backup available.

    Strategy Logic:
        Same arbitrage and ramp logic as ReferenceHybridStrategy,
        but surplus power is always sold (no PEM fallback).

    Example:
        >>> strategy = SoecOnlyStrategy()
        >>> result = strategy.decide(inputs, state)
        >>> print(f"SOEC: {result.P_soec} MW, Sold: {result.P_sold} MW")
    """

    def decide(self, inputs: DispatchInput, state: DispatchState) -> DispatchResult:
        """
        Calculate dispatch setpoints for SOEC-only topology.

        Args:
            inputs (DispatchInput): Current power offer, prices, and constraints.
            state (DispatchState): Previous SOEC power and arbitrage state.

        Returns:
            DispatchResult: Power allocation to SOEC and grid (P_pem always 0).
        """
        PPA_PRICE = inputs.ppa_price_eur_mwh
        H2_PRICE_KG = inputs.h2_price_eur_kg

        ref_h2_kwh_kg = inputs.soec_h2_kwh_kg
        
        # Calculate arbitrage threshold price (or use explicit override)
        if inputs.arbitrage_threshold_eur_mwh is not None:
            arbitrage_limit = inputs.arbitrage_threshold_eur_mwh
        else:
            h2_eq_price = (1000.0 / ref_h2_kwh_kg) * H2_PRICE_KG
            arbitrage_limit = PPA_PRICE + h2_eq_price

        P_soec_prev = state.P_soec_prev
        force_sell = state.force_sell
        minute_of_hour = inputs.minute % 60

        # Arbitrage check at start of hour
        if minute_of_hour == 0 and (inputs.P_offer - P_soec_prev) > 0:
            sale_profit = (inputs.P_offer - P_soec_prev) * 0.25 * (inputs.current_price - PPA_PRICE)
            E_surplus = (inputs.P_offer - P_soec_prev) * 0.25
            h2_profit = E_surplus * (1000.0 / ref_h2_kwh_kg) * H2_PRICE_KG

            force_sell = sale_profit > h2_profit

        if force_sell and (inputs.current_price <= arbitrage_limit):
            force_sell = False

        if minute_of_hour == 45:
            P_soec_set_fut = min(inputs.P_future_offer, inputs.soec_capacity_mw)
            if P_soec_prev > P_soec_set_fut:
                force_sell = False

        P_soec = 0.0
        P_sold = 0.0

        if force_sell:
            P_soec = P_soec_prev
            P_sold = inputs.P_offer - P_soec
        else:
            P_soec_target = min(inputs.P_offer, inputs.soec_capacity_mw)

            if minute_of_hour >= 45:
                P_soec_fut = min(inputs.P_future_offer, inputs.soec_capacity_mw)
                if P_soec_target > P_soec_fut:
                    P_soec_target = P_soec_fut

            P_soec = P_soec_target
            surplus = inputs.P_offer - P_soec

            if surplus > 0:
                P_sold = surplus

        return DispatchResult(
            P_soec=P_soec,
            P_pem=0.0,
            P_sold=P_sold,
            state_update={'force_sell': force_sell}
        )
