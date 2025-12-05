from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class DispatchInput:
    """Inputs required for dispatch decision."""
    minute: int
    P_offer: float
    P_future_offer: float
    current_price: float
    
    # Component Constraints (Capacity, Max Power)
    # If a component is missing, its capacity/max_power should be 0
    soec_capacity_mw: float
    pem_max_power_mw: float
    
    # Efficiency Parameters (for Arbitrage)
    soec_h2_kwh_kg: float = 37.5
    pem_h2_kwh_kg: float = 50.0 # Default fallback

@dataclass
class DispatchState:
    """State maintained between dispatch steps."""
    P_soec_prev: float = 0.0
    force_sell: bool = False

@dataclass
class DispatchResult:
    """Result of the dispatch decision."""
    P_soec: float
    P_pem: float
    P_sold: float
    state_update: Dict[str, Any] # Updates to DispatchState

class DispatchStrategy(ABC):
    """Abstract base class for dispatch strategies."""
    
    @abstractmethod
    def decide(self, inputs: DispatchInput, state: DispatchState) -> DispatchResult:
        """Calculate dispatch setpoints based on inputs and state."""
        pass

class ReferenceHybridStrategy(DispatchStrategy):
    """
    Replicates the reference 'manager.py' logic.
    Prioritizes SOEC, uses PEM for surplus, and implements specific arbitrage logic.
    Robust to missing components (capacity=0).
    """
    
    def decide(self, inputs: DispatchInput, state: DispatchState) -> DispatchResult:
        # Constants (should ideally come from config, but fixed for parity now)
        PPA_PRICE = 50.0 # EUR/MWh
        H2_PRICE_KG = 9.6 # EUR/kg
        
        # 1. Determine Effective Efficiency for Arbitrage
        # Reference uses SOEC efficiency for the threshold.
        # If SOEC is missing (capacity=0), we should logically use PEM efficiency,
        # but to maintain STRICT parity when SOEC exists, we use SOEC's.
        
        ref_h2_kwh_kg = inputs.soec_h2_kwh_kg
        if inputs.soec_capacity_mw <= 0 and inputs.pem_max_power_mw > 0:
             # Fallback to PEM efficiency if SOEC is absent
             ref_h2_kwh_kg = inputs.pem_h2_kwh_kg
             
        # Arbitrage Threshold
        h2_eq_price = (1000.0 / ref_h2_kwh_kg) * H2_PRICE_KG
        arbitrage_limit = PPA_PRICE + h2_eq_price
        
        # State
        P_soec_prev = state.P_soec_prev
        force_sell = state.force_sell
        
        minute_of_hour = inputs.minute % 60
        
        # --- LOGIC START (Matches manager.py) ---
        
        # 1. Arbitrage Check (Start of Hour)
        # Only checks if we have power to sell (P_offer > P_soec_prev)
        if minute_of_hour == 0 and (inputs.P_offer - P_soec_prev) > 0:
            # Check profitability
            sale_profit = (inputs.P_offer - P_soec_prev) * 0.25 * (inputs.current_price - PPA_PRICE)
            
            E_surplus = (inputs.P_offer - P_soec_prev) * 0.25
            h2_profit = E_surplus * (1000.0 / ref_h2_kwh_kg) * H2_PRICE_KG
            
            if sale_profit > h2_profit:
                force_sell = True
            else:
                force_sell = False

        # 2. Continuous Checks
        if force_sell and (inputs.current_price <= arbitrage_limit):
            force_sell = False
            
        # 3. Ramp Down Anticipation (Minute 45)
        if minute_of_hour == 45:
            # Note: Reference uses SOEC_MAX_CAPACITY here.
            # If SOEC is missing, this is 0, so P_soec_set_fut is 0.
            P_soec_set_fut = min(inputs.P_future_offer, inputs.soec_capacity_mw)
            if P_soec_prev > P_soec_set_fut:
                force_sell = False
                
        # --- DISPATCH CALCULATION ---
        P_soec = 0.0
        P_pem = 0.0
        P_sold = 0.0
        
        if force_sell:
            P_soec = P_soec_prev # Maintain previous power
            P_pem = 0.0
            P_sold = inputs.P_offer - P_soec
        else:
            # Normal Operation
            P_soec_target = min(inputs.P_offer, inputs.soec_capacity_mw)
            
            # Ramp Down Logic (45-60 min)
            if minute_of_hour >= 45:
                P_soec_fut = min(inputs.P_future_offer, inputs.soec_capacity_mw)
                if P_soec_target > P_soec_fut:
                    P_soec_target = P_soec_fut
            
            P_soec = P_soec_target
            
            # Surplus to PEM or Grid
            surplus = inputs.P_offer - P_soec
            
            if surplus > 0:
                # Check if we should sell surplus (Arbitrage)
                if inputs.current_price > arbitrage_limit:
                    P_sold = surplus
                    P_pem = 0.0
                else:
                    # Send to PEM
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
    Strategy for SOEC-only topology with dynamic price arbitrage.
    """
    
    def decide(self, inputs: DispatchInput, state: DispatchState) -> DispatchResult:
        # Constants
        PPA_PRICE = 50.0 # EUR/MWh
        H2_PRICE_KG = 9.6 # EUR/kg
        
        # 1. Arbitrage Threshold
        ref_h2_kwh_kg = inputs.soec_h2_kwh_kg
        h2_eq_price = (1000.0 / ref_h2_kwh_kg) * H2_PRICE_KG
        arbitrage_limit = PPA_PRICE + h2_eq_price
        
        # State
        P_soec_prev = state.P_soec_prev
        force_sell = state.force_sell
        
        minute_of_hour = inputs.minute % 60
        
        # --- LOGIC START ---
        
        # 1. Arbitrage Check (Start of Hour)
        if minute_of_hour == 0 and (inputs.P_offer - P_soec_prev) > 0:
            sale_profit = (inputs.P_offer - P_soec_prev) * 0.25 * (inputs.current_price - PPA_PRICE)
            E_surplus = (inputs.P_offer - P_soec_prev) * 0.25
            h2_profit = E_surplus * (1000.0 / ref_h2_kwh_kg) * H2_PRICE_KG
            
            if sale_profit > h2_profit:
                force_sell = True
            else:
                force_sell = False

        # 2. Continuous Checks
        if force_sell and (inputs.current_price <= arbitrage_limit):
            force_sell = False
            
        # 3. Ramp Down Anticipation (Minute 45)
        if minute_of_hour == 45:
            P_soec_set_fut = min(inputs.P_future_offer, inputs.soec_capacity_mw)
            if P_soec_prev > P_soec_set_fut:
                force_sell = False
                
        # --- DISPATCH CALCULATION ---
        P_soec = 0.0
        P_pem = 0.0 # No PEM in this strategy
        P_sold = 0.0
        
        if force_sell:
            P_soec = P_soec_prev # Maintain previous power
            P_sold = inputs.P_offer - P_soec
        else:
            # Normal Operation
            P_soec_target = min(inputs.P_offer, inputs.soec_capacity_mw)
            
            # Ramp Down Logic (45-60 min)
            if minute_of_hour >= 45:
                P_soec_fut = min(inputs.P_future_offer, inputs.soec_capacity_mw)
                if P_soec_target > P_soec_fut:
                    P_soec_target = P_soec_fut
            
            P_soec = P_soec_target
            
            # Surplus to Grid (since no PEM)
            surplus = inputs.P_offer - P_soec
            
            if surplus > 0:
                # Always sell surplus if SOEC is full or constrained
                P_sold = surplus
            else:
                P_sold = 0.0
                
        return DispatchResult(
            P_soec=P_soec,
            P_pem=0.0,
            P_sold=P_sold,
            state_update={'force_sell': force_sell}
        )
