"""
Isolated production pathway orchestration.

A production pathway encapsulates:
- Production source (Electrolyzer or ATR)
- Low-pressure buffer storage
- Compression to high-pressure storage
- High-pressure delivery storage
"""

from typing import Dict, Any, Optional
import logging

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.enums import ProductionState, TankState
from h2_plant.core.exceptions import ComponentNotFoundError

logger = logging.getLogger(__name__)


class IsolatedProductionPath(Component):
    """
    Orchestrates a complete isolated production pathway.
    
    Manages the flow: Production Source → LP Storage → Compressor → HP Storage
    
    Features:
    - Automatic pressure management (LP to HP transfer)
    - Production-storage coordination
    - Pathway-level state tracking
    - Economic metrics (cost per kg H2)
    
    Example:
        # Create electrolyzer pathway
        path = IsolatedProductionPath(
            pathway_id='electrolyzer',
            source_id='electrolyzer',
            lp_storage_id='elec_lp_tanks',
            hp_storage_id='elec_hp_tanks',
            compressor_id='elec_filling_compressor'
        )
        
        # Set production target
        path.production_target_kg_h = 50.0
        
        # Execute timestep
        path.step(t)
        
        # Read outputs
        h2_produced = path.h2_produced_kg
        h2_available = path.h2_available_kg
    """
    
    def __init__(
        self,
        pathway_id: str,
        source_id: str,
        lp_storage_id: str,
        hp_storage_id: str,
        compressor_id: str,
        lp_to_hp_threshold_kg: float = 100.0,
        hp_min_reserve_kg: float = 50.0
    ):
        """
        Initialize isolated production pathway.
        
        Args:
            pathway_id: Unique pathway identifier
            source_id: Production source component ID in registry
            lp_storage_id: Low-pressure storage component ID
            hp_storage_id: High-pressure storage component ID
            compressor_id: Filling compressor component ID
            lp_to_hp_threshold_kg: Minimum LP mass before transfer (kg)
            hp_min_reserve_kg: Minimum HP reserve to maintain (kg)
        """
        super().__init__()
        
        self.pathway_id = pathway_id
        self.source_id = source_id
        self.lp_storage_id = lp_storage_id
        self.hp_storage_id = hp_storage_id
        self.compressor_id = compressor_id
        self.lp_to_hp_threshold_kg = lp_to_hp_threshold_kg
        self.hp_min_reserve_kg = hp_min_reserve_kg
        
        # Component references (populated during initialize)
        self._source: Optional[Component] = None
        self._lp_storage: Optional[Component] = None
        self._hp_storage: Optional[Component] = None
        self._compressor: Optional[Component] = None
        
        # Inputs (set by coordinator before each step)
        self.production_target_kg_h = 0.0
        self.discharge_demand_kg = 0.0
        
        # Outputs (read by coordinator after each step)
        self.h2_produced_kg = 0.0
        self.h2_stored_lp_kg = 0.0
        self.h2_stored_hp_kg = 0.0
        self.h2_available_kg = 0.0  # HP storage available for delivery
        self.h2_delivered_kg = 0.0
        
        # State tracking
        self.cumulative_production_kg = 0.0
        self.cumulative_delivery_kg = 0.0
        self.cumulative_compression_energy_kwh = 0.0
        self.cumulative_production_energy_kwh = 0.0
        self.cumulative_cost = 0.0
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize pathway and resolve component references."""
        super().initialize(dt, registry)
        
        # Resolve component references
        try:
            self._source = registry.get(self.source_id)
            self._lp_storage = registry.get(self.lp_storage_id)
            self._hp_storage = registry.get(self.hp_storage_id)
            self._compressor = registry.get(self.compressor_id)
        except ComponentNotFoundError as e:
            raise ValueError(f"Pathway {self.pathway_id} component not found: {e}")
        
        logger.info(f"Initialized pathway '{self.pathway_id}'")
    
    def get_state(self) -> Dict[str, Any]:
        """Return pathway state for monitoring."""
        return {
            **super().get_state(),
            'pathway_id': self.pathway_id,
            'production_target_kg_h': float(self.production_target_kg_h),
            'h2_produced_kg': float(self.h2_produced_kg),
            'h2_stored_lp_kg': float(self.h2_stored_lp_kg),
            'h2_stored_hp_kg': float(self.h2_stored_hp_kg),
            'h2_available_kg': float(self.h2_available_kg),
            'h2_delivered_kg': float(self.h2_delivered_kg),
            'cumulative_production_kg': float(self.cumulative_production_kg),
            'cumulative_delivery_kg': float(self.cumulative_delivery_kg),
            'cumulative_cost': float(self.cumulative_cost),
            'specific_cost_per_kg': float(
                self.cumulative_cost / self.cumulative_production_kg
                if self.cumulative_production_kg > 0 else 0.0
            )
        }

    def step(self, t: float) -> None:
        """
        Execute single timestep of pathway orchestration.
        
        Workflow:
        1. Set source inputs and step the source.
        2. Read production output and fill LP storage.
        3. Make compression decisions, step the compressor, and fill HP storage.
        4. Fulfill discharge demand from HP storage.
        5. Step all storage components to update their internal states (e.g., pressure).
        
        Args:
            t: Current simulation time (hours)
        """
        super().step(t)
        
        # 1. Set source inputs and step the source
        self._set_production_inputs()
        self._source.step(t)
        self.h2_produced_kg = getattr(self._source, 'h2_output_kg', 0.0)

        # 2. Fill LP storage with new production
        self._fill_lp_storage()

        # 3. Execute compression from LP to HP
        self._run_compression_and_fill_hp(t)
        
        # 4. Fulfill any discharge demand
        self._manage_hp_storage()

        # 5. Step storage components to update pressures, etc.
        self._lp_storage.step(t)
        self._hp_storage.step(t)

        # 6. Update cumulative stats for the pathway
        self._update_state()

    def _set_production_inputs(self) -> None:
        """Set production source inputs based on target."""
        if hasattr(self._source, 'power_input_mw'):
            # Electrolyzer - convert kg/h target to power
            # Simplified: 50 kWh/kg at 65% efficiency ≈ 0.077 MW per kg/h
            self._source.power_input_mw = self.production_target_kg_h * 0.077
        
        elif hasattr(self._source, 'ng_flow_rate_kg_h'):
            # ATR - convert kg/h target to NG flow
            # Simplified: 1 kg NG → 0.28 kg H2 at 75% efficiency
            self._source.ng_flow_rate_kg_h = self.production_target_kg_h / 0.28 if self.production_target_kg_h > 0 else 0.0
    
    def _fill_lp_storage(self) -> None:
        """Store production in LP tanks."""
        if self.h2_produced_kg > 0:
            stored, overflow = self._lp_storage.fill(self.h2_produced_kg)
            
            if overflow > 0:
                logger.warning(
                    f"Pathway {self.pathway_id}: LP storage full, "
                    f"venting {overflow:.2f} kg H2"
                )
        self.h2_stored_lp_kg = self._lp_storage.get_total_mass()

    def _run_compression_and_fill_hp(self, t: float) -> None:
        """Check for LP to HP transfer, run compressor, and fill HP tank."""
        self._compressor.transfer_mass_kg = 0.0 # Reset from previous step
        
        if self.h2_stored_lp_kg >= self.lp_to_hp_threshold_kg:
            # Determine transfer amount
            transfer_mass = min(
                self.h2_stored_lp_kg,
                self._compressor.max_flow_kg_h * self.dt
            )
            
            hp_available_capacity = self._hp_storage.get_available_capacity()
            transfer_mass = min(transfer_mass, hp_available_capacity)
            
            if transfer_mass > 0:
                # Set compressor's input for this step
                lp_discharged = self._lp_storage.discharge(transfer_mass)
                self._compressor.transfer_mass_kg = lp_discharged
        
        # Step the compressor regardless to update its state
        self._compressor.step(t)
        
        # Now, get the actual mass transferred by the compressor
        actual_mass_transferred = getattr(self._compressor, 'actual_mass_transferred_kg', 0.0)
        
        if actual_mass_transferred > 0:
            # Fill HP tank with the mass that was actually compressed
            self._hp_storage.fill(actual_mass_transferred)

        # Track compression energy
        self.cumulative_compression_energy_kwh += getattr(self._compressor, 'energy_consumed_kwh', 0.0)
        
        # Update our view of the storage levels
        self.h2_stored_lp_kg = self._lp_storage.get_total_mass()
        self.h2_stored_hp_kg = self._hp_storage.get_total_mass()
    
    def _manage_hp_storage(self) -> None:
        """Manage HP storage and deliver based on demand."""
        # Update HP storage state
        self.h2_stored_hp_kg = self._hp_storage.get_total_mass()
        
        # Available for delivery (total minus reserve)
        self.h2_available_kg = max(0.0, self.h2_stored_hp_kg - self.hp_min_reserve_kg)
        
        # Deliver if demand exists
        if self.discharge_demand_kg > 0:
            # Don't violate minimum reserve
            deliverable = min(self.discharge_demand_kg, self.h2_available_kg)
            
            if deliverable > 0:
                self.h2_delivered_kg = self._hp_storage.discharge(deliverable)
                self.cumulative_delivery_kg += self.h2_delivered_kg
            else:
                self.h2_delivered_kg = 0.0
        else:
            self.h2_delivered_kg = 0.0
    
    def _update_state(self) -> None:
        """Update cumulative state tracking."""
        # Track production energy if available
        self.cumulative_production_energy_kwh = getattr(self._source, 'cumulative_energy_kwh', 0.0)
        
        # Track cost if available
        self.cumulative_cost = getattr(self._source, 'cumulative_cost', 0.0)
    
    def get_production_cost_per_kg(self) -> float:
        """
        Calculate specific production cost ($/kg H2).
        
        Returns:
            Cost per kg of hydrogen produced
        """
        if self.cumulative_production_kg > 0:
            return self.cumulative_cost / self.cumulative_production_kg
        return 0.0
    
    def get_total_energy_per_kg(self) -> float:
        """
        Calculate specific energy consumption (kWh/kg H2).
        
        Includes production energy and compression energy.
        
        Returns:
            Energy per kg of hydrogen produced
        """
        if self.cumulative_production_kg > 0:
            total_energy = (
                self.cumulative_production_energy_kwh +
                self.cumulative_compression_energy_kwh
            )
            return total_energy / self.cumulative_production_kg
        return 0.0
