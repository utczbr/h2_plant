"""
Isolated Production Pathway Orchestration.

This module provides pathway-level coordination for isolated hydrogen
production chains, managing the complete flow from production source
through compression to high-pressure storage.

Pathway Architecture:
    ```
    Production Source → LP Storage → Compressor → HP Storage → Delivery
    (Electrolyzer/ATR)   (Buffer)      (C-7)       (350 bar)
    ```

Use Cases:
    - Modular pathway testing and comparison.
    - Pathway-specific KPI tracking (cost, energy, emissions).
    - Dual-path coordination via DualPathCoordinator.

Component Lifecycle:
    The IsolatedProductionPath implements the Component Lifecycle
    Contract (Layer 1) and manages its child components:
    - `initialize()`: Resolves component references from registry.
    - `step()`: Orchestrates causal flow through pathway.
    - `get_state()`: Returns pathway-level aggregated metrics.
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

    Manages the material flow from production source through LP buffer
    storage, compression, and HP delivery storage.

    Features:
        - Automatic pressure management (LP → HP transfer).
        - Production-storage coordination.
        - Pathway-level economic tracking.
        - Reserve management for HP storage.

    Attributes:
        pathway_id (str): Unique pathway identifier.
        source_id (str): Production source component ID.
        lp_storage_id (str): LP buffer storage component ID.
        hp_storage_id (str): HP delivery storage component ID.
        compressor_id (str): Filling compressor component ID.
        lp_to_hp_threshold_kg (float): LP mass threshold for transfer.
        hp_min_reserve_kg (float): Minimum HP reserve to maintain.

    Example:
        >>> path = IsolatedProductionPath(
        ...     pathway_id='electrolyzer',
        ...     source_id='electrolyzer',
        ...     lp_storage_id='elec_lp_tanks',
        ...     hp_storage_id='elec_hp_tanks',
        ...     compressor_id='elec_filling_compressor'
        ... )
        >>> path.production_target_kg_h = 50.0
        >>> path.step(t)
        >>> print(f"Produced: {path.h2_produced_kg} kg")
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
            pathway_id (str): Unique pathway identifier.
            source_id (str): Production source component ID in registry.
            lp_storage_id (str): Low-pressure storage component ID.
            hp_storage_id (str): High-pressure storage component ID.
            compressor_id (str): Filling compressor component ID.
            lp_to_hp_threshold_kg (float): Minimum LP mass before transfer (kg).
                Default: 100.0.
            hp_min_reserve_kg (float): Minimum HP reserve to maintain (kg).
                Default: 50.0.
        """
        super().__init__()

        self.pathway_id = pathway_id
        self.source_id = source_id
        self.lp_storage_id = lp_storage_id
        self.hp_storage_id = hp_storage_id
        self.compressor_id = compressor_id
        self.lp_to_hp_threshold_kg = lp_to_hp_threshold_kg
        self.hp_min_reserve_kg = hp_min_reserve_kg

        # Component references (resolved during initialize)
        self._source: Optional[Component] = None
        self._lp_storage: Optional[Component] = None
        self._hp_storage: Optional[Component] = None
        self._compressor: Optional[Component] = None

        # Inputs (set by coordinator)
        self.production_target_kg_h = 0.0
        self.discharge_demand_kg = 0.0

        # Outputs (read by coordinator)
        self.h2_produced_kg = 0.0
        self.h2_stored_lp_kg = 0.0
        self.h2_stored_hp_kg = 0.0
        self.h2_available_kg = 0.0
        self.h2_delivered_kg = 0.0

        # Cumulative tracking
        self.cumulative_production_kg = 0.0
        self.cumulative_delivery_kg = 0.0
        self.cumulative_compression_energy_kwh = 0.0
        self.cumulative_production_energy_kwh = 0.0
        self.cumulative_cost = 0.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Initialize pathway and resolve component references.

        Fulfills the Component Lifecycle Contract by resolving
        component references from the registry.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Component registry.

        Raises:
            ValueError: If any pathway component not found in registry.
        """
        super().initialize(dt, registry)

        try:
            self._source = registry.get(self.source_id)
            self._lp_storage = registry.get(self.lp_storage_id)
            self._hp_storage = registry.get(self.hp_storage_id)
            self._compressor = registry.get(self.compressor_id)
        except ComponentNotFoundError as e:
            raise ValueError(f"Pathway {self.pathway_id} component not found: {e}")

        logger.info(f"Initialized pathway '{self.pathway_id}'")

    def get_state(self) -> Dict[str, Any]:
        """
        Return pathway state for monitoring.

        Fulfills the Component Lifecycle Contract by providing
        pathway-level aggregated metrics.

        Returns:
            Dict[str, Any]: Pathway state including:
                - Storage levels (LP, HP).
                - Production and delivery metrics.
                - Cumulative cost and specific cost per kg.
        """
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

        Fulfills the Component Lifecycle Contract with causal execution:
            1. Set source inputs and step the source.
            2. Fill LP storage with production.
            3. Run compression and fill HP storage.
            4. Fulfill discharge demand from HP storage.
            5. Step storage components to update pressures.
            6. Update cumulative tracking.

        Args:
            t (float): Current simulation time in hours.
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

        # 5. Step storage components to update pressures
        self._lp_storage.step(t)
        self._hp_storage.step(t)

        # 6. Update cumulative stats
        self._update_state()

    def _set_production_inputs(self) -> None:
        """
        Set production source inputs based on target.

        Converts kg/h production target to source-specific inputs:
        - Electrolyzer: Power input (MW).
        - ATR: Natural gas flow rate (kg/h).
        """
        if hasattr(self._source, 'power_input_mw'):
            # Electrolyzer: ~50 kWh/kg at 65% efficiency ≈ 0.077 MW per kg/h
            self._source.power_input_mw = self.production_target_kg_h * 0.077

        elif hasattr(self._source, 'ng_flow_rate_kg_h'):
            # ATR: 1 kg NG → 0.28 kg H₂ at 75% efficiency
            self._source.ng_flow_rate_kg_h = self.production_target_kg_h / 0.28 if self.production_target_kg_h > 0 else 0.0

    def _fill_lp_storage(self) -> None:
        """
        Store production in LP tanks.

        Logs warning if LP storage overflows (venting condition).
        """
        if self.h2_produced_kg > 0:
            stored, overflow = self._lp_storage.fill(self.h2_produced_kg)

            if overflow > 0:
                logger.warning(
                    f"Pathway {self.pathway_id}: LP storage full, "
                    f"venting {overflow:.2f} kg H2"
                )
        self.h2_stored_lp_kg = self._lp_storage.get_total_mass()

    def _run_compression_and_fill_hp(self, t: float) -> None:
        """
        Execute LP → HP transfer via compressor.

        Transfer occurs when LP mass exceeds threshold and HP
        has available capacity.

        Args:
            t (float): Current simulation time.
        """
        self._compressor.transfer_mass_kg = 0.0

        if self.h2_stored_lp_kg >= self.lp_to_hp_threshold_kg:
            # Determine transfer amount
            transfer_mass = min(
                self.h2_stored_lp_kg,
                self._compressor.max_flow_kg_h * self.dt
            )

            hp_available_capacity = self._hp_storage.get_available_capacity()
            transfer_mass = min(transfer_mass, hp_available_capacity)

            if transfer_mass > 0:
                lp_discharged = self._lp_storage.discharge(transfer_mass)
                self._compressor.transfer_mass_kg = lp_discharged

        # Always step compressor to update state
        self._compressor.step(t)

        # Fill HP with actually compressed mass
        actual_mass_transferred = getattr(self._compressor, 'actual_mass_transferred_kg', 0.0)

        if actual_mass_transferred > 0:
            self._hp_storage.fill(actual_mass_transferred)

        # Track compression energy
        self.cumulative_compression_energy_kwh += getattr(self._compressor, 'energy_consumed_kwh', 0.0)

        # Update storage level views
        self.h2_stored_lp_kg = self._lp_storage.get_total_mass()
        self.h2_stored_hp_kg = self._hp_storage.get_total_mass()

    def _manage_hp_storage(self) -> None:
        """
        Manage HP storage and fulfill discharge demand.

        Maintains minimum reserve to ensure operational flexibility.
        """
        self.h2_stored_hp_kg = self._hp_storage.get_total_mass()

        # Available for delivery (total minus reserve)
        self.h2_available_kg = max(0.0, self.h2_stored_hp_kg - self.hp_min_reserve_kg)

        # Deliver if demand exists
        if self.discharge_demand_kg > 0:
            deliverable = min(self.discharge_demand_kg, self.h2_available_kg)

            if deliverable > 0:
                self.h2_delivered_kg = self._hp_storage.discharge(deliverable)
                self.cumulative_delivery_kg += self.h2_delivered_kg
            else:
                self.h2_delivered_kg = 0.0
        else:
            self.h2_delivered_kg = 0.0

    def _update_state(self) -> None:
        """Update cumulative state tracking from source component."""
        self.cumulative_production_energy_kwh = getattr(self._source, 'cumulative_energy_kwh', 0.0)
        self.cumulative_cost = getattr(self._source, 'cumulative_cost', 0.0)

    def get_production_cost_per_kg(self) -> float:
        """
        Calculate specific production cost.

        Returns:
            float: Cost per kg of hydrogen produced (€/kg).
        """
        if self.cumulative_production_kg > 0:
            return self.cumulative_cost / self.cumulative_production_kg
        return 0.0

    def get_total_energy_per_kg(self) -> float:
        """
        Calculate specific energy consumption.

        Includes both production energy and compression energy.

        Returns:
            float: Energy per kg of hydrogen produced (kWh/kg).
        """
        if self.cumulative_production_kg > 0:
            total_energy = (
                self.cumulative_production_energy_kwh +
                self.cumulative_compression_energy_kwh
            )
            return total_energy / self.cumulative_production_kg
        return 0.0
