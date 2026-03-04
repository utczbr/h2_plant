"""
OPEX Data Models

Pydantic models for OPEX configuration and reporting:
- OpexCategory: Classification enum
- OpexItemConfig: Configuration for a single OPEX line item
- OpexResult: Calculated result for a line item
- OpexReport: Aggregated OPEX report
"""

from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class OpexCategory(str, Enum):
    """OPEX cost category classification."""
    VARIABLE = "Variable"       # Utilities, Feedstock
    FIXED = "Fixed"             # Labor, Admin, Insurance
    MAINTENANCE = "Maintenance" # Spares, Repairs, Catalyst


class OpexItemConfig(BaseModel):
    """Configuration for a single OPEX line item."""
    
    # Identification
    name: str = Field(..., description="Display name for this cost item")
    category: OpexCategory = Field(..., description="Cost category")
    strategy: str = Field("variable", description="Calculation strategy: variable, fixed, factor, turton_labor, scaling")
    
    # Simulation Data Link (for Variable costs)
    resource_id: Optional[str] = Field(None, description="Column pattern to match in simulation history")
    price_resource_id: Optional[str] = Field(
        None,
        description="Optional timestep price signal column for dynamic pricing",
    )
    pathway_driver_resource_ids: Optional[Dict[str, str]] = Field(
        None,
        description="Optional per-pathway driver columns for causal variable OPEX allocation",
    )
    metric: str = Field("sum", description="Aggregation method: sum, max, avg")
    
    # Cost Parameters
    price: float = Field(0.0, description="Unit price, fixed sum, or percentage factor")
    cost_multiplier: float = Field(
        1.0,
        description="Multiplier applied in variable costs (unit conversion/sign handling)",
    )
    unit: str = Field("USD/year", description="Display unit for reporting")
    
    # Factor Strategy Parameters
    base_reference: Optional[str] = Field(None, description="Base for factor calculation: FCI, Labor, C_OL")
    
    # Turton Labor Parameters
    turton_P: int = Field(0, description="Number of particulate processing steps")
    turton_Nnp: int = Field(75, description="Number of non-particulate processing steps")
    shifts: float = Field(4.8, description="Number of shifts (accounts for 24/7 operation)")
    hours_per_year: float = Field(2080, description="Working hours per operator per year")
    
    # Scaling Strategy Parameters
    ref_production: float = Field(1.0, description="Reference production for scaling")
    scaling_exponent: float = Field(0.6, description="Scaling exponent")
    
    # Metadata
    notes: List[str] = Field(default_factory=list, description="Engineering notes")
    is_credit: bool = Field(False, description="Whether this item represents a credit term")
    lcoh_component: Optional[str] = Field(
        None,
        description="Optional LCOH sub-breakdown tag (energy, water, compression, etc.)",
    )


class OpexResult(BaseModel):
    """Result for a single OPEX line item after calculation."""
    
    name: str
    category: OpexCategory
    annual_quantity: float = Field(0.0, description="Annual quantity used")
    unit_price: float = Field(0.0, description="Unit price applied")
    annual_cost: float = Field(0.0, description="Calculated annual cost")
    formula: str = Field("", description="Calculation formula string")
    source: str = Field("config", description="Data source: simulation, config, calculation")
    warnings: List[str] = Field(default_factory=list)
    pathway_shares: Optional[Dict[str, float]] = Field(
        None,
        description="Optional per-pathway shares used for causal LCOH allocation",
    )
    is_credit: bool = Field(False, description="Whether this result is a credit term")
    lcoh_component: Optional[str] = Field(
        None,
        description="Optional propagated LCOH sub-breakdown tag",
    )


class OpexReport(BaseModel):
    """Final aggregated OPEX report."""
    
    # Metadata
    scenario_name: str = Field("", description="Scenario identifier")
    simulation_hours: float = Field(0.0, description="Hours of simulation data used")
    annualization_factor: float = Field(1.0, description="Factor to convert simulation to annual (8760/sim_hours)")
    
    # Line Items
    items: List[OpexResult] = Field(default_factory=list)
    
    # Category Totals
    total_variable_cost: float = Field(0.0, description="Sum of Variable category")
    total_fixed_cost: float = Field(0.0, description="Sum of Fixed category")
    total_maintenance_cost: float = Field(0.0, description="Sum of Maintenance category")
    total_opex: float = Field(0.0, description="Total annual OPEX")
    total_opex_low: Optional[float] = Field(None, description="Low annual OPEX estimate")
    total_opex_high: Optional[float] = Field(None, description="High annual OPEX estimate")
    total_credit_cost: float = Field(0.0, description="Annual signed sum of credit terms")
    total_opex_cashflow: float = Field(
        0.0,
        description="Annual OPEX cash outflow after adding back explicit credits",
    )
    total_opex_cashflow_low: Optional[float] = Field(None, description="Low annual OPEX cash outflow")
    total_opex_cashflow_high: Optional[float] = Field(None, description="High annual OPEX cash outflow")

    # Year-by-year diagnostics (optional, backward-compatible)
    year_index: Optional[List[int]] = Field(
        default=None,
        description="Simulation-relative year index (1..N) for yearly OPEX series",
    )
    year_hours: Optional[List[float]] = Field(
        default=None,
        description="Covered simulation hours per year window",
    )
    total_opex_by_year: Optional[List[float]] = Field(
        default=None,
        description="Total yearly OPEX cash terms (EUR) per simulation-relative year",
    )
    total_opex_cashflow_by_year: Optional[List[float]] = Field(
        default=None,
        description="Total yearly OPEX cash outflow after explicit credits",
    )
    total_variable_cost_by_year: Optional[List[float]] = Field(
        default=None,
        description="Variable category yearly totals",
    )
    total_fixed_cost_by_year: Optional[List[float]] = Field(
        default=None,
        description="Fixed category yearly totals",
    )
    total_maintenance_cost_by_year: Optional[List[float]] = Field(
        default=None,
        description="Maintenance category yearly totals",
    )
    total_opex_low_by_year: Optional[List[float]] = Field(
        default=None,
        description="Low yearly OPEX totals (aligned to CAPEX low variant)",
    )
    total_opex_high_by_year: Optional[List[float]] = Field(
        default=None,
        description="High yearly OPEX totals (aligned to CAPEX high variant)",
    )
    total_opex_cashflow_low_by_year: Optional[List[float]] = Field(
        default=None,
        description="Low yearly OPEX cash outflow",
    )
    total_opex_cashflow_high_by_year: Optional[List[float]] = Field(
        default=None,
        description="High yearly OPEX cash outflow",
    )
    annual_h2_production_kg_by_year: Optional[List[float]] = Field(
        default=None,
        description="Annualized H2 production by simulation-relative year",
    )
    item_annual_cost_by_year: Optional[Dict[str, List[float]]] = Field(
        default=None,
        description="Per-item yearly OPEX cost vectors (item name -> yearly costs)",
    )
    
    # Reference Values (from CAPEX)
    fci: float = Field(0.0, description="Fixed Capital Investment (from CAPEX)")
    labor_cost: float = Field(0.0, description="Calculated Labor Cost")
    
    # Production Metrics
    annual_h2_production_kg: float = Field(0.0, description="Annual H2 production in kg")
    opex_per_kg_h2: float = Field(0.0, description="OPEX cost per kg H2")
    
    def calculate_totals(self) -> None:
        """Calculate category totals from line items."""
        self.total_variable_cost = sum(
            r.annual_cost for r in self.items if r.category == OpexCategory.VARIABLE
        )
        self.total_fixed_cost = sum(
            r.annual_cost for r in self.items if r.category == OpexCategory.FIXED
        )
        self.total_maintenance_cost = sum(
            r.annual_cost for r in self.items if r.category == OpexCategory.MAINTENANCE
        )
        self.total_opex = (
            self.total_variable_cost + 
            self.total_fixed_cost + 
            self.total_maintenance_cost
        )
        self.total_credit_cost = sum(
            r.annual_cost for r in self.items if bool(getattr(r, "is_credit", False))
        )
        self.total_opex_cashflow = self.total_opex - self.total_credit_cost
        
        # Calculate per-kg cost if production data available
        if self.annual_h2_production_kg > 0:
            self.opex_per_kg_h2 = self.total_opex / self.annual_h2_production_kg
