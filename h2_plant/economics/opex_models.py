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
    metric: str = Field("sum", description="Aggregation method: sum, max, avg")
    
    # Cost Parameters
    price: float = Field(0.0, description="Unit price, fixed sum, or percentage factor")
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
        
        # Calculate per-kg cost if production data available
        if self.annual_h2_production_kg > 0:
            self.opex_per_kg_h2 = self.total_opex / self.annual_h2_production_kg
