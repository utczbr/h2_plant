"""
Pydantic Models for CAPEX Configuration

Type-safe models for equipment mappings, cost coefficients, and CAPEX entries.
Includes validation, serialization, and AACE cost classification.
"""

from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, field_validator, model_validator
import logging

logger = logging.getLogger(__name__)


class AACECostClass(str, Enum):
    """
    AACE International Cost Estimate Classification System.
    
    Class 5: Concept Screening (±50% accuracy)
    Class 4: Study/Feasibility (±30% accuracy)
    Class 3: Budget Authorization (±20% accuracy)
    Class 2: Control/Bid-Tender (±15% accuracy)
    Class 1: Definitive/Check Estimate (±10% accuracy)
    """
    CLASS_5 = "Class 5"  # ±50% - Concept Screening
    CLASS_4 = "Class 4"  # ±30% - Study/Feasibility
    CLASS_3 = "Class 3"  # ±20% - Budget Authorization
    CLASS_2 = "Class 2"  # ±15% - Control/Bid-Tender
    CLASS_1 = "Class 1"  # ±10% - Definitive

    @property
    def accuracy_range(self) -> tuple[float, float]:
        """Return (low_factor, high_factor) for uncertainty band."""
        ranges = {
            "Class 5": (0.50, 1.50),
            "Class 4": (0.70, 1.30),
            "Class 3": (0.80, 1.20),
            "Class 2": (0.85, 1.15),
            "Class 1": (0.90, 1.10),
        }
        return ranges[self.value]


class CostCoefficients(BaseModel):
    """
    Cost estimation coefficients (Turton correlation format).
    
    Formula: log10(Cp0) = K1 + K2*log10(A) + K3*[log10(A)]^2
    Module cost: C_BM = Cp0 * F_BM * F_m  OR  Cp0 * (B1 + B2*F_m*F_p)
    """
    # Base correlation coefficients
    K1: float = Field(..., description="Intercept coefficient")
    K2: float = Field(..., description="Linear coefficient")
    K3: float = Field(0.0, description="Quadratic coefficient")
    
    # Material and pressure factors
    F_m: float = Field(1.0, description="Material factor (1.0 = carbon steel)")
    F_m_note: Optional[str] = Field(None, description="Material factor justification")
    
    F_p: float = Field(1.0, description="Pressure factor")
    F_p_note: Optional[str] = Field(None, description="Pressure factor justification")
    
    # Bare module factors (alternative to B1/B2)
    F_BM: Optional[float] = Field(None, description="Bare module factor (simple method)")
    F_BM_note: Optional[str] = None
    
    # B-factor method (for vessels, heat exchangers)
    B1: Optional[float] = Field(None, description="B1 constant for module factor")
    B2: Optional[float] = Field(None, description="B2 constant for module factor")
    
    # Additional multipliers
    F_multi: float = Field(1.0, description="Additional multiplier (e.g., demister addon)")
    F_multi_note: Optional[str] = None
    
    # Capacity bounds for correlation validity
    capacity_min: Optional[float] = Field(None, description="Minimum valid capacity")
    capacity_max: Optional[float] = Field(None, description="Maximum valid capacity")
    capacity_unit: str = Field("kW", description="Capacity unit for bounds")

    @property
    def uses_b_factors(self) -> bool:
        """Check if using B-factor method vs simple F_BM."""
        return self.B1 is not None and self.B2 is not None


class EquipmentMapping(BaseModel):
    """
    Maps equipment tags to topology IDs and defines cost calculation parameters.
    """
    tag: str = Field(..., description="Equipment tag (e.g., MTC-1)")
    block: str = Field("General", description="Block for installation cost grouping (PEM, SOEC, ATR, Storage, Water_Treatment, General)")
    name: str = Field(..., description="Equipment name/description")
    topology_ids: List[str] = Field(..., description="Component IDs from plant_topology.yaml")
    component_type: str = Field(..., description="Equipment type for coefficient lookup")
    
    # Process context
    process_description: Optional[str] = Field(None, description="Process function description")
    
    # Capacity extraction
    capacity_variable: str = Field("power_kw", description="Variable to extract (power_kw, area_m2, volume_m3)")
    capacity_unit: str = Field("kW", description="Unit for design capacity")
    capacity_aggregation: str = Field("sum", description="How to aggregate multiple components: sum, max, avg")
    capacity_mode: Optional[str] = Field(None, description="Override global capacity_mode: 'design' or 'history'")
    
    # Cost estimation
    coefficients: Optional[CostCoefficients] = Field(None, description="Turton coefficients (or inherit from type)")
    cost_source: str = Field("turton", description="Cost source: turton, vendor_quote, percentage")
    vendor_quote_usd: Optional[float] = Field(None, description="Vendor quote if available")
    fixed_cost_eur: Optional[float] = Field(None, description="Fixed cost in EUR (will be converted to USD)")
    
    # Metadata
    restrictions: List[str] = Field(default_factory=list, description="Validity restrictions")
    notes: List[str] = Field(default_factory=list, description="Engineering notes")


class CapexEntry(BaseModel):
    """
    Single CAPEX line item with full audit trail.
    """
    tag: str
    name: str
    topology_ids: List[str]
    component_type: str
    
    # Capacity
    design_capacity: float = Field(..., description="Extracted/calculated design capacity")
    capacity_unit: str
    capacity_source: str = Field(..., description="How capacity was determined")
    capacity_within_bounds: bool = Field(True, description="Is capacity within correlation bounds?")
    
    # Cost
    C_p0: Optional[float] = Field(None, description="Base purchased cost (before inflation)")
    C_BM: Optional[float] = Field(None, description="Bare module cost (USD, CEPCI 2026)")
    C_BM_low: Optional[float] = Field(None, description="Low estimate (uncertainty band)")
    C_BM_high: Optional[float] = Field(None, description="High estimate (uncertainty band)")
    
    cost_formula: Optional[str] = Field(None, description="Formula used for cost calculation")
    cost_source: str = Field("turton", description="Cost estimation method used")
    
    # Coefficients used
    coefficients: Optional[Dict[str, Any]] = Field(None, description="Coefficients used in calculation")
    
    # AACE classification
    cost_class: AACECostClass = Field(AACECostClass.CLASS_4, description="AACE cost estimate class")
    
    # Audit trail
    notes: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    @property
    def has_valid_cost(self) -> bool:
        return self.C_BM is not None and self.C_BM > 0


class CEPCIData(BaseModel):
    """Chemical Engineering Plant Cost Index data."""
    base_year: int = Field(2001, description="Base year for Turton correlations")
    base_index: float = Field(397.0, description="CEPCI for base year")
    current_year: int = Field(2026, description="Current/target year")
    current_index: float = Field(820.0, description="CEPCI for current year")
    
    @property
    def inflation_factor(self) -> float:
        """Cost inflation factor from base to current year."""
        return self.current_index / self.base_index


class BlockCostSummary(BaseModel):
    """Summary of costs by block with installation factors applied."""
    block_name: str = Field(..., description="Block identifier (PEM, SOEC, ATR, etc.)")
    equipment_tags: List[str] = Field(default_factory=list, description="Equipment tags in this block")
    equipment_total: float = Field(0.0, description="Sum of C_BM for all equipment in block")
    
    # Installation costs by category
    installation_costs: Dict[str, float] = Field(default_factory=dict, description="Installation cost per category")
    installation_total: float = Field(0.0, description="Sum of all installation costs")
    
    # Block total (equipment + installation)
    total_installed_cost: float = Field(0.0, description="Total installed cost = equipment + installation")
    
    def calculate(self, entries: List["CapexEntry"], installation_factors: Dict[str, float]) -> None:
        """Calculate equipment total and apply installation factors."""
        # Sum equipment costs
        self.equipment_total = sum(e.C_BM or 0 for e in entries if e.tag in self.equipment_tags)
        
        # Apply installation factors
        self.installation_costs = {}
        for category, factor in installation_factors.items():
            self.installation_costs[category] = self.equipment_total * factor
        
        self.installation_total = sum(self.installation_costs.values())
        self.total_installed_cost = self.equipment_total + self.installation_total


class CapexReport(BaseModel):
    """
    Complete CAPEX report with all entries and metadata.
    """
    # Metadata
    generated_at: str = Field(..., description="ISO timestamp")
    simulation_name: Optional[str] = None
    total_simulation_hours: Optional[int] = None
    
    # Economic parameters
    cepci: CEPCIData = Field(default_factory=CEPCIData)
    currency: str = Field("USD", description="Cost currency")
    
    # Cost classification
    overall_cost_class: AACECostClass = Field(AACECostClass.CLASS_4)
    
    # Entries
    entries: List[CapexEntry] = Field(default_factory=list)
    
    # Block Summaries (with installation costs)
    block_summaries: List[BlockCostSummary] = Field(default_factory=list, description="Cost summaries by block")
    
    # Equipment Totals (before installation)
    total_C_BM: float = Field(0.0, description="Sum of all C_BM values")
    total_C_BM_low: float = Field(0.0, description="Sum of low estimates")
    total_C_BM_high: float = Field(0.0, description="Sum of high estimates")
    
    # Total Installed Cost (after installation factors)
    total_installation: float = Field(0.0, description="Total installation costs")
    total_installed_cost: float = Field(0.0, description="Equipment + Installation")
    
    # Statistics
    entries_with_cost: int = Field(0, description="Entries with valid cost")
    entries_without_cost: int = Field(0, description="Entries missing cost")
    entries_out_of_bounds: int = Field(0, description="Entries with capacity outside bounds")
    
    def calculate_totals(self) -> None:
        """Recalculate totals from entries."""
        self.total_C_BM = sum(e.C_BM or 0 for e in self.entries)
        self.total_C_BM_low = sum(e.C_BM_low or 0 for e in self.entries)
        self.total_C_BM_high = sum(e.C_BM_high or 0 for e in self.entries)
        
        self.entries_with_cost = sum(1 for e in self.entries if e.has_valid_cost)
        self.entries_without_cost = sum(1 for e in self.entries if not e.has_valid_cost)
        self.entries_out_of_bounds = sum(1 for e in self.entries if not e.capacity_within_bounds)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return summary statistics for quick review."""
        return {
            "total_C_BM": f"${self.total_C_BM:,.0f}",
            "total_range": f"${self.total_C_BM_low:,.0f} - ${self.total_C_BM_high:,.0f}",
            "cost_class": self.overall_cost_class.value,
            "entries_valid": f"{self.entries_with_cost}/{len(self.entries)}",
            "entries_out_of_bounds": self.entries_out_of_bounds,
        }
