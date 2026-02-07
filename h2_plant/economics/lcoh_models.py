"""
LCOH Data Models

Pydantic models for LCOH configuration and reporting.
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class LcohReport(BaseModel):
    """Final aggregated LCOH report."""

    generated_at: str = Field(..., description="ISO timestamp")
    discount_rate: float = Field(0.0, description="Discount rate used")
    project_lifetime_years: int = Field(0, description="Project lifetime in years")
    discount_factor_sum: float = Field(0.0, description="Sum of discount factors over project life")

    capex_total: float = Field(0.0, description="Total installed CAPEX (EUR)")
    opex_annual_total: float = Field(0.0, description="Total annual OPEX (EUR/year)")

    annual_h2_total_kg: float = Field(0.0, description="Annual H2 production (kg/year)")
    annual_h2_by_pathway: Dict[str, float] = Field(default_factory=dict)
    capex_by_pathway: Dict[str, float] = Field(default_factory=dict)
    opex_by_pathway: Dict[str, float] = Field(default_factory=dict)

    lcoh_total: float = Field(0.0, description="Plant LCOH (EUR/kg)")
    lcoh_by_pathway: Dict[str, float] = Field(default_factory=dict)
    lcoh_weighted_plant: float = Field(0.0, description="Production-weighted LCOH (EUR/kg)")

    lcoh_breakdown: Dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
