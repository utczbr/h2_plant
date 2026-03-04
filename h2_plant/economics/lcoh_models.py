"""
LCOH Data Models

Pydantic models for LCOH configuration and reporting.
"""

from typing import Dict

from pydantic import BaseModel, Field


class LcohReport(BaseModel):
    """Final aggregated LCOH report."""

    generated_at: str = Field(..., description="ISO timestamp")
    discount_rate: float = Field(0.0, description="Discount rate used")
    project_lifetime_years: int = Field(0, description="Project lifetime in years")
    discount_factor_sum: float = Field(0.0, description="Sum of discount factors over project life")

    capex_total: float = Field(0.0, description="Total installed CAPEX (EUR)")
    opex_annual_total: float = Field(0.0, description="Total annual OPEX (EUR/year)")
    opex_by_year: Dict[str, float] = Field(
        default_factory=dict,
        description="Year-indexed OPEX values used in discounted calculation",
    )

    annual_h2_total_kg: float = Field(0.0, description="Annual H2 production (kg/year)")
    annual_h2_by_pathway: Dict[str, float] = Field(default_factory=dict)
    annual_h2_total_kg_by_year: Dict[str, float] = Field(
        default_factory=dict,
        description="Year-indexed H2 production values used in discounted calculation",
    )
    capex_by_pathway: Dict[str, float] = Field(default_factory=dict)
    opex_by_pathway: Dict[str, float] = Field(default_factory=dict)

    lcoh_total: float = Field(0.0, description="Plant LCOH (EUR/kg)")
    lcoh_by_pathway: Dict[str, float] = Field(default_factory=dict)
    lcoh_weighted_plant: float = Field(0.0, description="Production-weighted LCOH (EUR/kg)")
    discounted_opex_pv: float = Field(0.0, description="Present value of OPEX stream (EUR)")
    discounted_h2_pv: float = Field(0.0, description="Present value of H2 production stream (kg)")

    lcoh_breakdown: Dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class LcohVariantsReport(BaseModel):
    """Combined low/base/high LCOH report with base-field compatibility."""

    generated_at: str = Field(..., description="ISO timestamp")
    discount_rate: float = Field(0.0, description="Discount rate used")
    project_lifetime_years: int = Field(0, description="Project lifetime in years")
    variant_order: list[str] = Field(default_factory=lambda: ["low", "base", "high"])
    variants: Dict[str, LcohReport] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    # Backward-compatible top-level base fields.
    discount_factor_sum: float = Field(0.0, description="Base-variant discount factor sum")
    capex_total: float = Field(0.0, description="Base-variant CAPEX total")
    opex_annual_total: float = Field(0.0, description="Base-variant annual OPEX total")
    annual_h2_total_kg: float = Field(0.0, description="Base-variant annual H2 production")
    annual_h2_by_pathway: Dict[str, float] = Field(default_factory=dict)
    capex_by_pathway: Dict[str, float] = Field(default_factory=dict)
    opex_by_pathway: Dict[str, float] = Field(default_factory=dict)
    lcoh_total: float = Field(0.0, description="Base-variant plant LCOH (EUR/kg)")
    lcoh_by_pathway: Dict[str, float] = Field(default_factory=dict)
    lcoh_weighted_plant: float = Field(0.0, description="Base-variant production-weighted LCOH")
    lcoh_breakdown: Dict[str, float] = Field(default_factory=dict)
