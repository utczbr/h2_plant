"""
Helpers to load/normalize CAPEX, OPEX, and LCOH report payloads for GUI tables.

This module is Qt-free so it can be tested in isolation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


CAPEX_SUMMARY_FIELDS = [
    "generated_at",
    "simulation_name",
    "overall_cost_class",
    "currency",
    "total_C_BM",
    "total_C_BM_low",
    "total_C_BM_high",
    "total_installation",
    "total_installation_low",
    "total_installation_high",
    "total_installed_cost",
    "total_installed_cost_low",
    "total_installed_cost_high",
    "entries_with_cost",
    "entries_without_cost",
    "entries_out_of_bounds",
]

OPEX_SUMMARY_FIELDS = [
    "scenario_name",
    "simulation_hours",
    "annualization_factor",
    "total_variable_cost",
    "total_fixed_cost",
    "total_maintenance_cost",
    "total_opex",
    "total_opex_low",
    "total_opex_high",
    "fci",
    "annual_h2_production_kg",
    "opex_per_kg_h2",
]

LCOH_SUMMARY_FIELDS = [
    "discount_rate",
    "project_lifetime_years",
    "discount_factor_sum",
    "capex_total",
    "opex_annual_total",
    "annual_h2_total_kg",
    "lcoh_total",
    "lcoh_weighted_plant",
]


@dataclass
class TablePayload:
    columns: List[str]
    rows: List[List[Any]] = field(default_factory=list)


@dataclass
class ReportTableData:
    status: str  # ok | missing | error
    message: str
    source_path: Optional[Path]
    summary_rows: List[Tuple[str, Any]] = field(default_factory=list)
    tables: Dict[str, TablePayload] = field(default_factory=dict)


@dataclass
class _JsonLoadResult:
    status: str  # ok | missing | error
    message: str
    source_path: Optional[Path]
    payload: Optional[Dict[str, Any]]


def _dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    ordered: List[Path] = []
    seen: set[str] = set()
    for candidate in paths:
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        ordered.append(candidate)
    return ordered


def report_search_directories(
    output_dir: Optional[Path | str],
    scenarios_dir: Optional[Path | str] = None,
) -> List[Path]:
    """
    Resolve deterministic search directories for report files.

    Order:
    1) <output_dir>
    2) <output_dir>/Economics
    3) <output_dir.parent>/Economics
    4) <scenarios_dir>/Economics (if provided)
    """
    paths: List[Path] = []
    if output_dir:
        out = Path(str(output_dir))
        paths.extend([out, out / "Economics", out.parent / "Economics"])
    if scenarios_dir:
        scen = Path(str(scenarios_dir))
        paths.append(scen / "Economics")
    return _dedupe_paths(paths)


def resolve_report_path(
    report_filename: str,
    output_dir: Optional[Path | str],
    scenarios_dir: Optional[Path | str] = None,
) -> Optional[Path]:
    """Return the first matching report path based on deterministic lookup order."""
    for base in report_search_directories(output_dir, scenarios_dir):
        candidate = base / report_filename
        if candidate.exists():
            return candidate
    return None


def _load_json_report(
    report_filename: str,
    report_label: str,
    output_dir: Optional[Path | str],
    scenarios_dir: Optional[Path | str] = None,
) -> _JsonLoadResult:
    path = resolve_report_path(report_filename, output_dir, scenarios_dir)
    if path is None:
        return _JsonLoadResult(
            status="missing",
            message=f"{report_label} report not found.",
            source_path=None,
            payload=None,
        )

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _JsonLoadResult(
            status="error",
            message=f"Invalid JSON in {report_label} report: {exc}",
            source_path=path,
            payload=None,
        )
    except OSError as exc:
        return _JsonLoadResult(
            status="error",
            message=f"Failed to read {report_label} report: {exc}",
            source_path=path,
            payload=None,
        )

    if not isinstance(raw, dict):
        return _JsonLoadResult(
            status="error",
            message=f"Invalid {report_label} report: JSON root must be an object.",
            source_path=path,
            payload=None,
        )

    return _JsonLoadResult(
        status="ok",
        message=f"{report_label} report loaded.",
        source_path=path,
        payload=raw,
    )


def _summary_rows(payload: Dict[str, Any], fields: Iterable[str]) -> List[Tuple[str, Any]]:
    return [(field, payload.get(field)) for field in fields]


def _dict_rows(items: Any, field_name: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    if not isinstance(items, list):
        return None, f"Missing or invalid '{field_name}' list."
    if not all(isinstance(item, dict) for item in items):
        return None, f"Invalid '{field_name}' list: every item must be an object."
    return items, None


def load_capex_data(
    output_dir: Optional[Path | str],
    scenarios_dir: Optional[Path | str] = None,
) -> ReportTableData:
    loaded = _load_json_report("capex_report.json", "CAPEX", output_dir, scenarios_dir)
    if loaded.status != "ok":
        return ReportTableData(loaded.status, loaded.message, loaded.source_path)

    payload = loaded.payload or {}
    entries, entries_err = _dict_rows(payload.get("entries"), "entries")
    blocks, blocks_err = _dict_rows(payload.get("block_summaries"), "block_summaries")
    if entries_err or blocks_err:
        details = "; ".join(err for err in (entries_err, blocks_err) if err)
        return ReportTableData(
            status="error",
            message=f"Invalid CAPEX report: {details}",
            source_path=loaded.source_path,
        )

    block_columns = [
        "block_name",
        "equipment_total",
        "installation_total",
        "total_installed_cost",
        "total_installed_cost_low",
        "total_installed_cost_high",
    ]
    entry_columns = [
        "tag",
        "block",
        "name",
        "component_type",
        "design_capacity",
        "capacity_unit",
        "C_BM",
        "C_BM_low",
        "C_BM_high",
        "capacity_within_bounds",
    ]

    return ReportTableData(
        status="ok",
        message=loaded.message,
        source_path=loaded.source_path,
        summary_rows=_summary_rows(payload, CAPEX_SUMMARY_FIELDS),
        tables={
            "blocks": TablePayload(
                columns=block_columns,
                rows=[[row.get(col) for col in block_columns] for row in (blocks or [])],
            ),
            "entries": TablePayload(
                columns=entry_columns,
                rows=[[row.get(col) for col in entry_columns] for row in (entries or [])],
            ),
        },
    )


def load_opex_data(
    output_dir: Optional[Path | str],
    scenarios_dir: Optional[Path | str] = None,
) -> ReportTableData:
    loaded = _load_json_report("opex_report.json", "OPEX", output_dir, scenarios_dir)
    if loaded.status != "ok":
        return ReportTableData(loaded.status, loaded.message, loaded.source_path)

    payload = loaded.payload or {}
    items, items_err = _dict_rows(payload.get("items"), "items")
    if items_err:
        return ReportTableData(
            status="error",
            message=f"Invalid OPEX report: {items_err}",
            source_path=loaded.source_path,
        )

    item_columns = [
        "name",
        "category",
        "annual_quantity",
        "unit_price",
        "annual_cost",
        "source",
        "formula",
    ]

    return ReportTableData(
        status="ok",
        message=loaded.message,
        source_path=loaded.source_path,
        summary_rows=_summary_rows(payload, OPEX_SUMMARY_FIELDS),
        tables={
            "items": TablePayload(
                columns=item_columns,
                rows=[[row.get(col) for col in item_columns] for row in (items or [])],
            ),
        },
    )


def _normalize_lcoh_rows(payload: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    raw_variants = payload.get("variants")
    if isinstance(raw_variants, dict):
        variant_order = payload.get("variant_order")
        if not isinstance(variant_order, list):
            variant_order = sorted(raw_variants.keys())
        normalized: Dict[str, Dict[str, Any]] = {}
        for variant in variant_order:
            entry = raw_variants.get(variant)
            if not isinstance(entry, dict):
                continue
            normalized[str(variant)] = entry
        return normalized, [str(v) for v in variant_order if str(v) in normalized]

    # Legacy shape: no variants key, top-level single report fields.
    return {"base": payload}, ["base"]


def load_lcoh_data(
    output_dir: Optional[Path | str],
    scenarios_dir: Optional[Path | str] = None,
) -> ReportTableData:
    loaded = _load_json_report("lcoh_report.json", "LCOH", output_dir, scenarios_dir)
    if loaded.status != "ok":
        return ReportTableData(loaded.status, loaded.message, loaded.source_path)

    payload = loaded.payload or {}
    variants, variant_order = _normalize_lcoh_rows(payload)
    if not variants:
        return ReportTableData(
            status="error",
            message="Invalid LCOH report: no valid variant payloads found.",
            source_path=loaded.source_path,
        )

    variant_columns = [
        "variant",
        "capex_total",
        "opex_annual_total",
        "annual_h2_total_kg",
        "lcoh_total",
        "lcoh_weighted_plant",
    ]
    pathway_columns = ["variant", "pathway", "annual_h2", "capex", "opex", "lcoh"]
    breakdown_columns = ["variant", "component", "value"]
    warnings_columns = ["warning"]

    variant_rows: List[List[Any]] = []
    pathway_rows: List[List[Any]] = []
    breakdown_rows: List[List[Any]] = []
    warnings_rows: List[List[Any]] = []
    seen_warnings: set[str] = set()

    for warning in payload.get("warnings", []) or []:
        warning_text = str(warning)
        if warning_text in seen_warnings:
            continue
        seen_warnings.add(warning_text)
        warnings_rows.append([warning_text])

    for variant in variant_order:
        row = variants.get(variant, {})
        variant_rows.append(
            [
                variant,
                row.get("capex_total"),
                row.get("opex_annual_total"),
                row.get("annual_h2_total_kg"),
                row.get("lcoh_total"),
                row.get("lcoh_weighted_plant"),
            ]
        )

        annual_h2_by = row.get("annual_h2_by_pathway", {}) or {}
        capex_by = row.get("capex_by_pathway", {}) or {}
        opex_by = row.get("opex_by_pathway", {}) or {}
        lcoh_by = row.get("lcoh_by_pathway", {}) or {}
        pathway_names = sorted(
            set(annual_h2_by.keys())
            | set(capex_by.keys())
            | set(opex_by.keys())
            | set(lcoh_by.keys())
        )
        for pathway in pathway_names:
            pathway_rows.append(
                [
                    variant,
                    pathway,
                    annual_h2_by.get(pathway),
                    capex_by.get(pathway),
                    opex_by.get(pathway),
                    lcoh_by.get(pathway),
                ]
            )

        breakdown = row.get("lcoh_breakdown", {}) or {}
        if isinstance(breakdown, dict):
            for component, value in breakdown.items():
                breakdown_rows.append([variant, component, value])

        for warning in row.get("warnings", []) or []:
            warning_text = str(warning)
            if warning_text in seen_warnings:
                continue
            seen_warnings.add(warning_text)
            warnings_rows.append([warning_text])

    return ReportTableData(
        status="ok",
        message=loaded.message,
        source_path=loaded.source_path,
        summary_rows=_summary_rows(payload, LCOH_SUMMARY_FIELDS),
        tables={
            "variants": TablePayload(columns=variant_columns, rows=variant_rows),
            "pathways": TablePayload(columns=pathway_columns, rows=pathway_rows),
            "breakdown": TablePayload(columns=breakdown_columns, rows=breakdown_rows),
            "warnings": TablePayload(columns=warnings_columns, rows=warnings_rows),
        },
    )
