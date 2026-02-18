import pytest
from pydantic import ValidationError

from h2_plant.gui.core.economics_editor import (
    apply_general_econ_info,
    extract_general_econ_info,
    load_yaml_text,
    validate_capex_yaml_text,
    validate_opex_yaml_text,
)


def test_load_yaml_text_reads_full_file(tmp_path):
    path = tmp_path / "sample.yaml"
    content = "a: 1\nb:\n  c: 2\n"
    path.write_text(content, encoding="utf-8")
    assert load_yaml_text(path) == content


def test_validate_capex_yaml_text_accepts_schema_and_extracts_general_info():
    text = """
cepci:
  base_year: 2001
  base_index: 397.0
  current_year: 2025
  current_index: 797.0
capacity_mode: history
equipment:
  - tag: EQ-1
    block: General
    name: Main Compressor
    topology_ids: [comp_1]
    component_type: Compressor
""".strip()
    parsed = validate_capex_yaml_text(text)
    info = extract_general_econ_info(parsed)
    assert info["base_year"] == 2001
    assert info["base_index"] == pytest.approx(397.0)
    assert info["current_year"] == 2025
    assert info["current_index"] == pytest.approx(797.0)
    assert info["capacity_mode"] == "history"


def test_apply_general_econ_info_updates_payload():
    base = {
        "capacity_mode": "design",
        "equipment": [
            {
                "tag": "EQ-1",
                "block": "General",
                "name": "Main Compressor",
                "topology_ids": ["comp_1"],
                "component_type": "Compressor",
            }
        ],
    }
    merged = apply_general_econ_info(
        base,
        {
            "base_year": 2010,
            "base_index": 500.0,
            "current_year": 2026,
            "current_index": 820.0,
            "capacity_mode": "history",
        },
    )
    assert merged["capacity_mode"] == "history"
    assert merged["cepci"]["base_year"] == 2010
    assert merged["cepci"]["current_index"] == pytest.approx(820.0)
    assert merged["equipment"][0]["tag"] == "EQ-1"


def test_validate_capex_yaml_text_accepts_power_law_scaling_coefficients():
    text = """
equipment:
  - tag: HBT
    block: Storage
    name: Underground H2 Storage
    topology_ids: [LP_Storage_Tank]
    component_type: Underground Storage
    capacity_variable: unknown
    capacity_unit: "-"
    capacity_aggregation: sum
    cost_source: vendor_quote
    vendor_quote_eur: 6736268.0
    coefficients:
      cost_method: power_law_scaling
      C_ref_eur: 6736268.0
      capacity_ref: 4800
      exponent: 0.65
""".strip()
    parsed = validate_capex_yaml_text(text)
    assert parsed["equipment"][0]["coefficients"]["cost_method"] == "power_law_scaling"


def test_validate_capex_yaml_text_rejects_non_turton_unknown_cost_method_coefficients():
    text = """
equipment:
  - tag: HBT
    block: Storage
    name: Underground H2 Storage
    topology_ids: [LP_Storage_Tank]
    component_type: Underground Storage
    capacity_variable: unknown
    capacity_unit: "-"
    capacity_aggregation: sum
    cost_source: vendor_quote
    vendor_quote_eur: 6736268.0
    coefficients:
      cost_method: unsupported_scaling
      C_ref_eur: 6736268.0
      capacity_ref: 4800
      exponent: 0.65
""".strip()
    with pytest.raises(ValidationError):
        validate_capex_yaml_text(text)


def test_validate_capex_yaml_text_rejects_invalid_capacity_mode():
    text = """
capacity_mode: invalid_mode
equipment: []
""".strip()
    with pytest.raises(ValueError):
        validate_capex_yaml_text(text)


def test_validate_opex_yaml_text_requires_items():
    with pytest.raises(ValueError):
        validate_opex_yaml_text("scenario_name: test\n")

    text = """
scenario_name: test
opex_items:
  - name: Electricity
    category: Variable
    strategy: variable
    price: 0.2
""".strip()
    parsed = validate_opex_yaml_text(text)
    assert len(parsed["opex_items"]) == 1
