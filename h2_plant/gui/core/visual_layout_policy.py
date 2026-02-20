"""
Zone-based layout policy for industrial plant visual layout.

This module is the single source of truth for:
- Process zone definitions (PHASE_DEFINITIONS) — shared with audit module
- Zone display order (ZONE_ORDER) and Zone overlay colors (ZONE_COLORS)
- Equipment-block display order (BLOCK_ORDER) and Block overlay colors (BLOCK_COLORS)
- Deterministic spacing constants

It is intentionally kept free of Qt and YAML dependencies so it can be
imported by both the importer (headless) and the GUI (main_window).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Process zone taxonomy
# ---------------------------------------------------------------------------

PHASE_DEFINITIONS: List[Tuple[str, Tuple[str, ...]]] = [
    (
        "Phase 1: Electrolysis & Power Conditioning",
        ("PEM", "SOEC", "PowerTransformer"),
    ),
    (
        "Phase 2: Thermal Management",
        ("Chiller", "DryCooler", "ElectricBoiler", "Interchanger", "CoolingManager", "Attemperator"),
    ),
    (
        "Phase 3: Separation & Gas Purification",
        (
            "Coalescer",
            "KnockOutDrum",
            "HydrogenMultiCyclone",
            "DeoxoReactor",
            "PSA Unit",
            "SyngasPSA",
            "SeparationTank",
        ),
    ),
    (
        "Phase 4: Flow Control & Mixing",
        (
            "Valve",
            "Mixer",
            "DrainRecorderMixer",
            "SignalMakeupMixer",
            "ProportionalMakeupMixer",
            "StreamSplitter",
            "OxygenMakeupNode",
        ),
    ),
    (
        "Phase 5: Water Supply & Pumping",
        ("ExternalWaterSource", "WaterPurifier", "UltraPureWaterTank", "WaterPumpThermodynamic"),
    ),
    (
        "Phase 6: Storage & Delivery",
        ("DetailedTank", "CompressorSingle", "DischargeStation"),
    ),
    (
        "Phase 7: Reforming & Feedstock",
        ("IntegratedATRPlant", "ATR_Boiler", "BiogasSource"),
    ),
]

# Canonical left-to-right display order (matches phase numbering).
ZONE_ORDER: List[str] = [name for name, _ in PHASE_DEFINITIONS]

# Per-zone background colors for swimlane overlays (R, G, B, 0–255).
# Chosen for distinctiveness and light-background legibility.
ZONE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Phase 1: Electrolysis & Power Conditioning": (173, 216, 230),   # light blue
    "Phase 2: Thermal Management":                 (255, 200, 150),   # light orange
    "Phase 3: Separation & Gas Purification":      (180, 230, 180),   # light green
    "Phase 4: Flow Control & Mixing":              (210, 180, 230),   # light purple
    "Phase 5: Water Supply & Pumping":             (160, 210, 240),   # sky blue
    "Phase 6: Storage & Delivery":                 (240, 230, 160),   # light amber
    "Phase 7: Reforming & Feedstock":              (230, 170, 170),   # light red
    "Uncategorised":                               (200, 200, 200),   # neutral grey
}

# Reverse lookup: backend_type -> zone name
ZONE_BY_BACKEND_TYPE: Dict[str, str] = {
    backend_type: zone_name
    for zone_name, backend_types in PHASE_DEFINITIONS
    for backend_type in backend_types
}

# ---------------------------------------------------------------------------
# system_group → parent block mapping
# Maps fine-grained topology system_group values to their equipment block.
# Used as a fallback when a node is not listed in equipment_mappings.yaml.
# ---------------------------------------------------------------------------

SYSTEM_GROUP_TO_BLOCK: Dict[str, str] = {
    # SOEC sub-systems
    "SOEC":               "SOEC",
    "SOEC_Steam_Loop":    "SOEC",
    "SOEC_Water_Loop":    "SOEC",
    "SOEC_H2_Train":      "SOEC",
    # PEM sub-systems
    "PEM":                "PEM",
    "PEM_O2_Train":       "PEM",
    "PEM_Water_Loop":     "PEM",
    # ATR / reforming sub-systems
    "ATR_Unit":           "ATR",
    "ATR_Water_Loop":     "ATR",
    "O2_Header":          "ATR",
    # Utilities / BOP
    "BOP":                "Utilities",
    "Utilities":          "Utilities",
    # Water treatment
    "Water_Supply":       "Water_Treatment",
    "Water_Treatment":    "Water_Treatment",
}

# ---------------------------------------------------------------------------
# Equipment-block taxonomy (primary grouping for report-mode layout)
# ---------------------------------------------------------------------------

# Canonical left-to-right process order for equipment blocks.
# Blocks not listed here are appended at the end in alphabetical order.
BLOCK_ORDER: List[str] = [
    "Water_Treatment",
    "Utilities",
    "SOEC",
    "PEM",
    "ATR",
    "Storage",
    "General",
    "Uncategorised",
]

# Per-block background colors for swimlane overlays (R, G, B, 0–255).
BLOCK_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Water_Treatment": (160, 220, 255),   # light blue
    "Utilities":       (200, 200, 200),   # neutral grey
    "SOEC":            (173, 216, 230),   # steel blue
    "PEM":             (180, 230, 180),   # mint green
    "ATR":             (255, 200, 150),   # light orange
    "Storage":         (240, 230, 160),   # amber
    "General":         (220, 210, 200),   # beige
    "Other":           (215, 215, 215),   # light grey
    "Uncategorised":   (210, 210, 210),   # neutral
}

# ---------------------------------------------------------------------------
# Deterministic spacing constants (pixels)
# ---------------------------------------------------------------------------

COLUMN_SPACING: float = 220.0   # Horizontal gap between process columns within a block
ROW_SPACING: float = 140.0      # Vertical gap between nodes within a sub-lane
LANE_GAP: float = 100.0         # Vertical gap between sub-lanes within a block
ZONE_MARGIN: float = 40.0       # Padding around block bounding boxes
BLOCK_GAP: float = 120.0        # Horizontal gap between blocks


def assign_zone(params: dict, backend_type: str) -> str:
    """
    Return the zone name for a node (used by legacy layout and audit).

    Priority:
    1. ``system_group`` param explicitly set in the topology YAML.
    2. Backend type lookup via ZONE_BY_BACKEND_TYPE.
    3. Fallback: "Uncategorised".
    """
    system_group = str(params.get("system_group") or "").strip()
    if system_group:
        return system_group
    return ZONE_BY_BACKEND_TYPE.get(backend_type, "Uncategorised")
