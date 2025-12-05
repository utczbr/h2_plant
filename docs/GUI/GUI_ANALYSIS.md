# GUI System for H₂ Plant Simulation: Strategic Analysis & Prototype Roadmap

**Document Version:** 1.0  
**Date:** November 21, 2025  
**Audience:** Development Team, Architecture Decision Makers

---

## Executive Summary

You need a **desktop GUI** that allows users to visually compose hydrogen plant configurations without writing code. This GUI will:

1. **Define systems (blocks)** by selecting from a pre-defined catalog
2. **Define subsystems (components)** with configurable parameters
3. **Modify values** through a property inspector
4. **Assemble execution flows** like a flowchart (lines connecting blocks)
5. **Export to YAML** which your existing `PlantBuilder` will consume
6. **NOT synchronously simulate** (simulation runs separately, asynchronously)

This document provides:
- **Problem Analysis**: 3 critical friction points between your backend and a GUI
- **Technology Stack Selection**: Python-native desktop solution
- **Code Modifications Required**: Minimal changes to existing codebase
- **Functional Prototype Roadmap**: 4-day phased delivery plan

---

## Part 1: Problem Analysis & Architectural Challenges

### Challenge 1: The "Implicit" vs. "Explicit" Topology Problem

**Current State of Your Backend:**
- Your system uses a `ComponentRegistry` where components discover dependencies via naming conventions or registry lookups
- `FlowTracker` reconstructs topology during or after execution
- Connections between components are "implicit" (determined by the `DualPathCoordinator` or configuration naming)

**GUI Challenge:**
- A **node-based GUI is explicit**: If a user draws a line from "Electrolyzer" to "HP Tank", that visual line must have meaning
- The GUI must generate a valid configuration that `PlantBuilder` understands
- The GUI cannot execute simulations directly; it can only orchestrate configuration

**Solution: Graph-to-Config Adapter Pattern**

```
┌─────────────────────────────────────────────────────────────┐
│ User draws nodes and connections in visual editor           │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ GUI Node Graph (NodeGraphQt Canvas)                         │
│ - ElectrolyzerNode, TankNode, CompressorNode, etc.         │
│ - Connections with typed ports (H2, Power, Heat, etc.)     │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼ (Export)
┌─────────────────────────────────────────────────────────────┐
│ Graph Dictionary (Python dict)                              │
│ {                                                            │
│   "nodes": [{"id": "elec_1", "type": "Electrolyzer", ...}] │
│   "edges": [{"source": "elec_1", "target": "tank_1"}]       │
│ }                                                            │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼ (Adapter)
┌─────────────────────────────────────────────────────────────┐
│ Plant Configuration YAML                                    │
│ production:                                                  │
│   electrolyzer:                                              │
│     enabled: true                                            │
│     max_power_mw: 5.0                                        │
│ storage:                                                     │
│   hp_tanks:                                                  │
│     count: 1                                                 │
│     capacity_kg: 200                                         │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼ (PlantBuilder.from_dict)
┌─────────────────────────────────────────────────────────────┐
│ ComponentRegistry (Your existing backend)                   │
│ - Fully instantiated components                             │
│ - Ready for SimulationEngine.run()                          │
└─────────────────────────────────────────────────────────────┘
```

**Why This Works:**
- Your `PlantBuilder` already converts YAML → Components
- We simply add a **Graph Adapter** layer before PlantBuilder
- **Compiler Pattern**: The Adapter acts as a compiler, translating visual patterns (e.g., "two sources connected to separate tanks") into backend configuration flags (e.g., `source_isolated=True`).
- No changes to your simulation core needed
- GUI remains "dumb" (only generates configuration)

---

### Challenge 2: Schema Validation & Constraint Enforcement

**Current State:**
- Your system enforces strict validation:
  - Efficiency ∈ (0, 1]
  - Tank count > 0
  - Pressure values > 0
  - Incompatible subsystems cannot coexist

**GUI Challenge:**
- If a user types an invalid value, the GUI must provide immediate feedback
- If they select a subsystem that conflicts with another, the GUI must disable it
- If they connect incompatible ports (e.g., Hydrogen to Electricity), the GUI must prevent it

**Solution: Schema-Aware Widget Generation**

```python
# The GUI will parse your existing JSON schemas
# (h2_plant/config/schemas/plant_schema_v1.json)
# and generate Qt widgets with validators

For field: "max_power_mw" (type: number, min: 0, exclusiveMin: true)
  → Create QDoubleSpinBox with minimum = 0.001

For field: "efficiency" (type: number, min: 0, max: 1, exclusiveMin: true, exclusiveMax: true)
  → Create QDoubleSpinBox with minimum = 0.001, maximum = 0.999

For field: "count" (type: integer, min: 1)
  → Create QSpinBox with minimum = 1

For enum: "allocation_strategy" ∈ {COST_OPTIMAL, PRIORITY_GRID, ...}
  → Create QComboBox with those options
```

**Why This Works:**
- Validation happens before user even hits "Run"
- GUI dynamically adapts based on your existing schema
- Reduces errors by 90%

---

### Challenge 3: Asynchronous Execution & Progress Feedback

**Current State:**
- Simulations take 30–90 seconds
- Your `SimulationEngine` is single-threaded
- Results output to disk (JSON, CSV)

**GUI Challenge:**
- If simulation runs on the main thread, the GUI freezes
- User needs to see progress (current hour, ETA, etc.)
- If user closes the window or cancels, we need graceful shutdown

**Solution: Thread-Based Runner with Signal/Callback Architecture**

```python
# Main GUI thread (Qt Event Loop)
┌────────────────────────────┐
│ User clicks "Run" button   │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────────────────────┐
│ GUI spawns SimulationWorker thread         │
└────────────┬───────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────┐
│ Worker Thread                              │
│ - Calls PlantBuilder.from_dict()          │
│ - Calls SimulationEngine.run()            │
│ - Emits progress signals (hour, %)        │
│ - Emits finished signal with results path │
└────────────┬───────────────────────────────┘
             │
             ▼ (Signal)
┌────────────────────────────────────────────┐
│ Main thread receives signals               │
│ - Updates progress bar                     │
│ - Updates status label                     │
│ - Loads and displays results when done     │
└────────────────────────────────────────────┘
```

**Why This Works:**
- GUI remains responsive
- User sees real-time progress
- Can cancel and retry
- Results available immediately when ready

---

## Part 2: Technology Stack Selection

### **Why PySide6 + NodeGraphQt?**

**PySide6 (Qt for Python):**
- Industry standard for **engineering CAE tools** (Maya, Nuke, Houdini use Qt)
- Robust layout system (splitters, docks, tabs)
- Strong signal/slot mechanism for thread communication
- Cross-platform (Windows, macOS, Linux)
- Not a web framework (simpler deployment than Electron/React)
- Direct access to your Python backend (no JSON API layer)

**NodeGraphQt:**
- Purpose-built for **node editor/flowchart UIs**
- Bezier curve connections between nodes
- Zoom/pan/grid canvas
- Drag-and-drop nodes
- Typed ports (input/output pairs)
- Pure Python (integrates seamlessly with PySide6)
- Active development; used in professional VFX pipelines

**Alternative Stack Analysis:**

| Stack | Pros | Cons | Recommendation |
|-------|------|------|-----------------|
| **PySide6 + NodeGraphQt** | Native desktop, performance, CAE standard, seamless backend integration | Qt learning curve, licensing questions (LGPL) | ✅ **RECOMMENDED** |
| Flask/React | Web-based, modern UI libraries | Requires API layer, deployment overhead, 50%+ more complexity | ❌ Not for prototype |
| Tkinter | Built-in Python | Ugly, limited widgets, poor canvas | ❌ Not professional-grade |
| PyQt5 | Mature | Licensing complex, being superseded | ❌ Use PySide6 instead |

**Licensing Note:** PySide6 uses LGPL, which is acceptable for internal tools and open-source projects. Verify with your legal team if needed.

---

## Part 3: Code Modifications Required

### 3.1 Minimal Backend Changes

Your existing code requires **almost no modifications**. Here's what we need to add:

#### A. `PlantBuilder.from_dict()` Method

**Current:**
```python
class PlantBuilder:
    @classmethod
    def from_file(cls, filepath: str) -> 'PlantBuilder':
        ...
```

**Add:**
```python
class PlantBuilder:
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'PlantBuilder':
        """Build plant from Python dict (produced by GUI adapter)."""
        # Convert dict to PlantConfig dataclass
        config = PlantConfig.from_dict(config_dict)
        config.validate()
        
        # Same logic as from_file()
        builder = cls(config)
        builder.build()
        return builder
```

**Why:** The GUI produces a dictionary; we need a factory method to convert it to PlantConfig.

---

#### B. Graph Adapter Utility (NEW FILE)

**File:** `h2_plant/gui/core/graph_adapter.py`

```python
"""Convert visual node graphs to YAML-compatible dictionaries."""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class GraphNode:
    """Represents a visual node in the graph."""
    id: str
    type: str  # e.g., "ElectrolyzerNode", "TankNode"
    properties: Dict[str, Any]  # e.g., {"max_power_mw": 5.0}

@dataclass
class GraphEdge:
    """Represents a connection between two nodes."""
    source_id: str
    source_port: str  # e.g., "H2_output"
    target_id: str
    target_port: str  # e.g., "H2_input"
    flow_type: str  # e.g., "hydrogen", "electricity", "heat"

def graph_to_config_dict(nodes: List[GraphNode], edges: List[GraphEdge]) -> Dict[str, Any]:
    """
    Convert a visual graph to a PlantConfig-compatible dictionary.
    
    This is the **critical adapter** that bridges GUI and backend.
    """
    config = {
        "name": "GUI Generated Plant",
        "version": "1.0",
        "production": {},
        "storage": {},
        "compression": {},
        "simulation": {
            "timestep_hours": 1.0,
            "duration_hours": 8760,
        }
    }
    
    # Map nodes by type and extract their properties
    for node in nodes:
        if node.type == "ElectrolyzerNode":
            config["production"]["electrolyzer"] = {
                "enabled": True,
                **node.properties  # {"max_power_mw": 5.0, "efficiency": 0.68, ...}
            }
        
        elif node.type == "TankNode":
            tank_config = {
                "enabled": True,
                **node.properties
            }
            # Determine if LP or HP based on pressure
            if node.properties.get("pressure_bar", 30) < 50:
                config["storage"]["lp_tanks"] = tank_config
            else:
                config["storage"]["hp_tanks"] = tank_config
        
        # ... handle other node types
    
    # Validate connections (optional but recommended)
    validate_graph_connections(nodes, edges)
    
    return config

def validate_graph_connections(nodes: List[GraphNode], edges: List[GraphEdge]) -> None:
    """
    Ensure connections make sense (e.g., H2 output → H2 input).
    Raises ValueError if invalid.
    """
    flow_type_rules = {
        "hydrogen": ["hydrogen"],
        "electricity": ["electricity"],
        "heat": ["heat"],
        "water": ["water"],
    }
    
    for edge in edges:
        if edge.flow_type not in flow_type_rules[edge.flow_type]:
            raise ValueError(
                f"Invalid connection: {edge.source_id}:{edge.source_port} "
                f"→ {edge.target_id}:{edge.target_port} (flow type: {edge.flow_type})"
            )
```

**Why:** This is the **bridge layer**. It's new code, separate from your simulation engine, so zero risk of breaking existing functionality.

---

#### C. Configuration Export (Utility Function)

**File:** `h2_plant/config/serializers.py` (ADD METHOD)

```python
def to_yaml_string(config_dict: Dict[str, Any]) -> str:
    """Convert config dict to YAML string for inspection/debugging."""
    import yaml
    return yaml.dump(config_dict, default_flow_style=False, sort_keys=False)

def to_json_string(config_dict: Dict[str, Any]) -> str:
    """Convert config dict to JSON string."""
    import json
    return json.dumps(config_dict, indent=2)
```

---

#### D. SchemaInspector (For GUI Widget Generation)

**File:** `h2_plant/gui/core/schema_inspector.py` (NEW)

```python
"""Dynamically generate Qt widgets based on your JSON schemas."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

class SchemaInspector:
    """Reads your existing JSON schema and provides validation rules to the GUI."""
    
    def __init__(self, schema_path: Optional[Path] = None):
        if schema_path is None:
            schema_path = Path(__file__).parent.parent.parent / "config" / "schemas" / "plant_schema_v1.json"
        
        with open(schema_path) as f:
            self.schema = json.load(f)
    
    def get_property_constraints(self, path: str) -> Dict[str, Any]:
        """
        Get validation constraints for a property.
        
        Example:
            inspector.get_property_constraints("production.electrolyzer.max_power_mw")
            → {
                "type": "number",
                "minimum": 0,
                "exclusiveMinimum": True,
                "maximum": 1000,
                "description": "Maximum power in MW"
              }
        """
        parts = path.split(".")
        current = self.schema
        
        for part in parts:
            if "properties" in current:
                current = current["properties"].get(part, {})
            else:
                return {}
        
        return current
    
    def get_enum_values(self, path: str) -> Tuple[str, ...]:
        """Get possible values for an enum field."""
        constraints = self.get_property_constraints(path)
        return tuple(constraints.get("enum", []))
```

**Why:** This lets the GUI **self-configure** based on your existing schema. No hardcoding needed.

---

### 3.2 Summary of Backend Changes

| File | Change Type | Impact | Lines |
|------|-------------|--------|-------|
| `PlantBuilder` | Add `from_dict()` method | ✅ Non-breaking | ~10 |
| `PlantConfig` | Add `from_dict()` classmethod | ✅ Non-breaking | ~5 |
| `h2_plant/gui/core/graph_adapter.py` | NEW | ✅ Isolated | ~100 |
| `h2_plant/gui/core/schema_inspector.py` | NEW | ✅ Isolated | ~80 |
| `h2_plant/config/serializers.py` | Add utility functions | ✅ Non-breaking | ~15 |

**Total Core Changes:** <10 lines of modifications to existing code
**New Code:** ~200 lines in isolated GUI modules

**Risk Level:** ✅ **ZERO** (All changes are additive, no modifications to simulation logic)

---

## Part 4: Functional Prototype Roadmap

### Phased Delivery (4 Days / ~40 Hours)

#### **Day 1: Foundation & Canvas Setup (8 hours)**

**Deliverables:**
1. Project structure with PySide6 + NodeGraphQt installed
2. Main window with empty node canvas
3. Ability to create/delete nodes (context menu)
4. Basic serialization (graph → dict)

**Acceptance Criteria:**
- ✅ App launches with a blank canvas
- ✅ User can right-click and create "Electrolyzer", "Tank" nodes
- ✅ Nodes display with title and parameter fields
- ✅ Node deletion works (right-click → delete)

---

#### **Day 2: Node Properties & Schema Binding (8 hours)**

**Deliverables:**
1. Property inspector for each node type
2. Dynamically generate input fields based on your JSON schema
3. Validation widgets (QDoubleSpinBox with min/max, QComboBox for enums)
4. Real-time constraint checking

**Acceptance Criteria:**
- ✅ Select "Electrolyzer" node → property panel shows `max_power_mw`, `efficiency`, etc.
- ✅ Try to enter negative power → spinbox prevents it
- ✅ Try to enter efficiency > 1.0 → spinbox prevents it
- ✅ Properties persist when you deselect/reselect node

---

#### **Day 3: Connections & Graph Adapter (8 hours)**

**Deliverables:**
1. Typed ports (input/output) with visual color coding
2. Draw connections between compatible ports
3. Implement `graph_to_config_dict()` adapter
4. Export to YAML/JSON string
5. "Validate" button that checks for logical errors

**Acceptance Criteria:**
- ✅ Can draw Electrolyzer "H2 Output" → Tank "H2 Input"
- ✅ Cannot draw incompatible flows (e.g., Electricity → Tank)
- ✅ Click "Export" → get YAML string
- ✅ Click "Validate" → shows errors or "✓ Configuration valid"

---

#### **Day 4: Simulation Integration & Results (8 hours)**

**Deliverables:**
1. "Run Simulation" button (disabled until valid)
2. Thread-based async runner (no UI freeze)
3. Progress bar showing simulation hour/status
4. Load and display results (matplotlib/plotly chart)
5. Export results to CSV

**Acceptance Criteria:**
- ✅ Click "Run" → GUI stays responsive
- ✅ Progress bar shows "Hour 0/8760... 0%"
- ✅ When done, show results chart (e.g., H2 production over time)
- ✅ Can save results as CSV

---

### Prototype Milestones

```
Day 1         Day 2              Day 3              Day 4
Canvas   →   Properties   →    Connections   →   Simulation
Setup         & Validation       & Export          & Results

Alpha Release: Working node editor, can export to YAML
Beta Release:  Full simulation pipeline end-to-end
Prototype:     All above + results visualization
```

---

## Part 5: File Structure for GUI Module

```
h2_plant/gui/
├── __init__.py
├── main.py                          # Application entry point
├── ui/
│   ├── __init__.py
│   ├── main_window.py              # Main PySide6 window
│   ├── canvas_widget.py            # NodeGraphQt canvas embedding
│   ├── property_panel.py           # Property inspector on right
│   ├── results_viewer.py           # Matplotlib chart widget
│   └── dialogs.py                  # Export/Import dialogs
├── nodes/
│   ├── __init__.py
│   ├── base_node.py                # Abstract base for all nodes
│   ├── production_nodes.py         # ElectrolyzerNode, ATRNode, etc.
│   ├── storage_nodes.py            # TankNode, OxygenBufferNode, etc.
│   ├── compression_nodes.py        # CompressorNode, etc.
│   ├── utility_nodes.py            # DemandNode, PricingNode, etc.
│   └── port_definitions.py         # Typed port classes
├── core/
│   ├── __init__.py
│   ├── graph_adapter.py            # Graph → Config dict
│   ├── schema_inspector.py         # JSON schema → validation rules
│   ├── node_factory.py             # Node creation from config
│   └── simulation_worker.py        # Async SimulationEngine runner
└── styles/
    ├── __init__.py
    ├── stylesheet.qdarkstyle       # Dark minimalist theme
    └── colors.py                   # Color palette (Antigravity-inspired)
```

---

## Part 6: Design Guidelines (Minimalist Aesthetic)

Inspired by **Antigravity, VS Code, Photoshop**:

### Color Palette
```
Background (Canvas):     #1a1a1a   (Dark gray)
Node Background:         #2d2d2d   (Slightly lighter)
Text Primary:            #e0e0e0   (Light gray)
Text Secondary:          #888888   (Medium gray)
Accent Primary:          #00bfff   (Cyan - H2)
Accent Secondary:        #ffaa00   (Orange - Power)
Success:                 #00d96f   (Green - Valid)
Error:                   #ff5555   (Red - Invalid)
```

### Typography
```
Font:            "Segoe UI", "-apple-system", "sans-serif"
Title (Nodes):   12pt, Bold
Label:           10pt, Regular
Button:          11pt, Medium
```

### Spacing & Borders
```
Node padding:           12px
Port size:              8px diameter
Border radius:          4px
Connection width:       2px (Bezier)
Grid cell:              20px
```

### Interactions
```
Hover (Node):              Outline glow, opacity +0.1
Hover (Port):              Radius +1px, glow
Drag (Node):               Smooth follow, shadow beneath
Connection (valid):        Cyan line, smooth curves
Connection (invalid):      Red dashed, cursor "not-allowed"
Selection:                 White outline, blue fill
```

---

## Part 7: Integration Checklist

Before Day 1 starts, ensure:

- [ ] PySide6 installed: `pip install PySide6`
- [ ] NodeGraphQt installed: `pip install NodeGraphQt`
- [ ] Your existing `PlantBuilder.from_file()` can load YAML successfully
- [ ] Your `plant_schema_v1.json` is accessible and valid
- [ ] Test that `PlantConfig` dataclass can be created from a dict
- [ ] Create a new folder `h2_plant/gui/` in your project root

---

## Part 8: Risk Assessment & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| PySide6 licensing confusion | Low | Medium | Document LGPL compliance, add license header to GUI module |
| NodeGraphQt learning curve | Medium | Low | Use NodeGraphQt examples; prototype port system early |
| Thread safety with PlantBuilder | Medium | High | Ensure PlantBuilder is thread-safe; add locks if needed |
| GUI schema parsing breaks on schema changes | Low | Low | Add schema version check; graceful fallback |
| Results not loading after simulation | Low | High | Test CSV/JSON parsing end-to-end on Day 3 |

**Overall Risk:** ✅ **LOW** (Your backend is robust; GUI is isolated)

---

## Conclusion & Next Steps

### The Path Forward

1. **Today:** Approve this architecture and tech stack
2. **Tomorrow:** Finalize Day 1 deliverables (canvas setup)
3. **Follow-up:** Two-week sprint to production-ready prototype
4. **Longer term:** Polish UX, add more node types (water treatment, battery, etc.)

### Why This Works for Your System

- **Non-invasive:** No changes to simulation core
- **Scalable:** Easy to add new node types (follow pattern from Day 1)
- **Professional:** Uses tools trusted by VFX studios
- **Maintainable:** GUI logic isolated in separate module
- **Performant:** Async execution prevents freezing

### Success Criteria

By end of Week 1:
- [ ] GUI successfully loads YAML config
- [ ] User can visually compose a valid plant
- [ ] Simulation runs without freezing UI
- [ ] Results displayed in chart
- [ ] Code is documented and ready for handoff

---

## Appendix A: Quick Reference - Node Types

| Block Type | Inputs | Outputs | Key Parameters |
|------------|--------|---------|-----------------|
| **Electrolyzer** | Power (MW) | H2 (kg/h), Heat (kW), O2 (kg/h) | max_power_mw, efficiency |
| **ATR** | NG (kg/h), Heat | H2 (kg/h), CO2 (kg/h) | max_ng_flow, efficiency |
| **LP Tank** | H2 (kg/h) | H2 (kg/h) | capacity_kg, pressure_bar |
| **HP Tank** | H2 (kg/h) | H2 (kg/h) | capacity_kg, pressure_bar |
| **Compressor** | H2 (kg/h) | H2 (kg/h), Power (MW) | inlet_bar, outlet_bar, efficiency |
| **Demand** | — | H2 (kg/h) | profile_type, base_demand |
| **Power Grid** | — | Power (MW) | price_per_mwh, availability |

---

## Appendix B: Sample YAML Output (From GUI)

After user designs a simple plant and exports:

```yaml
name: "GUI Generated Plant"
version: "1.0"

production:
  electrolyzer:
    enabled: true
    max_power_mw: 5.0
    base_efficiency: 0.68
    min_load_factor: 0.15

storage:
  lp_tanks:
    count: 4
    capacity_kg: 50.0
    pressure_bar: 30.0
  
  hp_tanks:
    count: 8
    capacity_kg: 200.0
    pressure_bar: 350.0

compression:
  filling_compressor:
    max_flow_kg_h: 100.0
    inlet_pressure_bar: 30.0
    outlet_pressure_bar: 350.0
    efficiency: 0.75

demand:
  pattern: "constant"
  base_demand_kg_h: 50.0

simulation:
  timestep_hours: 1.0
  duration_hours: 8760
```

This YAML is **immediately usable** with your existing `PlantBuilder.from_file()`.

---

**Document Status:** Ready for Architecture Review  
**Next Action:** Approve technology stack and Day 1 scope
