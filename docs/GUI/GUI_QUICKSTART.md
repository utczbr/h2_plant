# GUI Development: Quick Start & Action Items

**Date:** November 21, 2025  
**Status:** Ready to begin development

---

## 📋 Executive Summary

You need a **visual node editor** for hydrogen plant configurations. 

**Stack:** PySide6 + NodeGraphQt + Python threading  
**Delivery:** 4-week phased prototype  
**Risk:** LOW (isolated from simulation core)  
**Impact:** Zero changes to existing simulation logic

---

## 🎯 Phase 1 Actions (Days 1-3)

### Pre-Development Checklist

- [ ] Create `requirements.txt`:
  ```text
  PySide6>=6.0.0
  NodeGraphQt>=0.6.0
  pyyaml>=6.0
  ```
- [ ] Set up Virtual Environment:
  ```bash
  python -m venv venv_gui
  source venv_gui/bin/activate  # Linux/Mac
  # venv_gui\Scripts\activate  # Windows
  pip install -r requirements.txt
  ```
- [ ] Install dependencies (if not using requirements.txt):
  ```bash
  pip install PySide6 NodeGraphQt pyyaml
  ```

- [ ] Create project structure:
  ```
  h2_plant/gui/
  ├── __init__.py
  ├── main.py
  ├── ui/
  ├── nodes/
  ├── core/
  └── styles/
  ```

- [ ] Verify existing code:
  - [ ] `PlantBuilder.from_file()` works
  - [ ] `plant_schema_v1.json` is valid
  - [ ] `PlantConfig` dataclass exists

---

## 🛠️ Modifications Required (Minimal)

### 1. Add `PlantBuilder.from_dict()` (5 minutes)

**File:** `h2_plant/config/plant_builder.py`

```python
@classmethod
def from_dict(cls, config_dict: Dict[str, Any]) -> 'PlantBuilder':
    """Build plant from Python dict (GUI produces this)."""
    # Convert dict → dataclass → builder
    # Same logic as from_file() but starting from dict
    config = PlantConfig.from_dict(config_dict)
    config.validate()
    builder = cls(config)
    builder.build()
    return builder
```

### 2. Create Adapter Module (60 minutes)

**File:** `h2_plant/gui/core/graph_adapter.py`

- **Dataclasses:** `GraphNode`, `GraphEdge`, `Port`, `FlowType`
- **Main class:** `GraphToConfigAdapter`
- **Methods:**
  - `add_node()`
  - `add_edge()`
  - `to_config_dict()` → returns PlantConfig-compatible dict
  - `validate()` → checks for logical errors

### 3. Create Schema Inspector (45 minutes)

**File:** `h2_plant/gui/core/schema_inspector.py`

- Read your JSON schema
- Provide validation rules to GUI
- Generate widget constraints dynamically

### 4. Create Node Base Class (90 minutes)

**File:** `h2_plant/gui/nodes/base_node.py`

- Abstract base: `ConfigurableNode`
- Port system with type checking
- Property system with validators
- `to_dict()` and `from_dict()` serialization

---

## 📦 Deliverables by Week

### Week 1: Foundation

| Day | Deliverable | Acceptance Criteria |
|-----|-------------|-------------------|
| **Mon** | Canvas setup | App launches, empty canvas |
| **Tue** | Create/delete nodes | Right-click menu, nodes appear |
| **Wed** | Property inspector | Select node → properties show |
| **Thu** | Port connections | Draw lines between nodes |

### Week 2: Validation & Export

| Day | Deliverable | Acceptance Criteria |
|-----|-------------|-------------------|
| **Mon** | Schema binding | Spinboxes enforce min/max |
| **Tue** | Connection validation | Enum prevents bad connections |
| **Wed** | Config export | "Export" button → YAML string |
| **Thu** | Import from YAML | Load saved config → nodes appear |

### Week 3: Simulation Integration

| Day | Deliverable | Acceptance Criteria |
|-----|-------------|-------------------|
| **Mon** | Threading infrastructure | "Run" doesn't freeze GUI |
| **Tue** | Progress bar | Shows 0% → 100% |
| **Wed** | Results loading | CSV/JSON parsed from disk |
| **Thu** | Chart rendering | Matplotlib/Plotly widget shows data |

### Week 4: Polish & Testing

| Day | Deliverable | Acceptance Criteria |
|-----|-------------|-------------------|
| **Mon** | Error dialogs | User-friendly error messages |
| **Tue** | Keyboard shortcuts | Ctrl+Z undo, Ctrl+S save |
| **Wed** | Dark theme | Minimalist Antigravity style |
| **Thu** | Unit tests + docs | 80%+ coverage, README complete |

---

## 🚀 Getting Started (Today)

### Step 1: Install Dependencies

```bash
# Create virtual environment (optional but recommended)
python -m venv venv_gui
source venv_gui/bin/activate  # On Windows: venv_gui\Scripts\activate

# Install PySide6 and NodeGraphQt
pip install PySide6 NodeGraphQt pyyaml
```

### Step 2: Create Skeleton Project

```python
# h2_plant/gui/main.py
from PySide6.QtWidgets import QApplication, QMainWindow
from nodegraphqt import NodeGraph

class PlantEditorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("H₂ Plant Configuration Editor")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create empty node graph
        self.graph = NodeGraph()
        self.setCentralWidget(self.graph.widget)
        
        self.show()

if __name__ == "__main__":
    app = QApplication([])
    window = PlantEditorWindow()
    app.exec()
```

### Step 3: Verify Graph Adapter Works

```python
# Test the adapter in isolation (before integrating with PySide6)
from h2_plant.gui.core.graph_adapter import GraphToConfigAdapter, GraphNode
from h2_plant.config.plant_builder import PlantBuilder

# Create adapter
adapter = GraphToConfigAdapter()

# Add nodes
node = GraphNode(
    id="e1",
    type="ElectrolyzerNode",
    display_name="Electrolyzer 1",
    x=0, y=0,
    properties={"max_power_mw": 5.0, "efficiency": 0.68},
    ports=[]
)
adapter.add_node(node)

# Export
config_dict = adapter.to_config_dict()
print(config_dict)

# Verify PlantBuilder can consume it
plant = PlantBuilder.from_dict(config_dict)
print("✓ Adapter works!")
```

---

## 📌 Key Design Decisions

### 1. Node Types → Pre-Defined Catalog

**NOT** a free-text component system. Users choose from:

```
Production:
  • Electrolyzer
  • ATR
  
Storage:
  • LP Tank
  • HP Tank
  • Oxygen Buffer
  
Compression:
  • Filling Compressor
  • Outgoing Compressor
  
Utility:
  • Demand Scheduler
  • Energy Price Tracker
```

Each has **predefined ports** and **validated properties**.

### 2. Direction Matters

Connections have **semantic meaning**:
- Electrolyzer "H₂ output" → Tank "H₂ input" ✅
- Electrolyzer "H₂ output" → Electricity input ❌

Port types prevent invalid connections.

### 3. No GUI Simulation

The GUI exports **configuration only**. Simulation runs in a **separate thread** (or even separate process in future):

```
GUI Thread          Worker Thread
─────────────────  ───────────────────
User clicks "Run"
                   → PlantBuilder.from_dict()
                   → SimulationEngine.run()
                   → Progress callbacks
UI responsive
                   ← Results saved
Load results  ← Emission complete
```

---

## 🎨 Minimalist Aesthetic Guidelines

### Color Palette

```
Canvas Background:     #1a1a1a
Node Background:       #2d2d2d
Node Border (hover):   #00bfff (cyan)
Text Primary:          #e0e0e0
Text Secondary:        #888888
Accent (H₂):          #00ffff (cyan)
Accent (Power):       #ffaa00 (orange)
Accent (Heat):        #ff6b6b (red)
Success:              #00d96f (green)
Error:                #ff5555 (bright red)
```

### Typography

- **Font:** "Segoe UI", system sans-serif
- **Node title:** 12pt bold
- **Property label:** 10pt regular
- **Button:** 11pt medium

### Spacing

- **Node padding:** 12px
- **Port size:** 8px diameter
- **Connection width:** 2px (Bezier curves)
- **Grid:** 20px cells (snapping)

---

## 🔍 Code Organization

### Minimal Viable Product (MVP)

```
h2_plant/gui/
├── main.py                    # Entry point
├── ui/
│   └── main_window.py         # PySide6 window
├── nodes/
│   ├── base_node.py          # Abstract base
│   └── production_nodes.py    # Electrolyzer, ATR
├── core/
│   ├── graph_adapter.py       # Graph → config conversion
│   ├── schema_inspector.py    # Schema validation
│   └── simulation_worker.py   # Threading
└── styles/
    └── stylesheet.py          # Dark theme
```

### Scale-Up (Future)

```
h2_plant/gui/
├── nodes/
│   ├── production_nodes.py
│   ├── storage_nodes.py
│   ├── compression_nodes.py
│   ├── mixing_nodes.py
│   ├── external_nodes.py
│   ├── water_nodes.py         # Water treatment
│   └── utility_nodes.py
├── ui/
│   ├── main_window.py
│   ├── property_panel.py
│   ├── results_viewer.py
│   └── dialogs.py
└── core/
    ├── graph_adapter.py
    ├── schema_inspector.py
    ├── simulation_worker.py
    └── node_factory.py        # Create nodes from config
```

---

## 🧪 Testing Approach

### Unit Test Example

```python
# tests/gui/test_graph_adapter.py
import pytest
from h2_plant.gui.core.graph_adapter import (
    GraphToConfigAdapter, GraphNode, FlowType
)

def test_electrolyzer_node_to_config():
    """Electrolyzer node exports to correct config."""
    node = GraphNode(
        id="e1",
        type="ElectrolyzerNode",
        display_name="Electrolyzer 1",
        x=0, y=0,
        properties={"max_power_mw": 5.0, "efficiency": 0.68},
        ports=[]
    )
    
    adapter = GraphToConfigAdapter()
    adapter.add_node(node)
    
    config = adapter.to_config_dict()
    
    assert config["production"]["electrolyzer"]["enabled"]
    assert config["production"]["electrolyzer"]["max_power_mw"] == 5.0
    assert config["production"]["electrolyzer"]["base_efficiency"] == 0.68

def test_invalid_connection_prevented():
    """H₂ output cannot connect to electricity input."""
    # TODO: Once ports are implemented
    pass

def test_export_to_yaml():
    """Config exports to valid YAML."""
    # TODO: Test YAML serialization
    pass
```

### Integration Test Example

```python
# tests/gui/test_e2e.py
def test_gui_config_to_simulation():
    """Complete flow: create graph → export → simulate."""
    # Create a valid graph
    adapter = GraphToConfigAdapter()
    
    # Add electrolyzer
    adapter.add_node(GraphNode(...))
    
    # Add tank
    adapter.add_node(GraphNode(...))
    
    # Connect
    adapter.add_edge(GraphEdge(...))
    
    # Validate
    is_valid, errors = adapter.validate()
    assert is_valid, f"Validation errors: {errors}"
    
    # Export
    config_dict = adapter.to_config_dict()
    
    # Build plant (this calls your PlantBuilder.from_dict)
    from h2_plant.config.plant_builder import PlantBuilder
    plant = PlantBuilder.from_dict(config_dict)
    
    # Verify components registered
    assert plant.registry.has("electrolyzer")
    assert plant.registry.has("lp_tank_array")
```

---

## 💾 File Locations & Responsibilities

| File | Purpose | Complexity | Priority |
|------|---------|-----------|----------|
| `gui/main.py` | App entry point | Low | P0 |
| `gui/ui/main_window.py` | PySide6 window | Medium | P0 |
| `gui/nodes/base_node.py` | Node abstraction | Medium | P0 |
| `gui/core/graph_adapter.py` | Graph → Config | High | P0 |
| `gui/core/schema_inspector.py` | Schema reading | Medium | P1 |
| `gui/core/simulation_worker.py` | Async runner | High | P1 |
| `config/plant_builder.py` | **MODIFY** `from_dict()` | Low | P0 |

---

## ⚠️ Potential Pitfalls & Solutions

| Pitfall | Risk | Solution |
|---------|------|----------|
| GUI freezes during simulation | High | Use threading + callbacks |
| Invalid config crashes PlantBuilder | High | Validate before calling `from_dict()` |
| Port type validation too complex | Medium | Start with "any-to-any" connections, add type safety later |
| Saving/loading graphs breaks | Medium | Implement `to_dict()` / `from_dict()` for nodes early |
| Users create invalid configurations | High | Red error messages + disable "Run" button |

---

## 📞 Support & Escalation

### Questions to Answer Before Development

1. **Schema locations:** Where is `plant_schema_v1.json`? Is it bundled or loaded?
2. **PlantBuilder API:** Does `PlantConfig` have `from_dict()` already?
3. **Threading safety:** Is `PlantBuilder` thread-safe? Need locks?
4. **Simulation output:** What format are results saved? (JSON, HDF5, CSV?)

### Integration Points with Existing Code

- **Input:** Your `plant_schema_v1.json`
- **Input:** Your existing `PlantBuilder.from_file()`
- **Output:** Dictionary passed to `PlantBuilder.from_dict()`
- **Output:** Signals/callbacks for progress (if SimulationEngine supports them)

---

## 📅 Timeline & Milestones

```
Week 1: Foundation
  ├─ Day 1: Canvas + node creation
  ├─ Day 2: Property editing
  ├─ Day 3: Port connections
  └─ Day 4: Graph serialization

Week 2: Integration
  ├─ Day 1: Schema validation
  ├─ Day 2: Export to YAML
  ├─ Day 3: Import from YAML
  └─ Day 4: Config preview/editing

Week 3: Simulation
  ├─ Day 1: Threading infrastructure
  ├─ Day 2: Progress bar
  ├─ Day 3: Results loading
  └─ Day 4: Chart rendering

Week 4: Polish
  ├─ Day 1: Error handling
  ├─ Day 2: Keyboard shortcuts
  ├─ Day 3: Dark theme
  └─ Day 4: Testing + documentation

✅ Prototype ready for review
```

---

## 🎓 Learning Resources

**PySide6:**
- Official docs: https://doc.qt.io/qtforpython/
- Tutorials: https://www.pythonguis.com/

**NodeGraphQt:**
- GitHub: https://github.com/jchanvfx/NodeGraphQt
- Examples: Check `examples/` folder

**Qt Design Patterns:**
- Model-View pattern for data
- Signal/slot for communication
- Threading with QThread or threading.Thread

---

## ✅ Success Criteria (End of Week 1)

- [ ] App launches without errors
- [ ] User can create 2+ nodes
- [ ] User can set node properties
- [ ] User can draw connections
- [ ] Export produces valid YAML
- [ ] Imported YAML recreates graph
- [ ] Graph passes validation
- [ ] Code documented with docstrings
- [ ] Unit tests passing
- [ ] No errors in existing simulation code

---

## 📝 Next Steps

1. **Today:** Approve this roadmap
2. **Tomorrow:** Set up project structure + install dependencies
3. **This Week:** Complete Week 1 deliverables
4. **Review:** Checkpoint with stakeholders (Friday EOD)
5. **Iterate:** Refine based on feedback

---

**Document Status:** ✅ **READY FOR DEVELOPMENT**  
**Questions?** See **Support & Escalation** section above

