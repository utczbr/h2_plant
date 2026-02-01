# Hydrogen Plant GUI - User Guide

## 1. Getting Started

### Launching the Application
To start the graphical interface, run the following command from the project root:

```bash
python h2_plant/gui/main.py
```

The main window will appear showing:
- **Left**: Nodes Palette (organized by category)
- **Center**: Node graph canvas
- **Right**: Properties panel

---

## 2. GUI Versions

**v1.0 (Legacy)**: System-level nodes (Electrolyzer, ATR)  
**v2.0**: Component-level nodes (PEM Stack, Rectifier, Heat Exchanger, etc.)  
**v2.1 (Current)**: Enhanced nodes with collapse functionality and organized property tabs

Both versions work! Use v2.1 for detailed plant design with improved UX.

---

## 3. Basic Workflow

### Step 1: Adding Components
**Using the Palette (Recommended)**:
1. Browse categories in **"Nodes"** panel (left sidebar)
2. Drag component onto canvas
3. Release to place

**Using Context Menu**:
1. Right-click on canvas
2. Navigate categories
3. Click to place

**Available Categories**:
- Electrolysis (PEM, SOEC, Rectifier)
- Reforming (ATR, WGS, Steam Generator)
- Separation (PSA, Separation Tank)
- Thermal (Heat Exchanger)
- Fluid (Compressor, Pump)
- Resources (Grid, Water, Heat, Natural Gas)
- Storage, Compression, Logic, Utilities

### Step 1.5: Understanding Node States
**Collapsed vs Expanded**:
- **Collapsed (Default)**: Nodes appear compact showing only ports and ID
  - Cleaner visual presentation
  - Easier to navigate complex graphs
- **Expanded**: Nodes show full property configuration panels
  - Click **arrow button** (bottom-right corner) to toggle
  - Arrow points down (▼) when collapsed, up (▲) when expanded

**When to Expand**:
- To configure component parameters
- To view/edit detailed settings
- To verify current configuration

### Step 2: Assigning to Systems
**Important**: Some components need system assignment!

Components requiring assignment:
- Rectifier → Choose PEM or SOEC
- Heat Exchanger, Steam Generator, PSA, Compressor, Pump → Choose PEM, SOEC, or ATR

**How to assign**:
1. Click component
2. In Properties panel, find **`system`** dropdown
3. Select: PEM (0), SOEC (1), or ATR (2)

### Step 3: Connecting Components
1. Click output port (circle) on source node
2. Drag to input port on target node
3. Release to create connection

**Port Colors**:
- Cyan: Hydrogen
- Orange: Oxygen  
- Yellow: Electricity
- Red: Heat
- Blue: Water
- Grey: Natural Gas/CO2

### Step 4: Configuring Properties
1. **Expand the node** (click arrow button in bottom-right corner)
2. **Select a component** to view properties panel (right sidebar)
3. **Navigate property tabs**:
   - **Properties Tab**: Component identification
     - `component_id`: Unique identifier (e.g., "PEM-Stack-1")
   - **Component-Specific Tab**: Operational parameters
     - Parameters show **units in parentheses** (kW, bar, °C, %, kg/h)
     - Examples: `rated_power_kw (kW)`, `efficiency_rated (%)`
     - Spinboxes enforce min/max validation ranges
   - **Custom Tab**: Visual customization
     - `node_color`: Color picker for node appearance
     - `custom_label`: Optional display label
4. **For resource nodes** (Grid, Water, Heat, Gas):
   - Set `supply_mode`: Choose from `on_demand`, `scaled`, or `constant`
   - Set `mode_value`: Meaning depends on selected mode
5. **Collapse the node** when done to keep canvas clean

### Step 5: Running Simulation
1. Click **"Run Simulation"** (toolbar)
2. Wait for progress dialog
3. Review results in Results Dialog

---

## 4. Design Examples

### Example 1: Simple PEM System
1. Add: Grid Connection
2. Add: Rectifier (set `system=PEM`)
3. Add: PEM Stack
4. Add: Heat Exchanger (set `system=PEM`)
5. Add: LP Tank
6. Connect: Grid → Rectifier → PEM Stack → Tank

### Example 2: Complete ATR System
See `docs/GUI/DETAILED_COMPONENTS_GUIDE.md` for comprehensive workflows

---

## 5. Keyboard Shortcuts

- **Del**: Delete selected nodes
- **F**: Fit zoom to selection
- **H**: Reset zoom
- **Ctrl+Z**: Undo (if supported by NodeGraphQt)

---

## 6. Troubleshooting

**Nodes not appearing in palette:**
- Restart application
- Check console for import errors

**"Schema validation failed":**
- Ensure all required properties set
- Check system assignments for shared components

**Node movement error (KeyError):**
- Harmless NodeGraphQt issue
- Ignore or restart if persistent

**Simulation won't run:**
- Check all components have valid properties
- Ensure at least one production source connected to storage

---

## 7. Advanced Features

### Collapse/Expand Nodes
- **Default State**: All nodes start collapsed for cleaner graphs
- **Toggle**: Click arrow button (bottom-right corner of node)
- **Keyboard**: Select node + double-click to expand
- **Visual Indicator**: 
  - ▼ (down arrow) = Collapsed, click to expand
  - ▲ (up arrow) = Expanded, click to collapse

### Resource Supply Modes
Grid Connection, Water Supply, Ambient Heat, and Natural Gas nodes support flexible supply strategies:

- **`on_demand`** (default): Provides exactly what downstream components request
- **`scaled`**: Multiplies demand by `mode_value` factor
  - Example: Water Supply with `mode_value=1.5` provides 1.5× requested flow
- **`constant`**: Fixed maximum availability of `mode_value` units
  - Example: Grid with `mode_value=10000 kW` caps at 10 MW

**Environment Manager**: Automatically provides time-series wind and pricing data

**Data Files** (in `h2_plant/data/`):
- `ATR_model_functions.pkl`: ATR reactor model
- `EnergyPriceAverage2023-24.csv`: Historical prices
- `wind_data.csv`: Wind power data

**Documentation**:
- Detailed Components: `docs/GUI/DETAILED_COMPONENTS_GUIDE.md`
- Phase 5 Changes: `docs/GUI/PHASE5_CHANGES.md`
- Template Config: `configs/standard_plant_template.yaml`

---

## 8. Support

For detailed component reference, see:  
📖 **`docs/GUI/DETAILED_COMPONENTS_GUIDE.md`**
