# Node Properties Quick Reference - Phase 5

Quick reference for all node parameters after Phase 5 enhancements.

---

## Electrolysis Nodes

### PEM Stack
**Identifier**: `h2_plant.electrolysis.pem`

| Property | Unit | Range | Default | Tab |
|----------|------|-------|---------|-----|
| component_id | - | - | PEM-Stack-1 | Properties |
| rated_power_kw | kW | 1.0+ | 2500.0 | PEM Stack |
| efficiency_rated | % | 0-100 | 65.0 | PEM Stack |
| number_of_cells | cells | 1.0+ | 85.0 | PEM Stack |
| active_area_m2 | m² | 0.001+ | 0.03 | PEM Stack |
| operating_temp_c | °C | 20-90 | 60.0 | PEM Stack |
| node_color | RGB | - | (0,255,255) | Custom |
| custom_label | text | - | - | Custom |

### SOEC Stack
**Identifier**: `h2_plant.electrolysis.soec`

| Property | Unit | Range | Default | Tab |
|----------|------|-------|---------|-----|
| component_id | - | - | SOEC-Stack-1 | Properties |
| rated_power_kw | kW | 1.0+ | 1000.0 | SOEC Stack |
| operating_temp_c | °C | 600-1000 | 800.0 | SOEC Stack |
| node_color | RGB | - | (255,200,100) | Custom |
| custom_label | text | - | - | Custom |

### Rectifier/Transformer
**Identifier**: `h2_plant.electrolysis.rectifier`

| Property | Unit | Range | Default | Tab |
|----------|------|-------|---------|-----|
| component_id | - | - | RT-1 | Properties |
| max_power_kw | kW | 1.0+ | 2500.0 | Rectifier |
| conversion_efficiency | % | 0-100 | 98.0 | Rectifier |
| node_color | RGB | - | (255,255,0) | Custom |
| custom_label | text | - | - | Custom |

---

## Reforming Nodes

### ATR Reactor
**Identifier**: `h2_plant.reforming.atr`

| Property | Unit | Range | Default | Tab |
|----------|------|-------|---------|-----|
| component_id | - | - | ATR-Reactor | Properties |
| max_flow_kg_h | kg/h | 1.0+ | 1500.0 | ATR Reactor |
| model_path | path | - | h2_plant/data/ATR_model_functions.pkl | ATR Reactor |
| operating_temp_c | °C | 500-1200 | 900.0 | ATR Reactor |
| node_color | RGB | - | (150,100,255) | Custom |
| custom_label | text | - | - | Custom |

### WGS Reactor
**Identifier**: `h2_plant.reforming.wgs`

| Property | Unit | Range | Default | Tab |
|----------|------|-------|---------|-----|
| component_id | - | - | WGS-HT | Properties |
| conversion_rate | % | 0-100 | 70.0 | WGS Reactor |
| operating_temp_c | °C | 200-500 | 350.0 | WGS Reactor |
| node_color | RGB | - | (150,150,250) | Custom |
| custom_label | text | - | - | Custom |

### Steam Generator
**Identifier**: `h2_plant.reforming.steam_gen`

| Property | Unit | Range | Default | Tab |
|----------|------|-------|---------|-----|
| component_id | - | - | HX-4 | Properties |
| max_flow_kg_h | kg/h | 1.0+ | 500.0 | Steam Generator |
| target_temp_c | °C | 100-300 | 150.0 | Steam Generator |
| efficiency | % | 0-100 | 90.0 | Steam Generator |
| node_color | RGB | - | (255,150,150) | Custom |
| custom_label | text | - | - | Custom |

---

## Separation Nodes

### PSA Unit
**Identifier**: `h2_plant.separation.psa`

| Property | Unit | Range | Default | Tab |
|----------|------|-------|---------|-----|
| component_id | - | - | D-1 | Properties |
| efficiency | % | 0-100 | 85.0 | PSA Unit |
| recovery_rate | % | 0-100 | 90.0 | PSA Unit |
| operating_pressure_bar | bar | 1.0+ | 30.0 | PSA Unit |
| node_color | RGB | - | (200,200,200) | Custom |
| custom_label | text | - | - | Custom |

### Separation Tank
**Identifier**: `h2_plant.separation.tank`

| Property | Unit | Range | Default | Tab |
|----------|------|-------|---------|-----|
| component_id | - | - | ST-1 | Properties |
| volume_m3 | m³ | 0.1+ | 5.0 | Separation Tank |
| operating_pressure_bar | bar | 1.0+ | 30.0 | Separation Tank |
| node_color | RGB | - | (150,150,200) | Custom |
| custom_label | text | - | - | Custom |

---

## Thermal Nodes

### Heat Exchanger
**Identifier**: `h2_plant.thermal.hx`

| Property | Unit | Range | Default | Tab |
|----------|------|-------|---------|-----|
| component_id | - | - | HX-1 | Properties |
| cooling_capacity_kw | kW | 1.0+ | 500.0 | Heat Exchanger |
| outlet_temp_setpoint_c | °C | -50 to 300 | 25.0 | Heat Exchanger |
| node_color | RGB | - | (255,100,100) | Custom |
| custom_label | text | - | - | Custom |

---

## Compression Nodes

### Filling Compressor
**Identifier**: `h2_plant.compression.filling`

| Property | Unit | Range | Default | Tab |
|----------|------|-------|---------|-----|
| component_id | - | - | FC-1 | Properties |
| max_flow_kg_h | kg/h | 1.0+ | 100.0 | Filling Compressor |
| inlet_pressure_bar | bar | 1-100 | 30.0 | Filling Compressor |
| outlet_pressure_bar | bar | 100-900 | 350.0 | Filling Compressor |
| num_stages | stages | 1-5 | 3.0 | Filling Compressor |
| efficiency | % | 0-100 | 75.0 | Filling Compressor |
| power_consumption_kw | kW | 0+ | 50.0 | Filling Compressor |
| node_color | RGB | - | (100,200,255) | Custom |
| custom_label | text | - | - | Custom |

### Outgoing Compressor
**Identifier**: `h2_plant.compression.outgoing`

| Property | Unit | Range | Default | Tab |
|----------|------|-------|---------|-----|
| component_id | - | - | OC-1 | Properties |
| max_flow_kg_h | kg/h | 1.0+ | 100.0 | Outgoing Compressor |
| inlet_pressure_bar | bar | 100-500 | 350.0 | Outgoing Compressor |
| outlet_pressure_bar | bar | 500-1000 | 900.0 | Outgoing Compressor |
| efficiency | % | 0-100 | 75.0 | Outgoing Compressor |
| power_consumption_kw | kW | 0+ | 75.0 | Outgoing Compressor |
| node_color | RGB | - | (150,200,255) | Custom |
| custom_label | text | - | - | Custom |

---

## Storage Nodes

### LP Tank Array
**Identifier**: `h2_plant.storage.lp`

| Property | Unit | Range | Default | Tab |
|----------|------|-------|---------|-----|
| component_id | - | - | LP-Array-1 | Properties |
| tank_count | tanks | 1.0+ | 4.0 | LP Tank Array |
| capacity_per_tank_kg | kg | 1.0+ | 50.0 | LP Tank Array |
| operating_pressure_bar | bar | 1-100 | 30.0 | LP Tank Array |
| min_fill_level | % | 0-100 | 5.0 | LP Tank Array |
| max_fill_level | % | 0-100 | 95.0 | LP Tank Array |
| ambient_temp_c | °C | -40 to 60 | 20.0 | LP Tank Array |
| node_color | RGB | - | (0,255,255) | Custom |
| custom_label | text | - | - | Custom |

### HP Tank Array
**Identifier**: `h2_plant.storage.hp`

| Property | Unit | Range | Default | Tab |
|----------|------|-------|---------|-----|
| component_id | - | - | HP-Array-1 | Properties |
| tank_count | tanks | 1.0+ | 8.0 | HP Tank Array |
| capacity_per_tank_kg | kg | 1.0+ | 200.0 | HP Tank Array |
| operating_pressure_bar | bar | 100-900 | 350.0 | HP Tank Array |
| min_fill_level | % | 0-100 | 5.0 | HP Tank Array |
| max_fill_level | % | 0-100 | 95.0 | HP Tank Array |
| ambient_temp_c | °C | -40 to 60 | 20.0 | HP Tank Array |
| material_type | text | - | Type IV Composite | HP Tank Array |
| node_color | RGB | - | (0,200,255) | Custom |
| custom_label | text | - | - | Custom |

### Oxygen Buffer
**Identifier**: `h2_plant.storage.o2`

| Property | Unit | Range | Default | Tab |
|----------|------|-------|---------|-----|
| component_id | - | - | O2-Buffer-1 | Properties |
| capacity_kg | kg | 1.0+ | 500.0 | Oxygen Buffer |
| operating_pressure_bar | bar | 1-50 | 10.0 | Oxygen Buffer |
| min_fill_level | % | 0-100 | 10.0 | Oxygen Buffer |
| max_fill_level | % | 0-100 | 90.0 | Oxygen Buffer |
| ambient_temp_c | °C | -40 to 60 | 20.0 | Oxygen Buffer |
| node_color | RGB | - | (255,200,0) | Custom |
| custom_label | text | - | - | Custom |

---

## Resource Nodes

### Grid Connection
**Identifier**: `h2_plant.resources.grid`

| Property | Unit | Range | Default | Tab |
|----------|------|-------|---------|-----|
| component_id | - | - | Grid-1 | Properties |
| supply_mode | enum | on_demand/scaled/constant | on_demand | Grid Connection |
| mode_value | kW | 0+ | 10000.0 | Grid Connection |
| node_color | RGB | - | (255,255,0) | Custom |

**Supply Mode Meaning:**
- `on_demand`: Provides exactly what's requested
- `scaled`: Provides `mode_value` × demand
- `constant`: Maximum `mode_value` kW available

### Water Supply
**Identifier**: `h2_plant.resources.water`

| Property | Unit | Range | Default | Tab |
|----------|------|-------|---------|-----|
| component_id | - | - | Water-Supply-1 | Properties |
| supply_mode | enum | on_demand/scaled/constant | on_demand | Water Supply |
| mode_value | m³/h | 0+ | 100.0 | Water Supply |
| node_color | RGB | - | (100,150,255) | Custom |

### Ambient Heat Source
**Identifier**: `h2_plant.resources.heat`

| Property | Unit | Range | Default | Tab |
|----------|------|-------|---------|-----|
| component_id | - | - | Ambient-Heat-1 | Properties |
| supply_mode | enum | on_demand/scaled/constant | on_demand | Ambient Heat |
| mode_value | kW | 0+ | 1000.0 | Ambient Heat |
| node_color | RGB | - | (255,100,100) | Custom |

### Natural Gas Supply
**Identifier**: `h2_plant.resources.ng`

| Property | Unit | Range | Default | Tab |
|----------|------|-------|---------|-----|
| component_id | - | - | NG-Supply-1 | Properties |
| supply_mode | enum | on_demand/scaled/constant | on_demand | Natural Gas Supply |
| mode_value | kg/h | 0+ | 500.0 | Natural Gas Supply |
| node_color | RGB | - | (200,200,200) | Custom |

---

## Port Colors

| Flow Type | Color | RGB |
|-----------|-------|-----|
| Hydrogen | Cyan | (0, 255, 255) |
| Compressed H₂ | Light Cyan | (0, 200, 255) |
| Oxygen | Orange | (255, 200, 0) |
| Electricity | Yellow | (255, 255, 0) |
| Heat | Red | (255, 100, 100) |
| Water | Blue | (100, 150, 255) |
| Gas (Natural Gas/CO₂) | Grey | (200, 200, 200) |
| Default | Grey | (128, 128, 128) |

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| **Del** | Delete selected nodes |
| **F** | Fit zoom to selection |
| **H** | Reset zoom |
| **Double-click** | Expand selected node |
| **Ctrl+Z** | Undo (if supported) |

---

**Reference**: Phase 5 - November 2025  
**See Also**: `PHASE5_CHANGES.md`, `GUI_USER_GUIDE.md`
