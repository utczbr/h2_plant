# H2 Plant Visualization System

Comprehensive visualization framework for H2 Plant simulation with configurable graph generation, multi-format export, and dashboard creation.

## Features

- ✅ **Configurable Graph Generation**: Enable/disable individual graphs or entire categories
- ✅ **Multiple Export Formats**: HTML (interactive), PNG, PDF, SVG
- ✅ **Automated Dashboards**: Multi-graph HTML dashboards
- ✅ **Modular Architecture**: Easy to extend with new graphs
- ✅ **Performance Optimized**: Handles large datasets with downsampling
- ✅ **Category-Based Organization**: Production, Performance, Economics, SOEC Ops, Storage

## Quick Start

### Installation

```bash
# Required dependencies
pip install plotly pandas numpy

# Optional (for PNG/PDF export)
pip install kaleido
```

### Basic Usage

```python
from h2_plant.visualization import MetricsCollector, GraphGenerator, GRAPH_REGISTRY

# 1. Collect metrics during simulation
collector = MetricsCollector()

# In your simulation loop:
for hour in range(672):
    # ... run simulation ...
    component_states = registry.get_all_states()
    collector.collect_step(hour, component_states)

# 2. Configure graphs
GRAPH_REGISTRY.enable_category('production')  # Enable all production graphs
GRAPH_REGISTRY.disable('tank_storage_timeline')  # Disable specific graph

# 3. Generate graphs
generator = GraphGenerator(collector)
all_graphs = generator.generate_all_enabled()

# 4. Export
generator.export_all('output/graphs', format='html')
```

## Graph Categories

### Production (`production`)
- `pem_h2_production_over_time` - PEM H2 production rate timeline
- `soec_h2_production_over_time` - SOEC H2 production rate timeline
- `total_h2_production_stacked` - Stacked area chart (PEM + SOEC)
- `cumulative_h2_production` - Cumulative production over time

### Performance (`performance`)
- `pem_cell_voltage_over_time` - PEM cell voltage monitoring (should be 1.8-2.0V)
- `pem_efficiency_over_time` - PEM system efficiency (% LHV)

### Economics (`economics`)
- `energy_price_over_time` - Energy price timeline
- `dispatch_strategy_stacked` - Power allocation (PEM/SOEC/Grid)
- `power_consumption_breakdown_pie` - Energy consumption pie chart

### SOEC Operations (`soec_ops`)
- `soec_active_modules_over_time` - Number of active SOEC modules (0-6)

### Storage (`storage`) **[Not Yet Implemented]**
- `tank_storage_timeline` - Tank fill levels over time (placeholder)

## Configuration

### YAML Configuration

Create a `visualization_config.yaml`:

```yaml
visualization:
  categories:
    production: true
    performance: true
    economics: true
    storage: false
  
  graphs:
    pem_cell_voltage_over_time: true
    tank_storage_timeline: false
  
  export:
    output_directory: "simulation_output/{simulation_name}/graphs"
    formats: ['html']
    
    dashboard:
      enabled: true
      title: "H2 Plant Dashboard"
      include_graphs:
        - total_h2_production_stacked
        - dispatch_strategy_stacked
        - pem_cell_voltage_over_time
```

### Programmatic Configuration

```python
from h2_plant.visualization import GRAPH_REGISTRY

# Enable/disable categories
GRAPH_REGISTRY.enable_category('production')
GRAPH_REGISTRY.disable_category('storage')

# Enable/disable individual graphs
GRAPH_REGISTRY.enable('pem_cell_voltage_over_time')
GRAPH_REGISTRY.disable('tank_storage_timeline')

# Disable all, then enable specific ones
GRAPH_REGISTRY.disable_all()
GRAPH_REGISTRY.enable('total_h2_production_stacked')
GRAPH_REGISTRY.enable('pem_cell_voltage_over_time')

# Check enabled graphs
print(GRAPH_REGISTRY.list_enabled())
```

## Advanced Usage

### Creating Custom Graphs

```python
from h2_plant.visualization.graph_catalog import GraphMetadata, GraphPriority, GraphLibrary
import plotly.graph_objects as go

def my_custom_graph(data, **kwargs):
    """Custom graph function."""
    timestamps = data.get('timestamps', [])
    values = data['custom'].get('metric', [])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timestamps, y=values))
    fig.update_layout(title=kwargs.get('title', 'Custom Graph'))
    return fig

# Register the custom graph
GRAPH_REGISTRY.register(GraphMetadata(
    graph_id='my_custom_graph',
    title='My Custom Metric',
    description='Description of the graph',
    function=my_custom_graph,
    library=GraphLibrary.PLOTLY,
    data_required=['custom.metric', 'timestamps'],
    priority=GraphPriority.MEDIUM,
    category='custom',
    enabled=True
))
```

### Exporting to Multiple Formats

```python
# HTML (interactive, default)
generator.export('pem_cell_voltage_over_time', 'output/voltage.html', format='html')

# PNG (requires kaleido)
generator.export('pem_cell_voltage_over_time', 'output/voltage.png', 
                 format='png', width=1200, height=600)

# PDF (requires kaleido)
generator.export('pem_cell_voltage_over_time', 'output/voltage.pdf',
                 format='pdf')
```

### Creating Dashboards

```python
# Specify custom graph order
dashboard_html = generator.create_dashboard(
    graph_ids=[
        'total_h2_production_stacked',
        'dispatch_strategy_stacked',
        'pem_cell_voltage_over_time',
        'cumulative_h2_production'
    ],
    title="Custom H2 Plant Dashboard"
)

# Save to file
with open('output/dashboard.html', 'w') as f:
    f.write(dashboard_html)
```

## Integration with SimulationEngine

To integrate with your simulation, add metrics collection to `SimulationEngine`:

```python
# In h2_plant/simulation/engine.py

from h2_plant.visualization import MetricsCollector, GraphGenerator

class SimulationEngine:
    def __init__(self, ...):
        # ... existing code ...
        self.metrics_collector = MetricsCollector()
    
    def _execute_timestep(self, hour: int):
        # Existing timestep logic
        self.registry.step_all(hour)
        self.flow_network.execute_all_flows(hour)
        
        # Collect metrics
        component_states = self.registry.get_all_states()
        self.metrics_collector.collect_step(hour, component_states)
    
    def run(self, ...):
        # ... run simulation ...
        
        # Generate graphs after simulation
        generator = GraphGenerator(self.metrics_collector)
        generator.generate_all_enabled()
        generator.export_all(f'{output_dir}/graphs', format='html')
        
        # Create dashboard
        dashboard_html = generator.create_dashboard(title=f"{self.config.plant_name} Dashboard")
        with open(f'{output_dir}/dashboard.html', 'w') as f:
            f.write(dashboard_html)
```

## Performance Considerations

For large simulations (>10,000 timesteps):

1. **Downsampling**: Use every Nth data point
2. **WebGL Rendering**: Use `scattergl` instead of `scatter` in Plotly
3. **Lazy Loading**: Generate graphs on-demand instead of all at once
4. **Caching**: Generated graphs are cached in `GraphGenerator._generated_graphs`

## Troubleshooting

### "plotly not installed"
```bash
pip install plotly
```

### "Cannot export PNG/PDF"
```bash
pip install kaleido
```

### "Graph data missing"
Check that the MetricsCollector is capturing the required component states:
```python
collector.summary()  # Shows what data has been collected
generator.summary()  # Shows enabled graphs and data availability
```

## Future Enhancements

- [ ] Tank storage 3D visualization
- [ ] SOEC state machine Gantt chart
- [ ] Real-time Plotly Dash dashboard
- [ ] Seaborn statistical plots
- [ ] 3D surface plots for performance analysis
- [ ] Calendar heatmaps for utilization

## Example Output

See `examples/visualization_example.py` for a complete demonstration.

Generated dashboard will include:
- Interactive Plotly graphs with zoom, pan, hover tooltips
- Responsive layout
- Export buttons (built into Plotly graphs)
- Automatic color theming
