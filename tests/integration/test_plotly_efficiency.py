import pandas as pd
import numpy as np
import pytest
from h2_plant.visualization import plotly_graphs as pg

def test_plot_global_efficiency_timeline():
    # 1. Setup Mock Data
    steps = 100
    df = pd.DataFrame({
        'minute': np.arange(steps) * 60, # 100 hours
        'integrated_global_efficiency': np.random.uniform(0.7, 0.9, steps)
    })
    
    # 2. Call Function
    # Note: If plotly is not installed, this might return a mock or raise ImportError depending on implementation
    if not pg.PLOTLY_AVAILABLE:
        pytest.skip("Plotly not installed")
        
    fig = pg.plot_global_efficiency_timeline(df, title="Test Efficiency")
    
    # 3. Verify
    assert fig is not None
    
    # Check layout
    assert fig.layout.title.text == "Test Efficiency"
    assert fig.layout.yaxis.title.text == "Efficiency (% LHV)"
    
    # Check traces
    # Trace 0: Efficiency Line
    efficiency_trace = fig.data[0]
    assert efficiency_trace.name == 'Integrated Plant Efficiency'
    assert len(efficiency_trace.y) <= steps # Downsampling might reduce current count
    
    # Trace 1: Mean Line (because we have data > 0.1)
    # Plotly lines are typically traces or shapes. 
    # add_hline usually adds a shape, but sometimes it's implemented as a trace in older helpers.
    # In plotly 4.12+ add_hline adds to layout.shapes.
    
    assert len(fig.layout.shapes) >= 1
    mean_line = fig.layout.shapes[0]
    assert mean_line.type == 'line'
    assert mean_line.line.dash == 'dash'

    print("Plotly efficiency chart verification successful.")

if __name__ == "__main__":
    test_plot_global_efficiency_timeline()
