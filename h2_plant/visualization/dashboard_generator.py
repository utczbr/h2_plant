"""
Dashboard generator for hydrogen production plant simulation.
Generates a standalone HTML dashboard with interactive charts.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super(NpEncoder, self).default(obj)

class DashboardGenerator:
    """
    Generates an HTML dashboard from simulation results.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        
    def generate(self, results: Dict[str, Any]) -> Path:
        """
        Generate the HTML dashboard.
        
        Args:
            results: Dictionary containing simulation results, metrics, and timeseries.
            
        Returns:
            Path to the generated HTML file.
        """
        logger.info("Generating HTML dashboard...")
        
        # Prepare data for JavaScript
        dashboard_data = self._prepare_data(results)
        
        # Generate HTML content
        html_content = self._get_html_template(dashboard_data)
        
        # Write to file
        output_path = self.output_dir / "dashboard.html"
        output_path.write_text(html_content, encoding='utf-8')
        
        logger.info(f"Dashboard generated at: {output_path}")
        return output_path

    def _prepare_data(self, results: Dict[str, Any]) -> str:
        """Convert results to JSON string for embedding."""
        # We can filter or transform data here if needed to reduce size
        return json.dumps(results, cls=NpEncoder)

    def _get_html_template(self, data_json: str) -> str:
        """Return the complete HTML template with embedded data."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>H2 Plant Simulation Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: #fff; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }}
        .card {{ background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .kpi-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .kpi-label {{ color: #7f8c8d; font-size: 14px; }}
        h1 {{ margin: 0; color: #2c3e50; }}
        h2 {{ color: #34495e; margin-top: 0; }}
        canvas {{ width: 100% !important; height: 300px !important; }}
        #sankeyChart {{ width: 100%; height: 500px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Hydrogen Plant Simulation Results</h1>
            <p id="sim-info">Simulation completed successfully.</p>
        </div>

        <div class="grid">
            <div class="card">
                <div class="kpi-label">Total H2 Production</div>
                <div class="kpi-value" id="kpi-production">-</div>
            </div>
            <div class="card">
                <div class="kpi-label">Total Demand</div>
                <div class="kpi-value" id="kpi-demand">-</div>
            </div>
            <div class="card">
                <div class="kpi-label">Total Cost</div>
                <div class="kpi-value" id="kpi-cost">-</div>
            </div>
            <div class="card">
                <div class="kpi-label">Avg Cost / kg H2</div>
                <div class="kpi-value" id="kpi-avg-cost">-</div>
            </div>
        </div>

        <div class="card" style="margin-bottom: 20px;">
            <h2>System Flow (Sankey)</h2>
            <div id="sankeyChart"></div>
        </div>

        <div class="card" style="margin-bottom: 20px;">
            <h2>Production vs Demand</h2>
            <canvas id="productionChart"></canvas>
        </div>

        <div class="grid">
            <div class="card">
                <h2>Storage Levels</h2>
                <canvas id="storageChart"></canvas>
            </div>
            <div class="card">
                <h2>Energy Price</h2>
                <canvas id="priceChart"></canvas>
            </div>
        </div>

        <div class="card" style="margin-bottom: 20px;">
            <h2>Flow Matrix</h2>
            <div id="flowMatrixContainer" style="overflow-x: auto;"></div>
        </div>

        <div class="header">
            <h1>Advanced Analytics (Generated Graphs)</h1>
        </div>

        <div class="grid" style="grid-template-columns: 1fr;">
            <!-- Generated Plotly Graphs -->
            <div class="card">
                <h3>Total Production Stacked</h3>
                <iframe src="graphs/total_h2_production_stacked.html" style="border:none; width:100%; height:600px;"></iframe>
            </div>
            <div class="card">
                <h3>Cumulative Production</h3>
                <iframe src="graphs/cumulative_h2_production.html" style="border:none; width:100%; height:600px;"></iframe>
            </div>
            <div class="card">
                <h3>PEM Performance Surface</h3>
                <iframe src="graphs/pem_performance_surface.html" style="border:none; width:100%; height:800px;"></iframe>
            </div>
             <div class="card">
                <h3>Grid Interaction Phase Portrait</h3>
                <iframe src="graphs/grid_interaction_phase_portrait.html" style="border:none; width:100%; height:600px;"></iframe>
            </div>
            <div class="card">
                <h3>Dispatch Strategy</h3>
                <iframe src="graphs/dispatch_strategy_stacked.html" style="border:none; width:100%; height:600px;"></iframe>
            </div>
            <div class="card">
                <h3>Energy Price</h3>
                <iframe src="graphs/energy_price_over_time.html" style="border:none; width:100%; height:600px;"></iframe>
            </div>
             <div class="card">
                <h3>Wind Utilization</h3>
                <iframe src="graphs/wind_utilization_duration_curve.html" style="border:none; width:100%; height:600px;"></iframe>
            </div>
             <div class="card">
                <h3>Power Consumption Breakdown</h3>
                <iframe src="graphs/power_consumption_breakdown_pie.html" style="border:none; width:100%; height:600px;"></iframe>
            </div>
            
            <div class="grid">
                <div class="card">
                    <h3>PEM Production</h3>
                    <iframe src="graphs/pem_h2_production_over_time.html" style="border:none; width:100%; height:500px;"></iframe>
                </div>
                <div class="card">
                    <h3>SOEC Production</h3>
                    <iframe src="graphs/soec_h2_production_over_time.html" style="border:none; width:100%; height:500px;"></iframe>
                </div>
            </div>

            <div class="grid">
                <div class="card">
                    <h3>PEM Voltage</h3>
                    <iframe src="graphs/pem_cell_voltage_over_time.html" style="border:none; width:100%; height:500px;"></iframe>
                </div>
                <div class="card">
                    <h3>PEM Efficiency</h3>
                    <iframe src="graphs/pem_efficiency_over_time.html" style="border:none; width:100%; height:500px;"></iframe>
                </div>
            </div>

             <div class="grid">
                <div class="card">
                    <h3>SOEC Active Modules</h3>
                    <iframe src="graphs/soec_active_modules_over_time.html" style="border:none; width:100%; height:500px;"></iframe>
                </div>
                <div class="card">
                    <h3>Ramp Rate Stress</h3>
                    <iframe src="graphs/ramp_rate_stress_distribution.html" style="border:none; width:100%; height:500px;"></iframe>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Embedded Simulation Data
        const data = {data_json};

        // Initialize Dashboard
        function initDashboard() {{
            updateKPIs();
            renderCharts();
            renderSankey();
            renderFlowMatrix();
        }}

        function updateKPIs() {{
            const metrics = data.metrics;
            document.getElementById('kpi-production').textContent = formatNumber(metrics.total_production_kg, ' kg');
            document.getElementById('kpi-demand').textContent = formatNumber(metrics.total_demand_kg, ' kg');
            document.getElementById('kpi-cost').textContent = formatCurrency(metrics.total_cost);
            document.getElementById('kpi-avg-cost').textContent = formatCurrency(metrics.average_cost_per_kg);
        }}

        function renderSankey() {{
            if (!data.dashboard_data.flows || !data.dashboard_data.flows.sankey) return;
            
            const sankeyData = data.dashboard_data.flows.sankey;
            
            const plotData = [{{
                type: "sankey",
                orientation: "h",
                node: {{
                    pad: 15,
                    thickness: 30,
                    line: {{ color: "black", width: 0.5 }},
                    label: sankeyData.nodes.map(n => n.name),
                    color: sankeyData.nodes.map(n => n.color || "#3498db")
                }},
                link: {{
                    source: sankeyData.links.map(l => l.source),
                    target: sankeyData.links.map(l => l.target),
                    value: sankeyData.links.map(l => l.value),
                    label: sankeyData.links.map(l => l.label)
                }}
            }}];

            const layout = {{
                font: {{ size: 10 }},
                margin: {{ t: 20, l: 20, r: 20, b: 20 }}
            }};

            Plotly.newPlot('sankeyChart', plotData, layout);
        }}

        function renderFlowMatrix() {{
            if (!data.dashboard_data.flows || !data.dashboard_data.flows.matrix) return;
            
            const matrix = data.dashboard_data.flows.matrix;
            const container = document.getElementById('flowMatrixContainer');
            
            let html = '<table><thead><tr><th>Source \\ Dest</th>';
            
            // Header row
            matrix.columns.forEach(col => {{
                html += `<th>${{col}}</th>`;
            }});
            html += '</tr></thead><tbody>';
            
            // Data rows
            matrix.rows.forEach((rowName, i) => {{
                html += `<tr><th>${{rowName}}</th>`;
                matrix.data[i].forEach(val => {{
                    const bg = val > 0 ? `rgba(46, 204, 113, ${{Math.min(val/1000, 0.5)}})` : '';
                    html += `<td style="background:${{bg}}">${{val > 0 ? formatNumber(val) : '-'}}</td>`;
                }});
                html += '</tr>';
            }});
            
            html += '</tbody></table>';
            container.innerHTML = html;
        }}

        function renderCharts() {{
            const ts = data.dashboard_data.timeseries;
            const hours = ts.hour;

            // Production vs Demand
            new Chart(document.getElementById('productionChart'), {{
                type: 'line',
                data: {{
                    labels: hours,
                    datasets: [
                        {{
                            label: 'Total Production (kg)',
                            data: ts.production.total_kg,
                            borderColor: '#2ecc71',
                            tension: 0.1
                        }},
                        {{
                            label: 'Demand (kg)',
                            data: ts.demand.requested_kg,
                            borderColor: '#e74c3c',
                            borderDash: [5, 5],
                            tension: 0.1
                        }}
                    ]
                }},
                options: {{ responsive: true, maintainAspectRatio: false }}
            }});

            // Storage Levels
            new Chart(document.getElementById('storageChart'), {{
                type: 'line',
                data: {{
                    labels: hours,
                    datasets: [
                        {{
                            label: 'LP Storage (kg)',
                            data: ts.storage.lp_level_kg,
                            borderColor: '#3498db',
                            fill: true,
                            backgroundColor: 'rgba(52, 152, 219, 0.1)'
                        }},
                        {{
                            label: 'HP Storage (kg)',
                            data: ts.storage.hp_level_kg,
                            borderColor: '#9b59b6',
                            fill: true,
                            backgroundColor: 'rgba(155, 89, 182, 0.1)'
                        }}
                    ]
                }},
                options: {{ responsive: true, maintainAspectRatio: false }}
            }});
            
            // Energy Price
            new Chart(document.getElementById('priceChart'), {{
                type: 'line',
                data: {{
                    labels: hours,
                    datasets: [{{
                        label: 'Energy Price ($/MWh)',
                        data: ts.price.per_mwh,
                        borderColor: '#f1c40f',
                        backgroundColor: 'rgba(241, 196, 15, 0.1)',
                        fill: true,
                        tension: 0.1
                    }}]
                }},
                options: {{ responsive: true, maintainAspectRatio: false }}
            }});
        }}

        function formatNumber(num, suffix = '') {{
            return new Intl.NumberFormat('en-US', {{ maximumFractionDigits: 0 }}).format(num) + suffix;
        }}

        function formatCurrency(num) {{
            return new Intl.NumberFormat('en-US', {{ style: 'currency', currency: 'USD' }}).format(num);
        }}

        initDashboard();
    </script>
</body>
</html>
"""
