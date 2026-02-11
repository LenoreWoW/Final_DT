"""
Dashboard Visualization for Quantum Trail.
Provides comprehensive dashboards for monitoring quantum systems, performance metrics, and results.
"""

import json
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import structlog

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.utils
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = structlog.get_logger(__name__)

class ChartType(Enum):
    """Types of charts available."""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    PIE = "pie"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    BOX = "box"
    VIOLIN = "violin"
    GAUGE = "gauge"
    RADAR = "radar"
    SANKEY = "sankey"
    TREEMAP = "treemap"

class DashboardTheme(Enum):
    """Dashboard themes."""
    DARK = "dark"
    LIGHT = "light"
    QUANTUM = "quantum"
    SCIENTIFIC = "scientific"

@dataclass
class ChartConfig:
    """Configuration for individual charts."""
    chart_id: str
    chart_type: ChartType
    title: str
    width: int = 6  # Grid width (1-12)
    height: int = 400
    data_source: str = ""
    refresh_interval: int = 30  # seconds
    filters: List[str] = field(default_factory=list)
    styling: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DashboardLayout:
    """Layout configuration for dashboard."""
    title: str
    description: str = ""
    theme: DashboardTheme = DashboardTheme.DARK
    auto_refresh: bool = True
    refresh_interval: int = 30
    charts: List[ChartConfig] = field(default_factory=list)
    layout_grid: List[List[str]] = field(default_factory=list)

class QuantumDashboardGenerator:
    """Generates interactive dashboards for quantum computing systems."""
    
    def __init__(self):
        self.dashboards: Dict[str, DashboardLayout] = {}
        self.chart_data_cache: Dict[str, Any] = {}
        self.themes = self._setup_themes()
        
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Dashboard functionality will be limited.")
    
    def _setup_themes(self) -> Dict[DashboardTheme, Dict[str, Any]]:
        """Setup dashboard themes."""
        return {
            DashboardTheme.DARK: {
                'background_color': '#0E1117',
                'paper_color': '#1E1E1E',
                'font_color': '#FFFFFF',
                'grid_color': '#2E2E2E',
                'primary_color': '#00D4FF',
                'secondary_color': '#FF6B6B',
                'accent_color': '#4ECDC4'
            },
            DashboardTheme.LIGHT: {
                'background_color': '#FFFFFF',
                'paper_color': '#F8F9FA',
                'font_color': '#212529',
                'grid_color': '#E9ECEF',
                'primary_color': '#007BFF',
                'secondary_color': '#DC3545',
                'accent_color': '#17A2B8'
            },
            DashboardTheme.QUANTUM: {
                'background_color': '#0A0A0F',
                'paper_color': '#1A1A2E',
                'font_color': '#E0E0FF',
                'grid_color': '#2A2A3E',
                'primary_color': '#00FFFF',
                'secondary_color': '#FF00FF',
                'accent_color': '#FFFF00'
            },
            DashboardTheme.SCIENTIFIC: {
                'background_color': '#FAFAFA',
                'paper_color': '#FFFFFF',
                'font_color': '#333333',
                'grid_color': '#CCCCCC',
                'primary_color': '#1976D2',
                'secondary_color': '#D32F2F',
                'accent_color': '#388E3C'
            }
        }
    
    def create_dashboard(self, dashboard_id: str, layout: DashboardLayout) -> bool:
        """Create a new dashboard."""
        try:
            self.dashboards[dashboard_id] = layout
            logger.info(f"Created dashboard: {dashboard_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create dashboard {dashboard_id}: {e}")
            return False
    
    def add_chart(self, dashboard_id: str, chart_config: ChartConfig) -> bool:
        """Add a chart to a dashboard."""
        if dashboard_id not in self.dashboards:
            logger.error(f"Dashboard {dashboard_id} not found")
            return False
        
        try:
            self.dashboards[dashboard_id].charts.append(chart_config)
            logger.info(f"Added chart {chart_config.chart_id} to dashboard {dashboard_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add chart: {e}")
            return False
    
    def generate_system_overview_dashboard(self) -> str:
        """Generate system overview dashboard."""
        dashboard_id = "system_overview"
        
        layout = DashboardLayout(
            title="Quantum Trail System Overview",
            description="Real-time monitoring of quantum computing system performance and health",
            theme=DashboardTheme.QUANTUM
        )
        
        # System Health Chart
        layout.charts.append(ChartConfig(
            chart_id="system_health",
            chart_type=ChartType.GAUGE,
            title="System Health Score",
            width=4,
            height=300
        ))
        
        # CPU and Memory Usage
        layout.charts.append(ChartConfig(
            chart_id="resource_usage",
            chart_type=ChartType.LINE,
            title="System Resources",
            width=8,
            height=300
        ))
        
        # Active Quantum Jobs
        layout.charts.append(ChartConfig(
            chart_id="quantum_jobs",
            chart_type=ChartType.BAR,
            title="Active Quantum Jobs by Backend",
            width=6,
            height=400
        ))
        
        # Performance Metrics
        layout.charts.append(ChartConfig(
            chart_id="performance_metrics",
            chart_type=ChartType.SCATTER,
            title="Performance vs Accuracy",
            width=6,
            height=400
        ))
        
        self.create_dashboard(dashboard_id, layout)
        return dashboard_id
    
    def generate_quantum_circuit_dashboard(self) -> str:
        """Generate quantum circuit analysis dashboard."""
        dashboard_id = "quantum_circuits"
        
        layout = DashboardLayout(
            title="Quantum Circuit Analysis",
            description="Analysis and visualization of quantum circuit executions",
            theme=DashboardTheme.SCIENTIFIC
        )
        
        # Circuit Execution Times
        layout.charts.append(ChartConfig(
            chart_id="circuit_times",
            chart_type=ChartType.HISTOGRAM,
            title="Circuit Execution Time Distribution",
            width=6,
            height=400
        ))
        
        # Success Rates by Backend
        layout.charts.append(ChartConfig(
            chart_id="success_rates",
            chart_type=ChartType.PIE,
            title="Success Rate by Backend",
            width=6,
            height=400
        ))
        
        # Quantum Advantage Analysis
        layout.charts.append(ChartConfig(
            chart_id="quantum_advantage",
            chart_type=ChartType.BOX,
            title="Quantum Advantage Distribution",
            width=12,
            height=400
        ))
        
        self.create_dashboard(dashboard_id, layout)
        return dashboard_id
    
    def generate_ml_pipeline_dashboard(self) -> str:
        """Generate machine learning pipeline dashboard."""
        dashboard_id = "ml_pipeline"
        
        layout = DashboardLayout(
            title="Quantum ML Pipeline",
            description="Monitoring quantum machine learning model training and performance",
            theme=DashboardTheme.DARK
        )
        
        # Model Training Progress
        layout.charts.append(ChartConfig(
            chart_id="training_progress",
            chart_type=ChartType.LINE,
            title="Training Loss Over Time",
            width=8,
            height=400
        ))
        
        # Model Accuracy
        layout.charts.append(ChartConfig(
            chart_id="model_accuracy",
            chart_type=ChartType.GAUGE,
            title="Current Model Accuracy",
            width=4,
            height=400
        ))
        
        # Feature Importance
        layout.charts.append(ChartConfig(
            chart_id="feature_importance",
            chart_type=ChartType.BAR,
            title="Quantum Feature Importance",
            width=6,
            height=400
        ))
        
        # Prediction Confidence
        layout.charts.append(ChartConfig(
            chart_id="prediction_confidence",
            chart_type=ChartType.VIOLIN,
            title="Prediction Confidence Distribution",
            width=6,
            height=400
        ))
        
        self.create_dashboard(dashboard_id, layout)
        return dashboard_id
    
    def generate_chart_data(self, chart_id: str, chart_type: ChartType, 
                          data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate chart data in the appropriate format."""
        if not PLOTLY_AVAILABLE:
            return self._generate_fallback_chart(chart_id, chart_type, data)
        
        try:
            if chart_type == ChartType.LINE:
                return self._generate_line_chart(data)
            elif chart_type == ChartType.BAR:
                return self._generate_bar_chart(data)
            elif chart_type == ChartType.SCATTER:
                return self._generate_scatter_chart(data)
            elif chart_type == ChartType.PIE:
                return self._generate_pie_chart(data)
            elif chart_type == ChartType.HISTOGRAM:
                return self._generate_histogram_chart(data)
            elif chart_type == ChartType.HEATMAP:
                return self._generate_heatmap_chart(data)
            elif chart_type == ChartType.GAUGE:
                return self._generate_gauge_chart(data)
            elif chart_type == ChartType.BOX:
                return self._generate_box_chart(data)
            elif chart_type == ChartType.VIOLIN:
                return self._generate_violin_chart(data)
            else:
                logger.warning(f"Chart type {chart_type} not implemented")
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate chart data for {chart_id}: {e}")
            return None
    
    def _generate_line_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate line chart data."""
        fig = go.Figure()
        
        x_data = data.get('x', list(range(len(data.get('y', [])))))
        y_data = data.get('y', [])
        
        if isinstance(y_data, dict):
            # Multiple lines
            for label, values in y_data.items():
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=values,
                    mode='lines+markers',
                    name=label
                ))
        else:
            # Single line
            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='lines+markers',
                name=data.get('label', 'Series 1')
            ))
        
        fig.update_layout(
            title=data.get('title', 'Line Chart'),
            xaxis_title=data.get('x_label', 'X Axis'),
            yaxis_title=data.get('y_label', 'Y Axis'),
            hovermode='x unified'
        )
        
        return json.loads(fig.to_json())
    
    def _generate_bar_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate bar chart data."""
        fig = go.Figure()
        
        x_data = data.get('x', [])
        y_data = data.get('y', [])
        
        fig.add_trace(go.Bar(
            x=x_data,
            y=y_data,
            name=data.get('label', 'Series 1'),
            marker_color=data.get('color', '#00D4FF')
        ))
        
        fig.update_layout(
            title=data.get('title', 'Bar Chart'),
            xaxis_title=data.get('x_label', 'Categories'),
            yaxis_title=data.get('y_label', 'Values')
        )
        
        return json.loads(fig.to_json())
    
    def _generate_scatter_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate scatter chart data."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data.get('x', []),
            y=data.get('y', []),
            mode='markers',
            marker=dict(
                size=data.get('size', 8),
                color=data.get('color', '#00D4FF'),
                opacity=0.8
            ),
            text=data.get('hover_text', []),
            name=data.get('label', 'Data Points')
        ))
        
        fig.update_layout(
            title=data.get('title', 'Scatter Plot'),
            xaxis_title=data.get('x_label', 'X Axis'),
            yaxis_title=data.get('y_label', 'Y Axis')
        )
        
        return json.loads(fig.to_json())
    
    def _generate_pie_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate pie chart data."""
        fig = go.Figure()
        
        fig.add_trace(go.Pie(
            labels=data.get('labels', []),
            values=data.get('values', []),
            hole=0.3 if data.get('donut', False) else 0
        ))
        
        fig.update_layout(
            title=data.get('title', 'Pie Chart')
        )
        
        return json.loads(fig.to_json())
    
    def _generate_histogram_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate histogram chart data."""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=data.get('x', []),
            nbinsx=data.get('bins', 20),
            name=data.get('label', 'Distribution')
        ))
        
        fig.update_layout(
            title=data.get('title', 'Histogram'),
            xaxis_title=data.get('x_label', 'Values'),
            yaxis_title='Frequency'
        )
        
        return json.loads(fig.to_json())
    
    def _generate_heatmap_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate heatmap chart data."""
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            z=data.get('z', []),
            x=data.get('x', []),
            y=data.get('y', []),
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title=data.get('title', 'Heatmap')
        )
        
        return json.loads(fig.to_json())
    
    def _generate_gauge_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate gauge chart data."""
        fig = go.Figure()
        
        value = data.get('value', 0)
        max_value = data.get('max_value', 100)
        
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': data.get('title', 'Gauge')},
            delta={'reference': data.get('reference', value * 0.9)},
            gauge={
                'axis': {'range': [None, max_value]},
                'bar': {'color': "#00D4FF"},
                'steps': [
                    {'range': [0, max_value * 0.5], 'color': "lightgray"},
                    {'range': [max_value * 0.5, max_value * 0.8], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_value * 0.9
                }
            }
        ))
        
        return json.loads(fig.to_json())
    
    def _generate_box_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate box plot chart data."""
        fig = go.Figure()
        
        if isinstance(data.get('y'), dict):
            # Multiple box plots
            for label, values in data['y'].items():
                fig.add_trace(go.Box(y=values, name=label))
        else:
            # Single box plot
            fig.add_trace(go.Box(y=data.get('y', []), name=data.get('label', 'Data')))
        
        fig.update_layout(
            title=data.get('title', 'Box Plot'),
            yaxis_title=data.get('y_label', 'Values')
        )
        
        return json.loads(fig.to_json())
    
    def _generate_violin_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate violin plot chart data."""
        fig = go.Figure()
        
        if isinstance(data.get('y'), dict):
            # Multiple violin plots
            for label, values in data['y'].items():
                fig.add_trace(go.Violin(y=values, name=label))
        else:
            # Single violin plot
            fig.add_trace(go.Violin(y=data.get('y', []), name=data.get('label', 'Data')))
        
        fig.update_layout(
            title=data.get('title', 'Violin Plot'),
            yaxis_title=data.get('y_label', 'Values')
        )
        
        return json.loads(fig.to_json())
    
    def _generate_fallback_chart(self, chart_id: str, chart_type: ChartType, 
                                data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback chart data when Plotly is not available."""
        return {
            'chart_id': chart_id,
            'chart_type': chart_type.value,
            'data': data,
            'message': 'Chart visualization requires Plotly installation',
            'fallback': True
        }
    
    def render_dashboard_html(self, dashboard_id: str) -> Optional[str]:
        """Render complete dashboard as HTML."""
        if dashboard_id not in self.dashboards:
            return None
        
        dashboard = self.dashboards[dashboard_id]
        theme = self.themes[dashboard.theme]
        
        # Generate sample data for demonstration
        sample_data = self._generate_sample_data(dashboard_id)
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{dashboard.title}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    background-color: {theme['background_color']};
                    color: {theme['font_color']};
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                }}
                .dashboard-header {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .dashboard-grid {{
                    display: grid;
                    grid-template-columns: repeat(12, 1fr);
                    gap: 20px;
                    max-width: 1400px;
                    margin: 0 auto;
                }}
                .chart-container {{
                    background-color: {theme['paper_color']};
                    border-radius: 8px;
                    padding: 20px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
                }}
                .chart-title {{
                    font-size: 18px;
                    font-weight: bold;
                    margin-bottom: 15px;
                    color: {theme['primary_color']};
                }}
            </style>
        </head>
        <body>
            <div class="dashboard-header">
                <h1>{dashboard.title}</h1>
                <p>{dashboard.description}</p>
            </div>
            <div class="dashboard-grid">
        """
        
        # Generate chart containers
        for chart in dashboard.charts:
            chart_data = sample_data.get(chart.chart_id, {})
            
            html_template += f"""
                <div class="chart-container" style="grid-column: span {chart.width};">
                    <div class="chart-title">{chart.title}</div>
                    <div id="{chart.chart_id}" style="height: {chart.height}px;"></div>
                </div>
            """
        
        html_template += """
            </div>
            <script>
        """
        
        # Generate chart JavaScript
        for chart in dashboard.charts:
            chart_data = sample_data.get(chart.chart_id, {})
            if chart_data and not chart_data.get('fallback', False):
                html_template += f"""
                    Plotly.newPlot('{chart.chart_id}', {json.dumps(chart_data)});
                """
        
        # Add auto-refresh if enabled
        if dashboard.auto_refresh:
            html_template += f"""
                setInterval(function() {{
                    // Auto-refresh functionality would go here
                    console.log('Dashboard refresh triggered');
                }}, {dashboard.refresh_interval * 1000});
            """
        
        html_template += """
            </script>
        </body>
        </html>
        """
        
        return html_template
    
    def _generate_sample_data(self, dashboard_id: str) -> Dict[str, Any]:
        """Generate sample data for dashboard demonstration."""
        import random
        
        sample_data = {}
        
        if dashboard_id == "system_overview":
            # System health gauge
            sample_data['system_health'] = {
                'value': random.uniform(75, 95),
                'max_value': 100,
                'title': 'System Health Score'
            }
            
            # Resource usage line chart
            time_points = list(range(24))  # 24 hours
            sample_data['resource_usage'] = {
                'x': time_points,
                'y': {
                    'CPU %': [random.uniform(20, 80) for _ in time_points],
                    'Memory %': [random.uniform(30, 70) for _ in time_points]
                },
                'title': 'System Resources Over Time',
                'x_label': 'Hours',
                'y_label': 'Usage %'
            }
            
            # Quantum jobs bar chart
            sample_data['quantum_jobs'] = {
                'x': ['IBM Simulator', 'IBM Quantum', 'Local Simulator', 'Google Cirq'],
                'y': [random.randint(5, 25) for _ in range(4)],
                'title': 'Active Quantum Jobs by Backend'
            }
            
            # Performance scatter plot
            sample_data['performance_metrics'] = {
                'x': [random.uniform(0.5, 1.0) for _ in range(50)],  # Accuracy
                'y': [random.uniform(1, 10) for _ in range(50)],     # Execution time
                'title': 'Performance vs Accuracy',
                'x_label': 'Accuracy',
                'y_label': 'Execution Time (s)'
            }
        
        elif dashboard_id == "quantum_circuits":
            # Circuit execution times histogram
            sample_data['circuit_times'] = {
                'x': [random.exponential(2) for _ in range(1000)],
                'title': 'Circuit Execution Time Distribution',
                'x_label': 'Execution Time (s)'
            }
            
            # Success rates pie chart
            sample_data['success_rates'] = {
                'labels': ['IBM Simulator', 'IBM Quantum', 'Local Simulator'],
                'values': [random.uniform(85, 99) for _ in range(3)],
                'title': 'Success Rate by Backend'
            }
            
            # Quantum advantage box plot
            sample_data['quantum_advantage'] = {
                'y': {
                    'QAOA': [random.uniform(0.8, 2.5) for _ in range(100)],
                    'VQE': [random.uniform(1.0, 3.0) for _ in range(100)],
                    'QNN': [random.uniform(1.2, 2.8) for _ in range(100)]
                },
                'title': 'Quantum Advantage by Algorithm'
            }
        
        elif dashboard_id == "ml_pipeline":
            # Training progress line chart
            epochs = list(range(1, 101))
            sample_data['training_progress'] = {
                'x': epochs,
                'y': {
                    'Training Loss': [1.0 * np.exp(-0.05 * x) + random.uniform(-0.1, 0.1) for x in epochs],
                    'Validation Loss': [1.0 * np.exp(-0.04 * x) + random.uniform(-0.1, 0.1) for x in epochs]
                },
                'title': 'Training Loss Over Time',
                'x_label': 'Epochs',
                'y_label': 'Loss'
            }
            
            # Model accuracy gauge
            sample_data['model_accuracy'] = {
                'value': random.uniform(85, 97),
                'max_value': 100,
                'title': 'Current Model Accuracy (%)'
            }
            
            # Feature importance bar chart
            features = [f'Feature_{i}' for i in range(1, 11)]
            sample_data['feature_importance'] = {
                'x': features,
                'y': [random.uniform(0.1, 1.0) for _ in features],
                'title': 'Quantum Feature Importance'
            }
            
            # Prediction confidence violin plot
            sample_data['prediction_confidence'] = {
                'y': {
                    'High Confidence': [random.uniform(0.8, 1.0) for _ in range(200)],
                    'Medium Confidence': [random.uniform(0.5, 0.8) for _ in range(150)],
                    'Low Confidence': [random.uniform(0.1, 0.5) for _ in range(100)]
                },
                'title': 'Prediction Confidence Distribution'
            }
        
        # Convert to chart format
        dashboard_charts = {}
        if dashboard_id in self.dashboards:
            for chart in self.dashboards[dashboard_id].charts:
                if chart.chart_id in sample_data:
                    chart_data = self.generate_chart_data(
                        chart.chart_id, 
                        chart.chart_type, 
                        sample_data[chart.chart_id]
                    )
                    if chart_data:
                        dashboard_charts[chart.chart_id] = chart_data
        
        return dashboard_charts
    
    def get_dashboard_config(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """Get dashboard configuration."""
        if dashboard_id not in self.dashboards:
            return None
        
        dashboard = self.dashboards[dashboard_id]
        return {
            'dashboard_id': dashboard_id,
            'title': dashboard.title,
            'description': dashboard.description,
            'theme': dashboard.theme.value,
            'auto_refresh': dashboard.auto_refresh,
            'refresh_interval': dashboard.refresh_interval,
            'charts': [
                {
                    'chart_id': chart.chart_id,
                    'chart_type': chart.chart_type.value,
                    'title': chart.title,
                    'width': chart.width,
                    'height': chart.height
                }
                for chart in dashboard.charts
            ]
        }
    
    def list_dashboards(self) -> List[Dict[str, Any]]:
        """List all available dashboards."""
        return [
            {
                'dashboard_id': dashboard_id,
                'title': dashboard.title,
                'description': dashboard.description,
                'theme': dashboard.theme.value,
                'chart_count': len(dashboard.charts)
            }
            for dashboard_id, dashboard in self.dashboards.items()
        ]
    
    def export_dashboard_config(self, dashboard_id: str) -> Optional[str]:
        """Export dashboard configuration as JSON."""
        config = self.get_dashboard_config(dashboard_id)
        if config:
            return json.dumps(config, indent=2)
        return None

# Global dashboard generator instance
dashboard_generator = QuantumDashboardGenerator()

# Convenience functions
def create_system_dashboard() -> str:
    """Create system overview dashboard."""
    return dashboard_generator.generate_system_overview_dashboard()

def create_quantum_dashboard() -> str:
    """Create quantum circuits dashboard."""
    return dashboard_generator.generate_quantum_circuit_dashboard()

def create_ml_dashboard() -> str:
    """Create ML pipeline dashboard."""
    return dashboard_generator.generate_ml_pipeline_dashboard()

def get_dashboard_html(dashboard_id: str) -> Optional[str]:
    """Get dashboard as HTML."""
    return dashboard_generator.render_dashboard_html(dashboard_id)

def list_available_dashboards() -> List[Dict[str, Any]]:
    """List all available dashboards."""
    return dashboard_generator.list_dashboards()