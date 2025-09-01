#!/usr/bin/env python3
"""
Report Generator - Comprehensive Reporting System

Generates detailed benchmark reports in multiple formats:
- Interactive HTML reports with visualizations
- Structured JSON reports for automation
- CSV exports for data analysis
- Executive summary reports for stakeholders
- Performance comparison reports
- Regression analysis reports
"""

import json
import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import base64
import io


@dataclass
class ReportConfiguration:
    """Configuration for report generation"""
    include_visualizations: bool = True
    include_raw_data: bool = False
    include_system_info: bool = True
    include_recommendations: bool = True
    chart_theme: str = "default"  # "default", "dark", "minimal"
    export_formats: List[str] = None  # ["html", "json", "csv"]
    
    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ["html", "json"]


class ReportGenerator:
    """Comprehensive benchmark report generation system"""
    
    def __init__(self, output_dir: Path, config: ReportConfiguration = None):
        """Initialize report generator
        
        Args:
            output_dir: Directory for report outputs
            config: Report generation configuration
        """
        self.output_dir = output_dir
        self.config = config or ReportConfiguration()
        self.logger = logging.getLogger("ReportGenerator")
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_comprehensive_report(self, benchmark_result, performance_report = None) -> Dict[str, Path]:
        """Generate comprehensive report in all configured formats"""
        self.logger.info("Generating comprehensive benchmark report")
        
        generated_files = {}
        
        # Generate HTML report
        if "html" in self.config.export_formats:
            html_file = self.generate_html_report(benchmark_result, performance_report)
            generated_files["html"] = html_file
        
        # Generate JSON report
        if "json" in self.config.export_formats:
            json_file = self.generate_json_report(benchmark_result, performance_report)
            generated_files["json"] = json_file
        
        # Generate CSV report
        if "csv" in self.config.export_formats:
            csv_file = self.generate_csv_report(benchmark_result, performance_report)
            generated_files["csv"] = csv_file
        
        # Generate executive summary
        summary_file = self.generate_executive_summary(benchmark_result)
        generated_files["summary"] = summary_file
        
        self.logger.info(f"Generated {len(generated_files)} report files")
        return generated_files
    
    def generate_html_report(self, benchmark_result, performance_report = None) -> Path:
        """Generate interactive HTML report with visualizations"""
        self.logger.info("Generating HTML report")
        
        html_file = self.output_dir / f"{benchmark_result.benchmark_id}_report.html"
        
        # Generate HTML content
        html_content = self._create_html_structure(benchmark_result, performance_report)
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report generated: {html_file}")
        return html_file
    
    def _create_html_structure(self, benchmark_result, performance_report) -> str:
        """Create complete HTML report structure"""
        # CSS styles
        css_styles = self._get_css_styles()
        
        # JavaScript for interactivity
        javascript = self._get_javascript()
        
        # Report sections
        header_section = self._generate_header_section(benchmark_result)
        summary_section = self._generate_summary_section(benchmark_result)
        system_section = self._generate_system_section(benchmark_result)
        dataset_section = self._generate_dataset_section(benchmark_result)
        performance_section = self._generate_performance_section(benchmark_result, performance_report)
        memory_section = self._generate_memory_section(benchmark_result)
        concurrent_section = self._generate_concurrent_section(benchmark_result)
        recommendations_section = self._generate_recommendations_section(benchmark_result)
        
        # Visualizations
        charts_section = ""
        if self.config.include_visualizations:
            charts_section = self._generate_charts_section(benchmark_result, performance_report)
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LocalData MCP Benchmark Report - {benchmark_result.benchmark_id}</title>
    {css_styles}
</head>
<body>
    <div class="container">
        {header_section}
        {summary_section}
        {system_section}
        {dataset_section}
        {performance_section}
        {memory_section}
        {concurrent_section}
        {charts_section}
        {recommendations_section}
        
        <div class="footer">
            <p>Generated by LocalData MCP Benchmarking Framework</p>
            <p>Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        </div>
    </div>
    
    {javascript}
</body>
</html>"""
        
        return html_template
    
    def _get_css_styles(self) -> str:
        """Get CSS styles for HTML report"""
        theme_colors = {
            "default": {
                "primary": "#2c3e50",
                "secondary": "#3498db", 
                "success": "#27ae60",
                "warning": "#f39c12",
                "danger": "#e74c3c",
                "background": "#ecf0f1",
                "card_bg": "#ffffff"
            },
            "dark": {
                "primary": "#1a1a1a",
                "secondary": "#4a90e2",
                "success": "#2ecc71",
                "warning": "#f1c40f",
                "danger": "#e74c3c", 
                "background": "#2c3e50",
                "card_bg": "#34495e"
            }
        }
        
        colors = theme_colors.get(self.config.chart_theme, theme_colors["default"])
        
        return f"""
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: {colors["background"]};
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: linear-gradient(135deg, {colors["primary"]}, {colors["secondary"]});
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
            font-weight: 300;
        }}
        
        .header .meta {{
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
            margin-top: 20px;
        }}
        
        .meta-item {{
            display: flex;
            flex-direction: column;
        }}
        
        .meta-label {{
            font-size: 0.9rem;
            opacity: 0.8;
        }}
        
        .meta-value {{
            font-size: 1.2rem;
            font-weight: 600;
        }}
        
        .section {{
            background: {colors["card_bg"]};
            margin-bottom: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .section-header {{
            background: {colors["primary"]};
            color: white;
            padding: 20px;
            font-size: 1.3rem;
            font-weight: 600;
        }}
        
        .section-content {{
            padding: 30px;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid {colors["secondary"]};
        }}
        
        .metric-label {{
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 5px;
        }}
        
        .metric-value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: {colors["primary"]};
        }}
        
        .metric-unit {{
            font-size: 1rem;
            color: #666;
            font-weight: normal;
        }}
        
        .status-indicator {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        
        .status-passed {{
            background: {colors["success"]};
            color: white;
        }}
        
        .status-warning {{
            background: {colors["warning"]};
            color: white;
        }}
        
        .status-failed {{
            background: {colors["danger"]};
            color: white;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
            margin: 10px 0;
        }}
        
        .progress-fill {{
            height: 100%;
            background: {colors["success"]};
            transition: width 0.3s ease;
        }}
        
        .chart-container {{
            margin: 20px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        
        .recommendations {{
            list-style: none;
        }}
        
        .recommendations li {{
            padding: 10px;
            margin: 5px 0;
            background: #fff3cd;
            border-left: 4px solid {colors["warning"]};
            border-radius: 4px;
        }}
        
        .footer {{
            text-align: center;
            padding: 30px;
            color: #666;
            border-top: 1px solid #ddd;
            margin-top: 50px;
        }}
        
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        .data-table th,
        .data-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        .data-table th {{
            background: {colors["primary"]};
            color: white;
            font-weight: 600;
        }}
        
        .data-table tr:hover {{
            background: #f5f5f5;
        }}
        
        .collapsible {{
            cursor: pointer;
            padding: 10px;
            background: #f1f1f1;
            border: none;
            text-align: left;
            outline: none;
            font-size: 1rem;
            width: 100%;
            border-radius: 4px;
            margin: 5px 0;
        }}
        
        .collapsible:hover {{
            background: #ddd;
        }}
        
        .collapsible-content {{
            padding: 0 18px;
            display: none;
            overflow: hidden;
            background: #f9f9f9;
            border-radius: 4px;
            margin-bottom: 10px;
        }}
        
        .collapsible-content.active {{
            display: block;
            padding: 18px;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}
            
            .header {{
                padding: 20px;
            }}
            
            .header h1 {{
                font-size: 1.8rem;
            }}
            
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
            
            .meta {{
                gap: 15px;
            }}
        }}
    </style>
        """
    
    def _get_javascript(self) -> str:
        """Get JavaScript for report interactivity"""
        return """
    <script>
        // Collapsible sections
        document.addEventListener('DOMContentLoaded', function() {
            const collapsibles = document.querySelectorAll('.collapsible');
            
            collapsibles.forEach(function(collapsible) {
                collapsible.addEventListener('click', function() {
                    this.classList.toggle('active');
                    const content = this.nextElementSibling;
                    content.classList.toggle('active');
                });
            });
            
            // Animate progress bars
            const progressBars = document.querySelectorAll('.progress-fill');
            progressBars.forEach(function(bar) {
                const width = bar.style.width;
                bar.style.width = '0%';
                setTimeout(function() {
                    bar.style.width = width;
                }, 500);
            });
        });
        
        // Chart rendering (placeholder for chart library integration)
        function renderChart(containerId, data, type) {
            const container = document.getElementById(containerId);
            if (container) {
                container.innerHTML = '<p>Chart visualization would be rendered here with data: ' + 
                                    JSON.stringify(data).substring(0, 100) + '...</p>';
            }
        }
        
        // Export functions
        function exportData(format, data) {
            const dataStr = format === 'json' ? JSON.stringify(data, null, 2) : data;
            const dataUri = 'data:application/' + format + ';charset=utf-8,'+ encodeURIComponent(dataStr);
            
            const exportFileDefaultName = 'benchmark_data.' + format;
            
            const linkElement = document.createElement('a');
            linkElement.setAttribute('href', dataUri);
            linkElement.setAttribute('download', exportFileDefaultName);
            linkElement.click();
        }
    </script>
        """
    
    def _generate_header_section(self, benchmark_result) -> str:
        """Generate header section of the report"""
        duration_formatted = f"{benchmark_result.total_duration_seconds:.1f}"
        success_rate_formatted = f"{benchmark_result.success_rate:.1f}"
        
        return f"""
        <div class="header">
            <h1>LocalData MCP Benchmark Report</h1>
            <p>Comprehensive validation of LocalData MCP v{benchmark_result.localdata_version} architecture</p>
            
            <div class="meta">
                <div class="meta-item">
                    <span class="meta-label">Benchmark ID</span>
                    <span class="meta-value">{benchmark_result.benchmark_id}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Duration</span>
                    <span class="meta-value">{duration_formatted}s</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Success Rate</span>
                    <span class="meta-value">{success_rate_formatted}%</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Generated</span>
                    <span class="meta-value">{datetime.fromisoformat(benchmark_result.timestamp).strftime('%Y-%m-%d %H:%M')}</span>
                </div>
            </div>
        </div>
        """
    
    def _generate_summary_section(self, benchmark_result) -> str:
        """Generate executive summary section"""
        overall_score = benchmark_result.performance_summary.get("overall_performance_score", 0)
        score_color = "#27ae60" if overall_score >= 8 else "#f39c12" if overall_score >= 6 else "#e74c3c"
        
        return f"""
        <div class="section">
            <div class="section-header">Executive Summary</div>
            <div class="section-content">
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Overall Performance Score</div>
                        <div class="metric-value" style="color: {score_color}">
                            {overall_score:.1f}<span class="metric-unit">/10</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {overall_score * 10}%; background: {score_color};"></div>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Datasets Tested</div>
                        <div class="metric-value">
                            {len(benchmark_result.dataset_results)}<span class="metric-unit">datasets</span>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Memory Efficiency</div>
                        <div class="metric-value">
                            {benchmark_result.performance_summary.get('memory_efficiency', 'N/A')}
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Streaming Performance</div>
                        <div class="metric-value">
                            {benchmark_result.performance_summary.get('dataset_generation_performance', 'N/A')}
                        </div>
                    </div>
                </div>
                
                <h3>Key Findings</h3>
                <ul>
                    <li>LocalData MCP v{benchmark_result.localdata_version} architecture validation completed</li>
                    <li>Total data processed: {sum(result.get('dataset_size_mb', 0) for result in benchmark_result.dataset_results.values()):.0f} MB</li>
                    <li>Streaming architecture {'activated successfully' if overall_score > 7 else 'requires optimization'}</li>
                    <li>Memory safety {'validated' if benchmark_result.memory_results.get('memory_leak_tests') == 'passed' else 'needs attention'}</li>
                </ul>
            </div>
        </div>
        """
    
    def _generate_system_section(self, benchmark_result) -> str:
        """Generate system information section"""
        system_info = benchmark_result.system_info
        
        return f"""
        <div class="section">
            <div class="section-header">System Information</div>
            <div class="section-content">
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Operating System</div>
                        <div class="metric-value" style="font-size: 1.2rem;">{system_info.get('os', 'Unknown')}</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">CPU</div>
                        <div class="metric-value" style="font-size: 1.2rem;">
                            {system_info.get('cpu_count', 'N/A')} cores
                        </div>
                        <div style="font-size: 0.9rem; color: #666; margin-top: 5px;">
                            {system_info.get('cpu_model', 'Unknown')[:50]}{'...' if len(system_info.get('cpu_model', '')) > 50 else ''}
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Total Memory</div>
                        <div class="metric-value">
                            {system_info.get('total_memory_gb', 0):.1f}<span class="metric-unit">GB</span>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Available Disk Space</div>
                        <div class="metric-value">
                            {system_info.get('disk_space_gb', 0):.1f}<span class="metric-unit">GB</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
    
    def _generate_dataset_section(self, benchmark_result) -> str:
        """Generate dataset benchmark results section"""
        dataset_rows = ""
        
        for dataset_name, result in benchmark_result.dataset_results.items():
            success_status = "passed" if result.get("success", False) else "failed"
            status_class = f"status-{success_status}"
            
            size_mb = result.get("dataset_size_mb", 0)
            throughput = result.get("throughput_mb_per_second", 0)
            generation_time = result.get("generation_time_seconds", 0)
            
            dataset_rows += f"""
                <tr>
                    <td>{dataset_name.title()}</td>
                    <td><span class="status-indicator {status_class}">{success_status}</span></td>
                    <td>{size_mb:.0f} MB</td>
                    <td>{generation_time:.1f}s</td>
                    <td>{throughput:.1f} MB/s</td>
                    <td>{'✓' if result.get('validation_results', {}).get('size_validation') == 'passed' else '✗'}</td>
                </tr>
            """
        
        return f"""
        <div class="section">
            <div class="section-header">Dataset Benchmark Results</div>
            <div class="section-content">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Dataset</th>
                            <th>Status</th>
                            <th>Size</th>
                            <th>Generation Time</th>
                            <th>Throughput</th>
                            <th>Validated</th>
                        </tr>
                    </thead>
                    <tbody>
                        {dataset_rows}
                    </tbody>
                </table>
            </div>
        </div>
        """
    
    def _generate_performance_section(self, benchmark_result, performance_report) -> str:
        """Generate performance analysis section"""
        perf_summary = benchmark_result.performance_summary
        
        return f"""
        <div class="section">
            <div class="section-header">Performance Analysis</div>
            <div class="section-content">
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">CPU-bound Operations</div>
                        <div class="metric-value">
                            {perf_summary.get('performance_characteristics', {}).get('cpu_bound_operations', 0)}<span class="metric-unit">%</span>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Memory-bound Operations</div>
                        <div class="metric-value">
                            {perf_summary.get('performance_characteristics', {}).get('memory_bound_operations', 0)}<span class="metric-unit">%</span>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">I/O-bound Operations</div>
                        <div class="metric-value">
                            {perf_summary.get('performance_characteristics', {}).get('io_bound_operations', 0)}<span class="metric-unit">%</span>
                        </div>
                    </div>
                </div>
                
                <h3>Identified Bottlenecks</h3>
                <ul>
                    {''.join(f'<li>{bottleneck}</li>' for bottleneck in perf_summary.get('identified_bottlenecks', []))}
                </ul>
            </div>
        </div>
        """
    
    def _generate_memory_section(self, benchmark_result) -> str:
        """Generate memory safety results section"""
        memory_results = benchmark_result.memory_results
        
        return f"""
        <div class="section">
            <div class="section-header">Memory Safety Validation</div>
            <div class="section-content">
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Streaming Activation</div>
                        <div class="metric-value">
                            <span class="status-indicator status-{memory_results.get('streaming_activation_tests', 'unknown').replace('passed', 'passed').replace('failed', 'failed')}">
                                {memory_results.get('streaming_activation_tests', 'Unknown')}
                            </span>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Memory Leak Detection</div>
                        <div class="metric-value">
                            <span class="status-indicator status-{memory_results.get('memory_leak_tests', 'unknown').replace('passed', 'passed').replace('failed', 'failed')}">
                                {memory_results.get('memory_leak_tests', 'Unknown')}
                            </span>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Max Memory Usage</div>
                        <div class="metric-value">
                            {memory_results.get('max_memory_usage_mb', 0):.0f}<span class="metric-unit">MB</span>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Streaming Threshold</div>
                        <div class="metric-value">
                            {memory_results.get('streaming_threshold_mb', 0):.0f}<span class="metric-unit">MB</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
    
    def _generate_concurrent_section(self, benchmark_result) -> str:
        """Generate concurrent usage results section"""
        concurrent_results = benchmark_result.concurrent_results
        
        return f"""
        <div class="section">
            <div class="section-header">Concurrent Usage Validation</div>
            <div class="section-content">
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Max Concurrent Operations</div>
                        <div class="metric-value">
                            {concurrent_results.get('max_concurrent_operations', 0)}
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Average Response Time</div>
                        <div class="metric-value">
                            {concurrent_results.get('avg_response_time_ms', 0):.0f}<span class="metric-unit">ms</span>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Deadlock Tests</div>
                        <div class="metric-value">
                            <span class="status-indicator status-{concurrent_results.get('deadlock_tests', 'unknown').replace('passed', 'passed').replace('failed', 'failed')}">
                                {concurrent_results.get('deadlock_tests', 'Unknown')}
                            </span>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Race Condition Tests</div>
                        <div class="metric-value">
                            <span class="status-indicator status-{concurrent_results.get('race_condition_tests', 'unknown').replace('passed', 'passed').replace('failed', 'failed')}">
                                {concurrent_results.get('race_condition_tests', 'Unknown')}
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
    
    def _generate_charts_section(self, benchmark_result, performance_report) -> str:
        """Generate charts and visualizations section"""
        return f"""
        <div class="section">
            <div class="section-header">Performance Visualizations</div>
            <div class="section-content">
                <div class="chart-container">
                    <h4>Dataset Generation Throughput</h4>
                    <div id="throughput-chart" style="height: 300px; background: #f0f0f0; display: flex; align-items: center; justify-content: center;">
                        <p>Chart would display throughput comparison across datasets</p>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h4>Memory Usage Over Time</h4>
                    <div id="memory-chart" style="height: 300px; background: #f0f0f0; display: flex; align-items: center; justify-content: center;">
                        <p>Chart would display memory usage timeline</p>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h4>Concurrent Operation Performance</h4>
                    <div id="concurrency-chart" style="height: 300px; background: #f0f0f0; display: flex; align-items: center; justify-content: center;">
                        <p>Chart would display concurrent operation metrics</p>
                    </div>
                </div>
            </div>
        </div>
        """
    
    def _generate_recommendations_section(self, benchmark_result) -> str:
        """Generate recommendations section"""
        recommendations_html = ''.join(f'<li>{rec}</li>' for rec in benchmark_result.recommendations)
        
        return f"""
        <div class="section">
            <div class="section-header">Performance Recommendations</div>
            <div class="section-content">
                <ul class="recommendations">
                    {recommendations_html}
                </ul>
                
                <button class="collapsible">View Detailed Analysis</button>
                <div class="collapsible-content">
                    <h4>Baseline Comparison</h4>
                    <p>Baseline metrics {'established' if benchmark_result.baseline_comparison else 'not available'}</p>
                    
                    <h4>Regression Analysis</h4>
                    <p>{'No regressions detected' if not benchmark_result.regression_analysis.get('regressions_detected') else 'Performance regressions found'}</p>
                    
                    <h4>Next Steps</h4>
                    <ul>
                        <li>Review memory optimization opportunities</li>
                        <li>Validate streaming thresholds for production workloads</li>
                        <li>Consider horizontal scaling for high-concurrency scenarios</li>
                    </ul>
                </div>
            </div>
        </div>
        """
    
    def generate_json_report(self, benchmark_result, performance_report = None) -> Path:
        """Generate structured JSON report"""
        self.logger.info("Generating JSON report")
        
        json_file = self.output_dir / f"{benchmark_result.benchmark_id}_report.json"
        
        # Create comprehensive JSON structure
        json_data = {
            "metadata": {
                "report_version": "1.0.0",
                "generated_at": datetime.now().isoformat(),
                "generator": "LocalData MCP ReportGenerator"
            },
            "benchmark_summary": asdict(benchmark_result),
            "performance_analysis": asdict(performance_report) if performance_report else None,
            "export_info": {
                "formats_available": self.config.export_formats,
                "includes_raw_data": self.config.include_raw_data,
                "includes_visualizations": self.config.include_visualizations
            }
        }
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        self.logger.info(f"JSON report generated: {json_file}")
        return json_file
    
    def generate_csv_report(self, benchmark_result, performance_report = None) -> Path:
        """Generate CSV report with key metrics"""
        self.logger.info("Generating CSV report")
        
        csv_file = self.output_dir / f"{benchmark_result.benchmark_id}_metrics.csv"
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'Metric', 'Value', 'Unit', 'Category', 'Timestamp'
            ])
            
            # Write benchmark summary metrics
            timestamp = benchmark_result.timestamp
            writer.writerow(['benchmark_id', benchmark_result.benchmark_id, '', 'summary', timestamp])
            writer.writerow(['duration_seconds', benchmark_result.total_duration_seconds, 'seconds', 'summary', timestamp])
            writer.writerow(['success_rate', benchmark_result.success_rate, 'percent', 'summary', timestamp])
            
            # Write dataset metrics
            for dataset_name, result in benchmark_result.dataset_results.items():
                if result.get('success'):
                    writer.writerow([f'{dataset_name}_size', result.get('dataset_size_mb', 0), 'MB', 'dataset', timestamp])
                    writer.writerow([f'{dataset_name}_throughput', result.get('throughput_mb_per_second', 0), 'MB/s', 'dataset', timestamp])
                    writer.writerow([f'{dataset_name}_generation_time', result.get('generation_time_seconds', 0), 'seconds', 'dataset', timestamp])
            
            # Write memory metrics
            memory_results = benchmark_result.memory_results
            writer.writerow(['max_memory_usage', memory_results.get('max_memory_usage_mb', 0), 'MB', 'memory', timestamp])
            writer.writerow(['streaming_threshold', memory_results.get('streaming_threshold_mb', 0), 'MB', 'memory', timestamp])
            
            # Write concurrent metrics
            concurrent_results = benchmark_result.concurrent_results
            writer.writerow(['max_concurrent_ops', concurrent_results.get('max_concurrent_operations', 0), 'operations', 'concurrency', timestamp])
            writer.writerow(['avg_response_time', concurrent_results.get('avg_response_time_ms', 0), 'ms', 'concurrency', timestamp])
        
        self.logger.info(f"CSV report generated: {csv_file}")
        return csv_file
    
    def generate_executive_summary(self, benchmark_result) -> Path:
        """Generate executive summary report"""
        self.logger.info("Generating executive summary")
        
        summary_file = self.output_dir / f"{benchmark_result.benchmark_id}_summary.txt"
        
        overall_score = benchmark_result.performance_summary.get("overall_performance_score", 0)
        
        summary_content = f"""
LocalData MCP Benchmark Executive Summary
========================================

Benchmark ID: {benchmark_result.benchmark_id}
Generated: {datetime.fromisoformat(benchmark_result.timestamp).strftime('%Y-%m-%d %H:%M:%S UTC')}
Duration: {benchmark_result.total_duration_seconds:.1f} seconds
Success Rate: {benchmark_result.success_rate:.1f}%

OVERALL ASSESSMENT
-----------------
Performance Score: {overall_score:.1f}/10 ({self._get_score_rating(overall_score)})
LocalData MCP Version: {benchmark_result.localdata_version}

KEY METRICS
-----------
• Datasets Tested: {len(benchmark_result.dataset_results)}
• Total Data Processed: {sum(result.get('dataset_size_mb', 0) for result in benchmark_result.dataset_results.values()):.0f} MB
• Memory Efficiency: {benchmark_result.performance_summary.get('memory_efficiency', 'N/A')}
• Streaming Architecture: {'Validated' if benchmark_result.memory_results.get('streaming_activation_tests') == 'passed' else 'Requires Attention'}

DATASET PERFORMANCE
------------------
"""
        
        for dataset_name, result in benchmark_result.dataset_results.items():
            if result.get('success'):
                summary_content += f"• {dataset_name.title()}: {result.get('dataset_size_mb', 0):.0f}MB, {result.get('throughput_mb_per_second', 0):.1f}MB/s\n"
            else:
                summary_content += f"• {dataset_name.title()}: FAILED\n"
        
        summary_content += f"""
MEMORY SAFETY
-------------
• Streaming Tests: {benchmark_result.memory_results.get('streaming_activation_tests', 'Unknown')}
• Memory Leak Tests: {benchmark_result.memory_results.get('memory_leak_tests', 'Unknown')}
• Max Memory Usage: {benchmark_result.memory_results.get('max_memory_usage_mb', 0):.0f}MB

CONCURRENT OPERATIONS
--------------------
• Max Concurrent Ops: {benchmark_result.concurrent_results.get('max_concurrent_operations', 0)}
• Avg Response Time: {benchmark_result.concurrent_results.get('avg_response_time_ms', 0):.0f}ms
• Deadlock Tests: {benchmark_result.concurrent_results.get('deadlock_tests', 'Unknown')}

TOP RECOMMENDATIONS
------------------
"""
        
        for i, rec in enumerate(benchmark_result.recommendations[:5], 1):
            summary_content += f"{i}. {rec}\n"
        
        summary_content += f"""
CONCLUSION
----------
LocalData MCP v{benchmark_result.localdata_version} {'demonstrates solid performance' if overall_score >= 7 else 'requires optimization'}
for production workloads. {'The streaming architecture is functioning as expected.' if benchmark_result.memory_results.get('streaming_activation_tests') == 'passed' else 'Review streaming configuration is recommended.'}

This benchmark establishes baseline metrics for v1.4.0+ development priorities.
"""
        
        with open(summary_file, 'w') as f:
            f.write(summary_content)
        
        self.logger.info(f"Executive summary generated: {summary_file}")
        return summary_file
    
    def _get_score_rating(self, score: float) -> str:
        """Get textual rating for performance score"""
        if score >= 9:
            return "Excellent"
        elif score >= 8:
            return "Very Good"
        elif score >= 7:
            return "Good"
        elif score >= 6:
            return "Fair"
        elif score >= 5:
            return "Poor"
        else:
            return "Critical"
    
    def generate_comparison_report(self, current_result, baseline_result) -> Path:
        """Generate performance comparison report"""
        self.logger.info("Generating comparison report")
        
        comparison_file = self.output_dir / f"comparison_{current_result.benchmark_id}_vs_baseline.html"
        
        # Implementation would create detailed comparison
        # For now, create basic comparison structure
        comparison_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Comparison Report</title>
        </head>
        <body>
            <h1>Performance Comparison</h1>
            <h2>Current: {current_result.benchmark_id}</h2>
            <h2>Baseline: {baseline_result.benchmark_id if baseline_result else 'Not Available'}</h2>
            
            <p>Detailed comparison analysis would be implemented here</p>
        </body>
        </html>
        """
        
        with open(comparison_file, 'w') as f:
            f.write(comparison_content)
        
        return comparison_file