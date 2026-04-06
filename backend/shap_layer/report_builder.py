"""
Report Builder Module for QuantSignal Arena SHAP Layer.

Generates professional PDF tearsheet reports.
"""

from typing import Dict, Any
import os
from datetime import date
import tempfile
import logging

from fpdf import FPDF
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportBuilder:
    """
    Generates professional PDF tearsheet reports.
    
    Combines backtest results, SHAP explanations, and drift analysis
    into publication-quality documentation.
    """
    
    def __init__(self, output_dir: str = "reports/") -> None:
        """
        Initialize ReportBuilder.
        
        Args:
            output_dir: Directory for PDF output (default "reports/")
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"ReportBuilder initialized with output_dir={output_dir}")
    
    def _save_chart_to_temp(self, fig) -> str:
        """Save matplotlib figure to temporary PNG file."""
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        fig.savefig(tmp.name, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return tmp.name
    
    def build(
        self,
        hypothesis: str,
        backtest_results: Dict[str, Any],
        metrics: Dict[str, Any],
        shap_results: Dict[str, Any],
        drift_results: Dict[str, Any],
        generated_code: str = ""
    ) -> str:
        """
        Generate complete PDF tearsheet.
        
        Args:
            hypothesis: Plain English hypothesis text
            backtest_results: Output from BacktestEngine.run_backtest()
            metrics: Output from MetricsCalculator.calculate_metrics()
            shap_results: Output from SignalExplainer.explain()
            drift_results: Output from DriftDetector.detect()
            generated_code: Python signal code (optional)
            
        Returns:
            Full path to generated PDF file
        """
        logger.info("Building PDF tearsheet report")
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Section 1: Header
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "QuantSignal Arena - Strategy Report", ln=True)
        pdf.set_font("Helvetica", "", 11)
        
        # Truncate hypothesis if too long
        hypothesis_text = hypothesis[:100] if len(hypothesis) > 100 else hypothesis
        pdf.cell(0, 8, f"Hypothesis: {hypothesis_text}", ln=True)
        pdf.cell(0, 8, f"Generated: {date.today().strftime('%Y-%m-%d')}", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)
        
        # Section 2: Performance summary table
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Performance Summary", ln=True)
        pdf.set_font("Helvetica", "", 10)
        
        # Get Sharpe ratio for color coding
        sharpe = metrics.get('sharpe_ratio', 0)
        
        # Determine color
        if sharpe > 1.0:
            color = (0, 128, 0)  # Green
        elif sharpe < 0:
            color = (255, 0, 0)  # Red
        else:
            color = (0, 0, 0)  # Black
        
        # Metrics to display
        metric_names = [
            ('sharpe_ratio', 'Sharpe Ratio'),
            ('sortino_ratio', 'Sortino Ratio'),
            ('max_drawdown', 'Max Drawdown'),
            ('cagr', 'CAGR'),
            ('win_rate', 'Win Rate'),
            ('total_return', 'Total Return'),
            ('volatility', 'Volatility')
        ]
        
        for key, display_name in metric_names:
            value = metrics.get(key, 0)
            if value is None:
                value_str = "N/A"
            else:
                value_str = f"{value:.2f}"
            
            # Set color for Sharpe row
            if key == 'sharpe_ratio':
                pdf.set_text_color(*color)
            
            pdf.cell(100, 6, display_name, border=0)
            pdf.cell(0, 6, value_str, border=0, ln=True, align='R')
            
            # Reset color
            pdf.set_text_color(0, 0, 0)
        
        pdf.ln(4)
        
        # Section 3: Equity curve chart
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Equity Curve", ln=True)
        
        portfolio_value = backtest_results.get('portfolio_value')
        if portfolio_value is not None and len(portfolio_value) > 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            portfolio_value.plot(ax=ax, color='blue', linewidth=1.5)
            ax.set_title("Portfolio Value Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Portfolio Value ($)")
            ax.grid(True, alpha=0.3)
            
            chart_path = self._save_chart_to_temp(fig)
            pdf.image(chart_path, x=10, w=190)
            os.unlink(chart_path)  # Clean up temp file
        
        pdf.ln(4)
        
        # Section 4: SHAP feature importance chart
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Signal Drivers - SHAP Feature Importance", ln=True)
        
        feature_importance = shap_results.get('feature_importance', {})
        if feature_importance:
            # Get top 8 features
            top_features = list(feature_importance.items())[:8]
            feature_names = [f[0] for f in top_features]
            importance_values = [f[1] for f in top_features]
            
            # Get mean_shap_signed for direction
            mean_shap_signed = shap_results.get('mean_shap_signed')
            if mean_shap_signed is not None:
                feature_names_list = shap_results.get('feature_names', [])
                colors = []
                for fname in feature_names:
                    if fname in feature_names_list:
                        idx = feature_names_list.index(fname)
                        direction = mean_shap_signed[idx]
                        colors.append('green' if direction > 0 else 'red')
                    else:
                        colors.append('gray')
            else:
                colors = ['blue'] * len(feature_names)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            y_pos = np.arange(len(feature_names))
            ax.barh(y_pos, importance_values, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_names)
            ax.invert_yaxis()
            ax.set_xlabel("Mean |SHAP Value|")
            ax.set_ylabel("Feature")
            ax.set_title("Signal Drivers - SHAP Feature Importance")
            ax.grid(True, alpha=0.3, axis='x')
            
            chart_path = self._save_chart_to_temp(fig)
            pdf.image(chart_path, x=10, w=190)
            os.unlink(chart_path)
        
        pdf.ln(4)
        
        # Section 5: Drift analysis
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Distribution Stability", ln=True)
        pdf.set_font("Helvetica", "", 10)
        
        signal_psi = drift_results.get('signal_psi', 0)
        return_psi = drift_results.get('return_psi', 0)
        max_psi = drift_results.get('max_psi', 0)
        drift_level = drift_results.get('drift_level', 'unknown')
        recommendation = drift_results.get('recommendation', 'N/A')
        
        pdf.cell(100, 6, "Signal PSI", border=0)
        pdf.cell(0, 6, f"{signal_psi:.4f}", border=0, ln=True, align='R')
        
        pdf.cell(100, 6, "Return PSI", border=0)
        pdf.cell(0, 6, f"{return_psi:.4f}", border=0, ln=True, align='R')
        
        pdf.cell(100, 6, "Max PSI", border=0)
        pdf.cell(0, 6, f"{max_psi:.4f}", border=0, ln=True, align='R')
        
        # Drift level with color
        if drift_level == "none":
            drift_color = (0, 128, 0)  # Green
        elif drift_level == "moderate":
            drift_color = (255, 200, 0)  # Yellow
        elif drift_level == "significant":
            drift_color = (255, 0, 0)  # Red
        else:
            drift_color = (0, 0, 0)  # Black
        
        pdf.set_text_color(*drift_color)
        pdf.cell(100, 6, "Drift Level", border=0)
        pdf.cell(0, 6, drift_level.upper(), border=0, ln=True, align='R')
        pdf.set_text_color(0, 0, 0)
        
        pdf.cell(0, 6, f"Recommendation: {recommendation}", ln=True)
        
        pdf.ln(4)
        
        # Section 6: Generated code
        if generated_code:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Signal Implementation", ln=True)
            pdf.set_font("Courier", "", 8)
            
            # Truncate to 50 lines if longer
            code_lines = generated_code.split('\n')[:50]
            for line in code_lines:
                # Handle long lines
                if len(line) > 90:
                    line = line[:87] + "..."
                pdf.cell(0, 4, line, ln=True)
        
        # Save PDF
        strategy_name = "strategy"
        filename = f"{strategy_name}_{date.today().strftime('%Y%m%d')}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        pdf.output(filepath)
        
        logger.info(f"PDF report generated: {filepath}")
        
        return filepath
