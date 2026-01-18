"""Hyperparameter Analysis and Visualization Tools.

Provides comprehensive analysis capabilities for hyperparameter tuning results:
- Interactive visualizations (Plotly)
- Parameter importance calculation
- High-dimensional visualization (HiPlot)
- Statistical analysis
- Report generation

Example:
    >>> from medai_compass.optimization.analysis import HPAnalyzer
    >>> analyzer = HPAnalyzer()
    >>> analyzer.load_results_from_path("/path/to/tune/results")
    >>> fig = analyzer.plot_parallel_coordinates()
    >>> importance = analyzer.calculate_parameter_importance()
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Import Plotly with graceful fallback
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None

# Import HiPlot with graceful fallback
try:
    import hiplot as hip
    HIPLOT_AVAILABLE = True
except ImportError:
    HIPLOT_AVAILABLE = False
    hip = None

# Import Ray Tune for results loading
try:
    from ray.tune import ResultGrid, ExperimentAnalysis
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


@dataclass
class TrialResult:
    """Container for a single trial's results.
    
    Attributes:
        trial_id: Unique trial identifier
        config: Hyperparameter configuration
        metrics: Trial metrics (loss, accuracy, etc.)
        status: Trial status (RUNNING, TERMINATED, ERROR)
        start_time: Trial start timestamp
        end_time: Trial end timestamp
    """
    trial_id: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    status: str = "COMPLETED"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate trial duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trial_id": self.trial_id,
            "config": self.config,
            "metrics": self.metrics,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class AnalysisReport:
    """Comprehensive analysis report.
    
    Attributes:
        best_trial: Best performing trial
        parameter_importance: Importance scores for each parameter
        correlation_matrix: Parameter-metric correlations
        summary_statistics: Summary statistics for all metrics
        recommendations: Optimization recommendations
    """
    best_trial: TrialResult
    parameter_importance: Dict[str, float]
    correlation_matrix: Dict[str, Dict[str, float]]
    summary_statistics: Dict[str, Dict[str, float]]
    recommendations: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "best_trial": self.best_trial.to_dict(),
            "parameter_importance": self.parameter_importance,
            "correlation_matrix": self.correlation_matrix,
            "summary_statistics": self.summary_statistics,
            "recommendations": self.recommendations,
            "generated_at": self.generated_at.isoformat(),
        }
    
    def to_json(self, filepath: str) -> None:
        """Save report to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def to_markdown(self) -> str:
        """Generate markdown formatted report."""
        lines = [
            "# Hyperparameter Tuning Analysis Report",
            f"\nGenerated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Best Trial Configuration",
            f"\n**Trial ID:** {self.best_trial.trial_id}",
            f"\n**Status:** {self.best_trial.status}",
            "\n### Configuration",
        ]
        
        for param, value in self.best_trial.config.items():
            lines.append(f"- **{param}:** {value}")
        
        lines.extend([
            "\n### Metrics",
        ])
        
        for metric, value in self.best_trial.metrics.items():
            lines.append(f"- **{metric}:** {value:.6f}")
        
        lines.extend([
            "\n## Parameter Importance",
        ])
        
        sorted_importance = sorted(
            self.parameter_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for param, importance in sorted_importance:
            lines.append(f"- **{param}:** {importance:.4f}")
        
        lines.extend([
            "\n## Recommendations",
        ])
        
        for i, rec in enumerate(self.recommendations, 1):
            lines.append(f"{i}. {rec}")
        
        return "\n".join(lines)


class HPAnalyzer:
    """Hyperparameter analysis and visualization tool.
    
    Provides comprehensive analysis of hyperparameter tuning results
    with interactive visualizations and statistical analysis.
    
    Example:
        >>> analyzer = HPAnalyzer()
        >>> analyzer.load_results_from_path("/path/to/results")
        >>> 
        >>> # Generate visualizations
        >>> fig = analyzer.plot_parallel_coordinates()
        >>> fig.show()
        >>> 
        >>> # Calculate importance
        >>> importance = analyzer.calculate_parameter_importance()
        >>> print(importance)
        >>> 
        >>> # Generate report
        >>> report = analyzer.generate_report()
        >>> print(report.to_markdown())
    """
    
    def __init__(self):
        """Initialize HPAnalyzer."""
        self._trials: List[TrialResult] = []
        self._df: Optional[pd.DataFrame] = None
        self._metric_columns: List[str] = []
        self._param_columns: List[str] = []
        self._experiment_name: str = ""
    
    @property
    def trials(self) -> List[TrialResult]:
        """Get all trial results."""
        return self._trials
    
    @property
    def dataframe(self) -> pd.DataFrame:
        """Get trials as pandas DataFrame."""
        if self._df is None:
            self._df = self._create_dataframe()
        return self._df
    
    @property
    def num_trials(self) -> int:
        """Get number of trials."""
        return len(self._trials)
    
    @property
    def metric_columns(self) -> List[str]:
        """Get metric column names."""
        return self._metric_columns
    
    @property
    def param_columns(self) -> List[str]:
        """Get parameter column names."""
        return self._param_columns
    
    def load_results_from_path(self, path: str) -> "HPAnalyzer":
        """Load results from Ray Tune experiment path.
        
        Args:
            path: Path to Ray Tune experiment results
            
        Returns:
            Self for chaining
        """
        if RAY_AVAILABLE:
            try:
                analysis = ExperimentAnalysis(path)
                
                for trial_path, trial_data in analysis.trial_dataframes.items():
                    if len(trial_data) > 0:
                        # Get last row (final result)
                        final = trial_data.iloc[-1].to_dict()
                        
                        # Separate config from metrics
                        config = {}
                        metrics = {}
                        
                        for key, value in final.items():
                            if key.startswith("config/"):
                                config[key.replace("config/", "")] = value
                            elif isinstance(value, (int, float)) and not pd.isna(value):
                                metrics[key] = value
                        
                        trial = TrialResult(
                            trial_id=os.path.basename(trial_path),
                            config=config,
                            metrics=metrics,
                            status="COMPLETED",
                        )
                        self._trials.append(trial)
                
                self._experiment_name = os.path.basename(path)
                self._update_columns()
                
            except Exception as e:
                logger.warning(f"Failed to load Ray results: {e}")
        
        return self
    
    def load_results_from_dict(
        self,
        results: List[Dict[str, Any]],
    ) -> "HPAnalyzer":
        """Load results from list of dictionaries.
        
        Args:
            results: List of trial result dictionaries
            
        Returns:
            Self for chaining
        """
        for i, result in enumerate(results):
            trial = TrialResult(
                trial_id=result.get("trial_id", f"trial_{i}"),
                config=result.get("config", {}),
                metrics=result.get("metrics", {}),
                status=result.get("status", "COMPLETED"),
            )
            self._trials.append(trial)
        
        self._update_columns()
        return self
    
    def _update_columns(self) -> None:
        """Update metric and parameter column lists."""
        if self._trials:
            self._param_columns = list(self._trials[0].config.keys())
            self._metric_columns = list(self._trials[0].metrics.keys())
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Create DataFrame from trials."""
        if not self._trials:
            return pd.DataFrame()
        
        data = []
        for trial in self._trials:
            row = {
                "trial_id": trial.trial_id,
                "status": trial.status,
            }
            row.update({f"config_{k}": v for k, v in trial.config.items()})
            row.update({f"metric_{k}": v for k, v in trial.metrics.items()})
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_best_trial(
        self,
        metric: str = "eval_loss",
        mode: str = "min",
    ) -> TrialResult:
        """Get the best trial by a specific metric.
        
        Args:
            metric: Metric to optimize
            mode: "min" or "max"
            
        Returns:
            Best performing trial
        """
        if not self._trials:
            raise ValueError("No trials loaded")
        
        best_trial = None
        best_value = float("inf") if mode == "min" else float("-inf")
        
        for trial in self._trials:
            value = trial.metrics.get(metric, float("inf") if mode == "min" else float("-inf"))
            
            if mode == "min" and value < best_value:
                best_value = value
                best_trial = trial
            elif mode == "max" and value > best_value:
                best_value = value
                best_trial = trial
        
        return best_trial
    
    # =========================================================================
    # Visualization Methods
    # =========================================================================
    
    def plot_parallel_coordinates(
        self,
        color_metric: str = "eval_loss",
        params: Optional[List[str]] = None,
    ) -> "go.Figure":
        """Create parallel coordinates plot.
        
        Args:
            color_metric: Metric to use for coloring
            params: Parameters to include (None for all)
            
        Returns:
            Plotly Figure
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly not available")
        
        df = self.dataframe
        if df.empty:
            raise ValueError("No data loaded")
        
        # Select parameter columns
        if params is None:
            params = self._param_columns
        
        config_cols = [f"config_{p}" for p in params if f"config_{p}" in df.columns]
        metric_col = f"metric_{color_metric}" if f"metric_{color_metric}" in df.columns else None
        
        # Build dimensions for parallel coordinates
        dimensions = []
        
        for col in config_cols:
            values = df[col].dropna()
            if len(values) == 0:
                continue
                
            if values.dtype in [np.int64, np.float64]:
                dimensions.append(
                    dict(
                        label=col.replace("config_", ""),
                        values=df[col],
                        range=[values.min(), values.max()],
                    )
                )
            else:
                # Categorical
                unique_vals = values.unique()
                tick_vals = list(range(len(unique_vals)))
                tick_text = [str(v) for v in unique_vals]
                val_map = {v: i for i, v in enumerate(unique_vals)}
                
                dimensions.append(
                    dict(
                        label=col.replace("config_", ""),
                        values=df[col].map(val_map),
                        tickvals=tick_vals,
                        ticktext=tick_text,
                    )
                )
        
        # Add metric
        if metric_col and metric_col in df.columns:
            metric_values = df[metric_col].dropna()
            dimensions.append(
                dict(
                    label=color_metric,
                    values=df[metric_col],
                    range=[metric_values.min(), metric_values.max()],
                )
            )
        
        # Create figure
        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=df[metric_col] if metric_col else None,
                colorscale="Viridis",
                showscale=True,
                cmin=df[metric_col].min() if metric_col else 0,
                cmax=df[metric_col].max() if metric_col else 1,
            ),
            dimensions=dimensions,
        ))
        
        fig.update_layout(
            title=f"Parallel Coordinates - {self._experiment_name}",
            width=1200,
            height=600,
        )
        
        return fig
    
    def plot_optimization_history(
        self,
        metric: str = "eval_loss",
        mode: str = "min",
    ) -> "go.Figure":
        """Create optimization history plot.
        
        Shows best metric value over trials.
        
        Args:
            metric: Metric to plot
            mode: "min" or "max"
            
        Returns:
            Plotly Figure
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly not available")
        
        if not self._trials:
            raise ValueError("No trials loaded")
        
        # Get metric values in order
        values = [t.metrics.get(metric, np.nan) for t in self._trials]
        
        # Calculate running best
        running_best = []
        best_so_far = float("inf") if mode == "min" else float("-inf")
        
        for v in values:
            if not np.isnan(v):
                if mode == "min":
                    best_so_far = min(best_so_far, v)
                else:
                    best_so_far = max(best_so_far, v)
            running_best.append(best_so_far)
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f"{metric} per Trial", f"Best {metric} Over Time"),
            vertical_spacing=0.15,
        )
        
        # Individual trial values
        fig.add_trace(
            go.Scatter(
                x=list(range(len(values))),
                y=values,
                mode="markers",
                name="Trial Value",
                marker=dict(size=8, color="blue", opacity=0.6),
            ),
            row=1, col=1,
        )
        
        # Running best
        fig.add_trace(
            go.Scatter(
                x=list(range(len(running_best))),
                y=running_best,
                mode="lines+markers",
                name="Best So Far",
                line=dict(color="red", width=2),
                marker=dict(size=6),
            ),
            row=2, col=1,
        )
        
        fig.update_layout(
            title=f"Optimization History - {self._experiment_name}",
            height=700,
            width=1000,
            showlegend=True,
        )
        
        fig.update_xaxes(title_text="Trial Number", row=1, col=1)
        fig.update_xaxes(title_text="Trial Number", row=2, col=1)
        fig.update_yaxes(title_text=metric, row=1, col=1)
        fig.update_yaxes(title_text=f"Best {metric}", row=2, col=1)
        
        return fig
    
    def plot_parameter_importance(
        self,
        metric: str = "eval_loss",
    ) -> "go.Figure":
        """Create parameter importance bar chart.
        
        Args:
            metric: Target metric for importance calculation
            
        Returns:
            Plotly Figure
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly not available")
        
        importance = self.calculate_parameter_importance(metric)
        
        # Sort by importance
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        params = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        fig = go.Figure(data=[
            go.Bar(
                x=values,
                y=params,
                orientation="h",
                marker=dict(
                    color=values,
                    colorscale="Viridis",
                ),
            )
        ])
        
        fig.update_layout(
            title=f"Parameter Importance for {metric}",
            xaxis_title="Importance Score",
            yaxis_title="Parameter",
            height=400 + len(params) * 30,
            width=800,
        )
        
        return fig
    
    def plot_contour(
        self,
        param1: str,
        param2: str,
        metric: str = "eval_loss",
    ) -> "go.Figure":
        """Create contour plot for two parameters.
        
        Args:
            param1: First parameter
            param2: Second parameter
            metric: Target metric
            
        Returns:
            Plotly Figure
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly not available")
        
        df = self.dataframe
        
        col1 = f"config_{param1}"
        col2 = f"config_{param2}"
        metric_col = f"metric_{metric}"
        
        if col1 not in df.columns or col2 not in df.columns:
            raise ValueError(f"Parameters {param1} or {param2} not found")
        
        if metric_col not in df.columns:
            raise ValueError(f"Metric {metric} not found")
        
        fig = go.Figure(data=go.Scatter(
            x=df[col1],
            y=df[col2],
            mode="markers",
            marker=dict(
                size=12,
                color=df[metric_col],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=metric),
            ),
            text=[f"{metric}: {v:.4f}" for v in df[metric_col]],
            hovertemplate=f"{param1}: %{{x}}<br>{param2}: %{{y}}<br>%{{text}}<extra></extra>",
        ))
        
        fig.update_layout(
            title=f"{param1} vs {param2} - {metric}",
            xaxis_title=param1,
            yaxis_title=param2,
            width=800,
            height=600,
        )
        
        return fig
    
    def create_hiplot_experiment(self) -> "hip.Experiment":
        """Create HiPlot experiment for high-dimensional visualization.
        
        Returns:
            HiPlot Experiment object
        """
        if not HIPLOT_AVAILABLE:
            raise ImportError("HiPlot not available. Install with: pip install hiplot")
        
        if not self._trials:
            raise ValueError("No trials loaded")
        
        data = []
        for trial in self._trials:
            row = {"trial_id": trial.trial_id, "status": trial.status}
            row.update(trial.config)
            row.update(trial.metrics)
            data.append(row)
        
        return hip.Experiment.from_iterable(data)
    
    # =========================================================================
    # Analysis Methods
    # =========================================================================
    
    def calculate_parameter_importance(
        self,
        metric: str = "eval_loss",
        method: str = "correlation",
    ) -> Dict[str, float]:
        """Calculate parameter importance scores.
        
        Uses correlation-based importance or f-ANOVA (if available).
        
        Args:
            metric: Target metric
            method: "correlation" or "fanova"
            
        Returns:
            Dictionary of parameter -> importance score
        """
        df = self.dataframe
        metric_col = f"metric_{metric}"
        
        if metric_col not in df.columns:
            raise ValueError(f"Metric {metric} not found")
        
        importance = {}
        
        for param in self._param_columns:
            col = f"config_{param}"
            if col not in df.columns:
                continue
            
            # Get numeric values
            values = df[col]
            if values.dtype == object:
                # Encode categorical
                unique_vals = values.unique()
                val_map = {v: i for i, v in enumerate(unique_vals)}
                values = values.map(val_map)
            
            # Skip if all same value
            if values.nunique() <= 1:
                importance[param] = 0.0
                continue
            
            # Calculate correlation
            valid_mask = ~(values.isna() | df[metric_col].isna())
            if valid_mask.sum() < 3:
                importance[param] = 0.0
                continue
            
            corr = np.abs(np.corrcoef(
                values[valid_mask].astype(float),
                df.loc[valid_mask, metric_col].astype(float)
            )[0, 1])
            
            importance[param] = float(corr) if not np.isnan(corr) else 0.0
        
        # Normalize to [0, 1]
        max_imp = max(importance.values()) if importance else 1.0
        if max_imp > 0:
            importance = {k: v / max_imp for k, v in importance.items()}
        
        return importance
    
    def calculate_correlations(
        self,
        metric: str = "eval_loss",
    ) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix between parameters and metrics.
        
        Args:
            metric: Primary metric for analysis
            
        Returns:
            Correlation matrix as nested dictionary
        """
        df = self.dataframe
        
        # Select numeric columns
        numeric_cols = []
        for col in df.columns:
            if col.startswith("config_") or col.startswith("metric_"):
                if df[col].dtype in [np.int64, np.float64]:
                    numeric_cols.append(col)
        
        if not numeric_cols:
            return {}
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Convert to dict
        result = {}
        for col in corr_matrix.columns:
            result[col] = {}
            for row in corr_matrix.index:
                result[col][row] = float(corr_matrix.loc[row, col])
        
        return result
    
    def calculate_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate summary statistics for all metrics.
        
        Returns:
            Statistics dictionary with mean, std, min, max, median
        """
        df = self.dataframe
        stats = {}
        
        for metric in self._metric_columns:
            col = f"metric_{metric}"
            if col not in df.columns:
                continue
            
            values = df[col].dropna()
            if len(values) == 0:
                continue
            
            stats[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "median": float(values.median()),
                "count": int(len(values)),
            }
        
        return stats
    
    def generate_recommendations(
        self,
        metric: str = "eval_loss",
        mode: str = "min",
    ) -> List[str]:
        """Generate optimization recommendations based on analysis.
        
        Args:
            metric: Target metric
            mode: "min" or "max"
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Get best trial
        best_trial = self.get_best_trial(metric, mode)
        
        # Calculate importance
        importance = self.calculate_parameter_importance(metric)
        
        # Sort by importance
        sorted_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        # Recommend focusing on most important parameters
        if sorted_params:
            top_params = [p[0] for p in sorted_params[:3] if p[1] > 0.1]
            if top_params:
                recommendations.append(
                    f"Focus tuning efforts on: {', '.join(top_params)} (highest importance)"
                )
        
        # Check for best config patterns
        best_config = best_trial.config
        
        # Learning rate recommendations
        if "learning_rate" in best_config:
            lr = best_config["learning_rate"]
            if lr < 1e-4:
                recommendations.append(
                    f"Best learning rate ({lr:.2e}) is relatively low - consider using warmup"
                )
            elif lr > 5e-4:
                recommendations.append(
                    f"Best learning rate ({lr:.2e}) is relatively high - monitor for instability"
                )
        
        # LoRA recommendations
        if "lora_r" in best_config:
            lora_r = best_config["lora_r"]
            if lora_r >= 64:
                recommendations.append(
                    f"High LoRA rank ({lora_r}) - consider if smaller rank suffices to reduce memory"
                )
        
        # Batch size recommendations
        if "batch_size" in best_config:
            batch_size = best_config.get("batch_size", 0)
            if batch_size == 1:
                recommendations.append(
                    "Batch size is 1 - consider gradient accumulation for stability"
                )
        
        # Sample size recommendations
        if self.num_trials < 20:
            recommendations.append(
                f"Only {self.num_trials} trials completed - consider running more for robust conclusions"
            )
        
        return recommendations
    
    def generate_report(
        self,
        metric: str = "eval_loss",
        mode: str = "min",
    ) -> AnalysisReport:
        """Generate comprehensive analysis report.
        
        Args:
            metric: Target metric
            mode: "min" or "max"
            
        Returns:
            Complete AnalysisReport
        """
        best_trial = self.get_best_trial(metric, mode)
        importance = self.calculate_parameter_importance(metric)
        correlations = self.calculate_correlations(metric)
        stats = self.calculate_summary_statistics()
        recommendations = self.generate_recommendations(metric, mode)
        
        return AnalysisReport(
            best_trial=best_trial,
            parameter_importance=importance,
            correlation_matrix=correlations,
            summary_statistics=stats,
            recommendations=recommendations,
        )
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def save_figure(
        self,
        fig: "go.Figure",
        filepath: str,
        format: Optional[str] = None,
    ) -> None:
        """Save Plotly figure to file.
        
        Args:
            fig: Plotly Figure
            filepath: Output file path
            format: Output format (html, png, pdf, svg)
        """
        if format is None:
            format = Path(filepath).suffix.lstrip(".")
        
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        
        if format == "html":
            fig.write_html(filepath)
        else:
            try:
                fig.write_image(filepath, format=format)
            except Exception as e:
                logger.warning(f"Failed to save as {format}, saving as HTML: {e}")
                fig.write_html(filepath.replace(f".{format}", ".html"))
    
    def export_to_csv(self, filepath: str) -> None:
        """Export results to CSV file.
        
        Args:
            filepath: Output CSV path
        """
        df = self.dataframe
        df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(df)} trials to {filepath}")
    
    def export_to_json(self, filepath: str) -> None:
        """Export results to JSON file.
        
        Args:
            filepath: Output JSON path
        """
        data = [t.to_dict() for t in self._trials]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported {len(data)} trials to {filepath}")


def create_comparison_plot(
    experiments: Dict[str, HPAnalyzer],
    metric: str = "eval_loss",
    mode: str = "min",
) -> "go.Figure":
    """Create comparison plot across multiple experiments.
    
    Args:
        experiments: Dictionary of experiment_name -> HPAnalyzer
        metric: Metric to compare
        mode: "min" or "max"
        
    Returns:
        Plotly Figure
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly not available")
    
    fig = go.Figure()
    
    for name, analyzer in experiments.items():
        values = [t.metrics.get(metric, np.nan) for t in analyzer.trials]
        
        # Calculate running best
        running_best = []
        best_so_far = float("inf") if mode == "min" else float("-inf")
        
        for v in values:
            if not np.isnan(v):
                if mode == "min":
                    best_so_far = min(best_so_far, v)
                else:
                    best_so_far = max(best_so_far, v)
            running_best.append(best_so_far)
        
        fig.add_trace(go.Scatter(
            x=list(range(len(running_best))),
            y=running_best,
            mode="lines+markers",
            name=name,
        ))
    
    fig.update_layout(
        title=f"Experiment Comparison - {metric}",
        xaxis_title="Trial Number",
        yaxis_title=f"Best {metric}",
        width=1000,
        height=600,
    )
    
    return fig
