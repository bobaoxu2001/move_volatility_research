"""
Visualization module for charts and dashboards.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Optional, Tuple

from .config import OUTPUT_CONFIG


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_time_series(
    data: pd.DataFrame,
    columns: List[str],
    title: str = '',
    ylabel: str = '',
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot time series data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to plot
    columns : list of str
        Column names to plot
    title : str
        Plot title
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for col in columns:
        ax.plot(data.index, data[col], label=col, alpha=0.8)
    
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=OUTPUT_CONFIG['figure_dpi'], bbox_inches='tight')
    
    return fig


def plot_zscore_with_regimes(
    data: pd.DataFrame,
    zscore_col: str = 'move_zscore',
    high_threshold: float = 1.0,
    low_threshold: float = -1.0,
    title: str = 'Z-Score with Regime Bands',
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot z-score with regime highlighting.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data containing z-score column
    zscore_col : str
        Z-score column name
    high_threshold : float
        High volatility threshold
    low_threshold : float
        Low volatility threshold
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(data.index, data[zscore_col], color='navy', alpha=0.8)
    ax.axhline(y=high_threshold, color='red', linestyle='--', alpha=0.5, label='High Vol')
    ax.axhline(y=low_threshold, color='green', linestyle='--', alpha=0.5, label='Low Vol')
    
    # Fill regions
    ax.fill_between(
        data.index, high_threshold, data[zscore_col].clip(lower=high_threshold).values,
        where=(data[zscore_col] > high_threshold).values, alpha=0.3, color='red'
    )
    ax.fill_between(
        data.index, low_threshold, data[zscore_col].clip(upper=low_threshold).values,
        where=(data[zscore_col] < low_threshold).values, alpha=0.3, color='green'
    )
    
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_ylabel('Z-Score')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=OUTPUT_CONFIG['figure_dpi'], bbox_inches='tight')
    
    return fig


def plot_scatter_with_fit(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = '',
    include_quadratic: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot scatter with linear and optionally quadratic fit.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to plot
    x_col : str
        X column name
    y_col : str
        Y column name
    title : str
        Plot title
    include_quadratic : bool
        Whether to include quadratic fit
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter
    ax.scatter(data[x_col], data[y_col], alpha=0.3, s=10)
    
    # Linear fit
    x_clean = data[x_col].dropna()
    y_clean = data[y_col].dropna()
    common_idx = x_clean.index.intersection(y_clean.index)
    x_vals = x_clean.loc[common_idx].values
    y_vals = y_clean.loc[common_idx].values
    
    z1 = np.polyfit(x_vals, y_vals, 1)
    p1 = np.poly1d(z1)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax.plot(x_line, p1(x_line), 'r-', linewidth=2, label=f'Linear β={z1[0]:.3f}')
    
    # Quadratic fit
    if include_quadratic:
        z2 = np.polyfit(x_vals, y_vals, 2)
        p2 = np.poly1d(z2)
        ax.plot(x_line, p2(x_line), 'b--', linewidth=2, label='Quadratic')
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=OUTPUT_CONFIG['figure_dpi'], bbox_inches='tight')
    
    return fig


def plot_rolling_ic(
    ic_series: pd.Series,
    title: str = 'Rolling Information Coefficient',
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot rolling IC with fill colors.
    
    Parameters
    ----------
    ic_series : pd.Series
        Rolling IC values
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(ic_series.index, ic_series, color='navy', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axhline(y=0.03, color='green', linestyle='--', alpha=0.5, label='Weak IC')
    ax.axhline(y=0.05, color='blue', linestyle='--', alpha=0.5, label='Good IC')
    
    # Fill positive/negative
    ic_filled = ic_series.fillna(0)
    ax.fill_between(
        ic_series.index, 0, ic_filled.clip(lower=0).values,
        where=(ic_filled > 0).values, alpha=0.3, color='green'
    )
    ax.fill_between(
        ic_series.index, 0, ic_filled.clip(upper=0).values,
        where=(ic_filled < 0).values, alpha=0.3, color='red'
    )
    
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_ylabel('Spearman IC')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=OUTPUT_CONFIG['figure_dpi'], bbox_inches='tight')
    
    return fig


def plot_regime_comparison(
    regime_metrics: pd.DataFrame,
    metric_col: str = 'ic',
    title: str = 'IC by Regime',
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot bar chart comparing regimes.
    
    Parameters
    ----------
    regime_metrics : pd.DataFrame
        DataFrame with regime and metric columns
    metric_col : str
        Metric column to plot
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = {'Low Vol': 'green', 'Normal': 'gray', 'High Vol': 'red',
              'Low VIX': 'green', 'Normal VIX': 'gray', 'High VIX': 'red'}
    
    bar_colors = [colors.get(r, 'steelblue') for r in regime_metrics['regime']]
    
    ax.bar(regime_metrics['regime'], regime_metrics[metric_col], color=bar_colors)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_ylabel(metric_col.upper())
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=OUTPUT_CONFIG['figure_dpi'], bbox_inches='tight')
    
    return fig


def plot_model_comparison(
    model_results: pd.DataFrame,
    metric_col: str = 'oos_r2',
    title: str = 'Model Comparison',
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot bar chart comparing models.
    
    Parameters
    ----------
    model_results : pd.DataFrame
        DataFrame with model and metric columns
    metric_col : str
        Metric column to plot
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['navy', 'orange', 'green', 'red', 'purple']
    
    ax.bar(model_results['model'], model_results[metric_col], 
           color=colors[:len(model_results)])
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_ylabel(metric_col.replace('_', ' ').title())
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=OUTPUT_CONFIG['figure_dpi'], bbox_inches='tight')
    
    return fig


def plot_feature_importance(
    feature_names: List[str],
    coefficients: np.ndarray,
    title: str = 'Feature Importance',
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot horizontal bar chart of feature importance.
    
    Parameters
    ----------
    feature_names : list of str
        Feature names
    coefficients : np.ndarray
        Coefficient values
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by absolute value
    importance = pd.DataFrame({
        'feature': feature_names,
        'coef': np.abs(coefficients)
    }).sort_values('coef', ascending=True)
    
    ax.barh(importance['feature'], importance['coef'], color='steelblue')
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xlabel('Absolute Coefficient')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=OUTPUT_CONFIG['figure_dpi'], bbox_inches='tight')
    
    return fig


def plot_key_chart(
    data: pd.DataFrame,
    metrics: Dict[str, float],
    title: str = 'MOVE Index as Predictor of Treasury Term-Spread Volatility',
    figsize: Tuple[int, int] = (14, 7),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate the key summary chart.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with move_zscore and spread_vol_zscore columns
    metrics : dict
        Dictionary of metrics to display
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Primary axis: MOVE Z-Score
    color1 = 'navy'
    ax1.plot(data.index, data['move_zscore'], color=color1, 
             linewidth=1.5, alpha=0.9, label='MOVE Z-Score')
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.4)
    ax1.axhline(y=-1, color='green', linestyle='--', alpha=0.4)
    
    # Fill regime regions
    ax1.fill_between(
        data.index, 1, data['move_zscore'].clip(lower=1).values,
        alpha=0.2, color='red', label='High Vol Regime',
        where=(data['move_zscore'] > 1).values
    )
    ax1.fill_between(
        data.index, -1, data['move_zscore'].clip(upper=-1).values,
        alpha=0.2, color='green', label='Low Vol Regime',
        where=(data['move_zscore'] < -1).values
    )
    
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('MOVE Z-Score', color=color1, fontsize=11)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Secondary axis: Spread Volatility
    ax2 = ax1.twinx()
    color2 = 'darkorange'
    ax2.plot(data.index, data['spread_vol_zscore'], color=color2, 
             linewidth=1.5, alpha=0.7, label='Spread Vol Z-Score')
    ax2.set_ylabel('Spread Volatility Z-Score', color=color2, fontsize=11)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    ax1.set_title(title + '\nEnhanced with Regularization & Time Series Models',
                  fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Metrics box
    textstr = '\n'.join([
        f"Mean IC: {metrics.get('mean_ic', 0):.3f}",
        f"OOS IC: {metrics.get('oos_ic', 0):.3f}",
        f"Best Model: {metrics.get('best_model', 'N/A')}",
        f"Granger p: {metrics.get('granger_p', 'N/A')}"
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    return fig


def create_eda_dashboard(
    data: pd.DataFrame,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create EDA dashboard with multiple panels.
    
    Parameters
    ----------
    data : pd.DataFrame
        Processed data
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Panel 1: MOVE Z-Score with regimes
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(data.index, data['move_zscore'], color='navy', alpha=0.8)
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(y=-1, color='green', linestyle='--', alpha=0.5)
    ax1.fill_between(
        data.index, 1, data['move_zscore'].clip(lower=1).values,
        where=(data['move_zscore'] > 1).values, alpha=0.3, color='red'
    )
    ax1.fill_between(
        data.index, -1, data['move_zscore'].clip(upper=-1).values,
        where=(data['move_zscore'] < -1).values, alpha=0.3, color='green'
    )
    ax1.set_title('MOVE Z-Score with Regime Bands', fontweight='bold')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Panel 2: Scatter with fit
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(data['move_zscore'], data['spread_vol_zscore'], alpha=0.3, s=10)
    x_vals = data['move_zscore'].dropna().values
    y_vals = data['spread_vol_zscore'].dropna().values
    common_len = min(len(x_vals), len(y_vals))
    z1 = np.polyfit(x_vals[:common_len], y_vals[:common_len], 1)
    z2 = np.polyfit(x_vals[:common_len], y_vals[:common_len], 2)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax2.plot(x_line, np.poly1d(z1)(x_line), 'r-', linewidth=2, label=f'Linear β={z1[0]:.3f}')
    ax2.plot(x_line, np.poly1d(z2)(x_line), 'b--', linewidth=2, label='Quadratic')
    ax2.set_xlabel('MOVE Z-Score')
    ax2.set_ylabel('Spread Vol Z-Score')
    ax2.set_title('MOVE → Spread Vol (Linear vs Quadratic)', fontweight='bold')
    ax2.legend()
    
    # Panel 3: VIX vs VXN
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(data.index, data['vix'], label='VIX', alpha=0.7)
    ax3.plot(data.index, data['vxn'], label='VXN', alpha=0.7)
    ax3.set_title('VIX vs VXN', fontweight='bold')
    ax3.legend()
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Panel 4: Spread Volatility
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(data.index, data['spread_volatility'], color='purple', alpha=0.8)
    ax4.set_title('Treasury Term-Spread Volatility Proxy', fontweight='bold')
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=OUTPUT_CONFIG['figure_dpi'], bbox_inches='tight')
    
    return fig
