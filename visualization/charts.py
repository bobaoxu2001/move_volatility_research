"""
Visualization utilities for MOVE volatility research.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from ..config import CONFIG


def setup_style():
    """Set up professional chart style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")


def plot_eda_dashboard(df: pd.DataFrame, save_path: Optional[str] = None,
                       dpi: int = 150) -> plt.Figure:
    """
    Create EDA dashboard with 4 panels.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    save_path : str, optional
        Path to save figure
    dpi : int
        Figure resolution
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    setup_style()
    fig = plt.figure(figsize=(16, 8))
    
    # 1. MOVE Z-Score with regime bands
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(df.index, df['move_zscore'], color='navy', alpha=0.8)
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(y=-1, color='green', linestyle='--', alpha=0.5)
    ax1.fill_between(df.index, 1, df['move_zscore'].clip(lower=1).values,
                     where=(df['move_zscore'] > 1).values, alpha=0.3, color='red')
    ax1.fill_between(df.index, -1, df['move_zscore'].clip(upper=-1).values,
                     where=(df['move_zscore'] < -1).values, alpha=0.3, color='green')
    ax1.set_title('MOVE Z-Score with Regime Bands', fontweight='bold')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # 2. Scatter with linear and quadratic fit
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(df['move_zscore'], df['spread_vol_zscore'], alpha=0.3, s=10)
    
    x_clean = df['move_zscore'].dropna()
    y_clean = df['spread_vol_zscore'].dropna()
    common_idx = x_clean.index.intersection(y_clean.index)
    x_clean = x_clean.loc[common_idx]
    y_clean = y_clean.loc[common_idx]
    
    z1 = np.polyfit(x_clean, y_clean, 1)
    z2 = np.polyfit(x_clean, y_clean, 2)
    p1 = np.poly1d(z1)
    p2 = np.poly1d(z2)
    x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
    
    ax2.plot(x_line, p1(x_line), 'r-', linewidth=2, label=f'Linear β={z1[0]:.3f}')
    ax2.plot(x_line, p2(x_line), 'b--', linewidth=2, label='Quadratic')
    ax2.set_xlabel('MOVE Z-Score')
    ax2.set_ylabel('Spread Vol Z-Score')
    ax2.set_title('MOVE → Spread Vol (Linear vs Quadratic)', fontweight='bold')
    ax2.legend()
    
    # 3. VIX vs VXN
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(df.index, df['vix'], label='VIX', alpha=0.7)
    ax3.plot(df.index, df['vxn'], label='VXN', alpha=0.7)
    ax3.set_title('VIX vs VXN', fontweight='bold')
    ax3.legend()
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # 4. Spread volatility
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(df.index, df['spread_volatility'], color='purple', alpha=0.8)
    ax4.set_title('Treasury Term-Spread Volatility Proxy', fontweight='bold')
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_advanced_analysis(df: pd.DataFrame, 
                           ic_series: pd.Series,
                           regime_stats: pd.DataFrame,
                           model_results: Dict,
                           lasso_coefs: Dict,
                           save_path: Optional[str] = None,
                           dpi: int = 150) -> plt.Figure:
    """
    Create advanced analysis dashboard.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    ic_series : pd.Series
        Rolling IC series
    regime_stats : pd.DataFrame
        Regime statistics
    model_results : dict
        Model comparison results
    lasso_coefs : dict
        LASSO coefficients
    save_path : str, optional
        Path to save figure
    dpi : int
        Figure resolution
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    setup_style()
    fig = plt.figure(figsize=(16, 12))
    
    # 1. IC by regime
    ax1 = fig.add_subplot(2, 2, 1)
    colors = {'Low Vol': 'green', 'Normal': 'gray', 'High Vol': 'red'}
    regime_colors = [colors.get(r, 'gray') for r in regime_stats['regime']]
    ax1.bar(regime_stats['regime'], regime_stats['ic'], color=regime_colors)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_title('IC by MOVE Regime', fontweight='bold')
    ax1.set_ylabel('Spearman IC')
    
    # 2. OOS R² by model
    ax2 = fig.add_subplot(2, 2, 2)
    models = list(model_results['models'].keys())
    oos_r2s = [model_results['models'][m]['oos_r2'] for m in models]
    colors_model = ['navy', 'orange', 'green'][:len(models)]
    ax2.bar([m.upper() for m in models], oos_r2s, color=colors_model)
    ax2.set_title('OOS R² by Model', fontweight='bold')
    ax2.set_ylabel('R²')
    
    # 3. Rolling IC
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(ic_series.index, ic_series, color='navy', alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.fill_between(ic_series.index, 0, ic_series.fillna(0).clip(lower=0).values,
                     where=(ic_series.fillna(0) > 0).values, alpha=0.3, color='green')
    ax3.fill_between(ic_series.index, 0, ic_series.fillna(0).clip(upper=0).values,
                     where=(ic_series.fillna(0) < 0).values, alpha=0.3, color='red')
    ax3.set_title('Rolling Spearman IC', fontweight='bold')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # 4. LASSO feature importance
    ax4 = fig.add_subplot(2, 2, 4)
    feat_df = pd.DataFrame({
        'feature': list(lasso_coefs.keys()),
        'coef': [abs(v) for v in lasso_coefs.values()]
    }).sort_values('coef', ascending=True)
    ax4.barh(feat_df['feature'], feat_df['coef'], color='steelblue')
    ax4.set_title('LASSO Feature Importance (|coef|)', fontweight='bold')
    ax4.set_xlabel('Absolute Coefficient')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_key_chart(df: pd.DataFrame,
                   metrics: Dict,
                   save_path: Optional[str] = None,
                   dpi: int = 200) -> plt.Figure:
    """
    Create key summary chart for report.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    metrics : dict
        Key metrics to display
    save_path : str, optional
        Path to save figure
    dpi : int
        Figure resolution
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    setup_style()
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Primary axis: MOVE Z-Score
    color1 = 'navy'
    ax1.plot(df.index, df['move_zscore'], color=color1,
             linewidth=1.5, alpha=0.9, label='MOVE Z-Score')
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.4)
    ax1.axhline(y=-1, color='green', linestyle='--', alpha=0.4)
    ax1.fill_between(df.index, 1, df['move_zscore'].clip(lower=1).values,
                     alpha=0.2, color='red', label='High Vol Regime',
                     where=(df['move_zscore'] > 1).values)
    ax1.fill_between(df.index, -1, df['move_zscore'].clip(upper=-1).values,
                     alpha=0.2, color='green', label='Low Vol Regime',
                     where=(df['move_zscore'] < -1).values)
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('MOVE Z-Score', color=color1, fontsize=11)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Secondary axis: Spread Volatility
    ax2 = ax1.twinx()
    color2 = 'darkorange'
    ax2.plot(df.index, df['spread_vol_zscore'], color=color2,
             linewidth=1.5, alpha=0.7, label='Spread Vol Z-Score')
    ax2.set_ylabel('Spread Volatility Z-Score', color=color2, fontsize=11)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    ax1.set_title('MOVE Index as Predictor of Treasury Term-Spread Volatility Proxy\n'
                  'Enhanced with Regularization & Time Series Models',
                  fontsize=14, fontweight='bold', pad=20)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Metrics box
    textstr = '\n'.join([
        f"Mean IC: {metrics.get('mean_ic', 0):.3f}",
        f"OOS IC: {metrics.get('oos_ic', 0):.3f}",
        f"Best Model: {metrics.get('best_model', 'N/A')}",
        f"Granger p: {metrics.get('granger_p', 0):.4f}"
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig
