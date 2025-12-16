"""
Statistical analysis module.
"""

from .regime import (
    compute_regime_statistics,
    compute_cross_regime_statistics,
    test_regime_separation,
)
from .ic import (
    compute_rolling_ic,
    compute_ic_statistics,
    compute_hit_rate,
    compute_oos_metrics,
)
from .granger import (
    test_granger_causality,
    test_granger_both_directions,
    test_granger_on_differences,
)

__all__ = [
    'compute_regime_statistics',
    'compute_cross_regime_statistics',
    'test_regime_separation',
    'compute_rolling_ic',
    'compute_ic_statistics',
    'compute_hit_rate',
    'compute_oos_metrics',
    'test_granger_causality',
    'test_granger_both_directions',
    'test_granger_on_differences',
]
