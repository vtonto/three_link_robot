"""通用数值积分外围代码，不包含机器人动力学算法。"""

from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class SimulationResult:
    t: np.ndarray  # (N,)
    states: np.ndarray  # (N, state_dimension)


def integrate(rhs: Callable, initial_state, times, *, rtol=1e-7, atol=1e-9) -> SimulationResult:
    """用 RK45 积分；times 是输出采样时刻，内部步长由 SciPy 自适应选择。"""
    t = np.asarray(times, dtype=float)
    y0 = np.asarray(initial_state, dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("times must contain at least two finite, strictly increasing values")
    if y0.ndim != 1 or y0.size == 0 or not np.all(np.isfinite(y0)):
        raise ValueError("initial_state must be a finite, nonempty vector")

    def checked_rhs(time, state):
        derivative = np.asarray(rhs(time, state), dtype=float)
        if derivative.shape != y0.shape or not np.all(np.isfinite(derivative)):
            raise ValueError("rhs must return a finite vector matching initial_state")
        return derivative

    result = solve_ivp(checked_rhs, (t[0], t[-1]), y0, t_eval=t, rtol=rtol, atol=atol)
    if not result.success:
        raise RuntimeError(f"Integration failed: {result.message}")
    return SimulationResult(result.t, result.y.T)
