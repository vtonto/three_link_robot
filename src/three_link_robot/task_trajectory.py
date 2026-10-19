"""末端世界坐标系二维位置参考；与关节参考独立。"""

from dataclasses import dataclass
import numpy as np


def vector(value, size, name):
    value = np.array(value, dtype=float, copy=True)
    if value.shape != (size,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite with shape ({size},)")
    return value


@dataclass
class TaskReference:
    x: np.ndarray    # (2,), m
    dx: np.ndarray   # (2,), m/s
    ddx: np.ndarray  # (2,), m/s²

    def __post_init__(self):
        for name in ("x", "dx", "ddx"):
            setattr(self, name, vector(getattr(self, name), 2, name))


def task_hold(x_target):
    target = vector(x_target, 2, "x_target")

    def trajectory(t):
        return TaskReference(target.copy(), np.zeros(2), np.zeros(2))

    return trajectory


def task_circle(center, *, radius=0.05, omega=0.5, phase=0.0):
    """圆形参考，omega 为 rad/s；轨迹可达性由实验者检查。"""
    center = vector(center, 2, "center")
    if not all(np.isfinite(v) for v in (radius, omega, phase)) or radius <= 0:
        raise ValueError("radius must be positive; radius, omega and phase must be finite")

    def trajectory(t):
        angle = omega * t + phase
        radial = np.array([np.cos(angle), np.sin(angle)])
        tangent = np.array([-np.sin(angle), np.cos(angle)])
        return TaskReference(center + radius * radial, radius * omega * tangent,
                             -radius * omega**2 * radial)

    return trajectory
