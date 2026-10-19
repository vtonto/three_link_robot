"""关节空间参考轨迹。"""

from dataclasses import dataclass
import numpy as np


@dataclass
class JointReference:
    q: np.ndarray  # (3,), rad
    dq: np.ndarray  # (3,), rad/s
    ddq: np.ndarray  # (3,), rad/s²


def hold(q_target):
    """返回 trajectory(t) -> JointReference；每次调用返回独立数组。"""
    target = np.array(q_target, dtype=float, copy=True)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("q_target must be three finite joint angles")

    def trajectory(t: float) -> JointReference:
        return JointReference(target.copy(), np.zeros(3), np.zeros(3))

    return trajectory


def sinusoidal(q_center, amplitude, omega):
    """生成正弦关节轨迹；omega 单位为 rad/s。"""
    q_center = np.asarray(q_center, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)
    omega = np.asarray(omega, dtype=float)

    for name, value in (
        ("q_center", q_center),
        ("amplitude", amplitude),
        ("omega", omega),
    ):
        if value.shape != (3,):
            raise ValueError(f"{name} must have shape (3,)")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain finite values")

    if np.any(omega < 0):
        raise ValueError("omega must be nonnegative")

    def trajectory(t: float) -> JointReference:
        phase = omega * t

        q = q_center + amplitude * np.sin(phase)
        dq = amplitude * omega * np.cos(phase)
        ddq = -amplitude * omega**2 * np.sin(phase)

        return JointReference(q, dq, ddq)

    return trajectory
