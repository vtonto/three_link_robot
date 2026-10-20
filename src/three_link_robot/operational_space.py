"""固定基座、无接触、二维位置任务的 OSC 学习入口。

偏置补偿采用完整 h=c+g：tau=h+J.T@Lambda@(a_cmd-Jdot_dq)+N.T@tau0。
这与仅在任务方向补偿 h 的写法在零空间行为上不同，不要混用。
"""

import numpy as np
from .kinematics import jacobian


class TaskSingularityError(ValueError):
    """当前构型不适合使用未阻尼的满行秩 OSC。"""


def check_task_jacobian(J, *, min_sigma=1e-4):
    """返回奇异值；阈值单位依本项目位置 Jacobian，为 m/rad。

    这是实验保护阈值，不是通用的奇异性容差或轨迹安全保证。
    不自动退回阻尼逆，以免把近似解耦误当成精确解耦。
    """
    J = np.asarray(J, dtype=float)
    if J.shape != (2, 3) or not np.all(np.isfinite(J)):
        raise ValueError("J must be finite with shape (2, 3)")
    if not np.isfinite(min_sigma) or min_sigma <= 0:
        raise ValueError("min_sigma must be positive and finite")
    sigma = np.linalg.svd(J, compute_uv=False)
    if sigma[-1] <= min_sigma:
        raise TaskSingularityError(
            f"sigma_min={sigma[-1]:.3g} <= {min_sigma:g}; use a bent configuration "
            "or study a separately validated damped controller")
    return sigma


def task_inertia(J, M):
    """TODO OSC-02: 返回 Lambda=(J M^-1 J.T)^-1，shape (2,2)。

    用 solve(M,J.T) 和 solve(A,eye(2))，不要显式 inv(M)。
    调用者先检查 J 满行秩；M 来自已经验证的机器人模型。
    """
    A = J @ np.linalg.solve(M,J.T)
    Lambda = np.linalg.solve(A, np.eye(2))
    return Lambda


def task_torque(J, Lambda, h, a_cmd, jdot_dq):
    """TODO OSC-03: 返回 h + J.T @ Lambda @ (a_cmd-jdot_dq)，shape (3,)。

    h 是关节空间的 c+g，不可直接与二维任务力相加。
    这里的力矩已经包含完整 h；外层不要再添加一次重力补偿。
    """
    tau = h + J.T @ Lambda @ (a_cmd - jdot_dq)
    return tau


def nullspace_projector(J, M, Lambda):
    """TODO OSC-04: Jbar=solve(M,J.T)@Lambda；返回 N=I-Jbar@J。

    N 投影关节速度；力矩应使用 N.T@tau0。验证 J@N=0、
    J@solve(M,N.T)=0。N 通常不对称，不能省略转置。
    """
    Jbar = np.linalg.solve(M, J.T) @ Lambda
    N = np.eye(3) - Jbar @ J
    return N


def task_diagnostics(q, robot):
    """只依赖已有运动学，TODO 未填写也可观察构型。"""
    J = jacobian(q, robot)
    sigma = np.linalg.svd(J, compute_uv=False)
    return {"singular_values": sigma, "rank": int(np.linalg.matrix_rank(J)),
            "manipulability": float(np.prod(sigma))}
