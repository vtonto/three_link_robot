"""OSC 外围连接已完成；核心四个公式位于相应练习入口。"""

import numpy as np
from . import operational_space as ops
from . import kinematics as kin
from .dynamics import mass_matrix, coriolis_forces, gravity_forces
from .task_trajectory import vector


def osc_control(t, q, dq, reference, robot, *, kp=(25., 25.), kd=(10., 10.),
                q_posture=None, kp_posture=(2., 2., 2.), kd_posture=(1., 1., 1.),
                min_sigma=1e-4):
    """与现有 make_robot_rhs 兼容；返回 (3,) 关节力矩。

    kp/kd 将末端误差变成加速度；姿态 kp/kd 将关节误差变成力矩。
    q_posture=None 时不调用 OSC-04；否则使用 N.T 投影姿态 PD。
    精确解耦仅适用于精确模型、满行秩、无饱和的当前模型。
    """
    q, dq = vector(q, 3, "q"), vector(dq, 3, "dq")
    kp, kd = vector(kp, 2, "kp"), vector(kd, 2, "kd")
    if np.any(kp < 0) or np.any(kd < 0):
        raise ValueError("task gains must be nonnegative")
    J = kin.jacobian(q, robot)
    ops.check_task_jacobian(J, min_sigma=min_sigma)
    x = kin.forward_kinematics(q, robot)[-1]
    a_cmd = reference.ddx + kd * (reference.dx - J @ dq) + kp * (reference.x - x)
    jdot_dq = kin.jacobian_dot_times_dq(q, dq, robot)
    M = mass_matrix(q, robot)
    h = coriolis_forces(q, dq, robot) + gravity_forces(q, robot)
    Lambda = ops.task_inertia(J, M)
    tau = ops.task_torque(J, Lambda, h, a_cmd, jdot_dq)
    if q_posture is not None:
        q0 = vector(q_posture, 3, "q_posture")
        kp0 = vector(kp_posture, 3, "kp_posture")
        kd0 = vector(kd_posture, 3, "kd_posture")
        if np.any(kp0 < 0) or np.any(kd0 < 0):
            raise ValueError("posture gains must be nonnegative")
        N = ops.nullspace_projector(J, M, Lambda)
        tau = tau + N.T @ (kp0 * (q0 - q) - kd0 * dq)
    return vector(tau, 3, "tau")
