"""阻抗与法向控力/切向控位练习；外围模型连接已完成。"""

import numpy as np

from .dynamics import mass_matrix, coriolis_forces, gravity_forces
from .kinematics import forward_kinematics, jacobian, jacobian_dot_times_dq
from .operational_space import check_task_jacobian, task_inertia
from .task_trajectory import vector


def impedance_wrench(x, dx, reference, stiffness, damping):
    """TODO INT-01: 返回 K(x_d-x)+D(dx_d-dx)，shape (2,)，单位 N。

    stiffness 单位 N/m，damping 单位 N s/m；两者为 x、y 两轴增益向量。
    不要在此加入环境力：阻抗通过与环境相互作用自然产生位移。
    """
    raise NotImplementedError("INT-01: implement impedance_wrench")


def impedance_control(t, q, dq, reference, robot, f_env, *,
                      stiffness=(60., 60.), damping=(20., 20.), min_sigma=1e-4):
    """已完成外围连接；环境力经动力学进入，不作为控制律输入。"""
    q, dq = vector(q, 3, "q"), vector(dq, 3, "dq")
    vector(f_env, 2, "f_env")
    K, D = vector(stiffness, 2, "stiffness"), vector(damping, 2, "damping")
    if np.any(K < 0) or np.any(D < 0):
        raise ValueError("impedance gains must be nonnegative")
    J = jacobian(q, robot)
    check_task_jacobian(J, min_sigma=min_sigma)
    x, dx = forward_kinematics(q, robot)[-1], J @ dq
    M = mass_matrix(q, robot)
    Lambda = task_inertia(J, M)
    h = coriolis_forces(q, dq, robot) + gravity_forces(q, robot)
    bias = jacobian_dot_times_dq(q, dq, robot)
    F_imp = impedance_wrench(x, dx, reference, K, D)
    tau = h + J.T @ (Lambda @ (reference.ddx - bias) + vector(F_imp, 2, "F_imp"))
    return vector(tau, 3, "tau")


def hybrid_force_wrench(force_desired, force_measured, y_desired, y, dy, z,
                        *, kp_force, ki_force, kp_y, kd_y):
    """TODO INT-02: 返回二维任务力 [u_n,u_y]，单位 N。

    e_f = force_desired-force_measured，z=积分 e_f dt（N s）。
    u_n = force_desired + kp_force*e_f + ki_force*z。
    u_y = kp_y*(y_desired-y) - kd_y*dy。
    法向正方向指向墙；环境作用于机器人的力方向相反。
    """
    raise NotImplementedError("INT-02: implement hybrid_force_wrench")


def hybrid_force_control(t, q, dq, reference, robot, f_env, z, *,
                         kp_force=0.5, ki_force=1.0, kp_y=80., kd_y=16.,
                         min_sigma=1e-4):
    """已完成外围连接；只在测得接触力时积分，返回 (tau, dz)。

    接触模型和传感器理想；失去接触时冻结积分，不假装能够控制法向力。
    """
    q, dq = vector(q, 3, "q"), vector(dq, 3, "dq")
    f_env = vector(f_env, 2, "f_env")
    gains = np.asarray([kp_force, ki_force, kp_y, kd_y], dtype=float)
    if not np.all(np.isfinite(gains)) or np.any(gains < 0):
        raise ValueError("force and tangential gains must be finite and nonnegative")
    if not np.isfinite(z):
        raise ValueError("z must be finite")
    J = jacobian(q, robot)
    check_task_jacobian(J, min_sigma=min_sigma)
    x, dx = forward_kinematics(q, robot)[-1], J @ dq
    measured = float(-f_env[0])
    if measured < 0 or f_env[1] != 0:
        raise ValueError("expected a frictionless wall force along negative x")
    wrench = hybrid_force_wrench(reference.force_n, measured, reference.y,
                                 x[1], dx[1], z, kp_force=kp_force,
                                 ki_force=ki_force, kp_y=kp_y, kd_y=kd_y)
    h = coriolis_forces(q, dq, robot) + gravity_forces(q, robot)
    tau = h + J.T @ vector(wrench, 2, "wrench")
    dz = reference.force_n - measured if measured > 0 else 0.0
    return vector(tau, 3, "tau"), dz
