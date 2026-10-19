"""学习区：M(q) ddq + c(q, dq) + g(q) = tau。

c 是速度相关力矩向量 (3,)，不是科氏矩阵 C。
无摩擦、碰撞或关节限位；所有力矩单位为 N m。
"""

import numpy as np
from .parameters import RobotParameters
from scipy.linalg import expm

def mass_matrix(q: np.ndarray, robot: RobotParameters) -> np.ndarray:
    """返回 (3, 3) 质量矩阵。"""
    # TODO 04: 自行推导、实现，并验证对称正定。
    zero = np.zeros(3)
    gravity = 0
    e1 = np.array([1,0,0])
    M1 = inverse_dynamics_rnea(q, zero, e1, robot, gravity=gravity)
    e2 = np.array([0,1,0])
    M2 = inverse_dynamics_rnea(q, zero, e2, robot, gravity=gravity)
    e3 = np.array([0,0,1])
    M3 = inverse_dynamics_rnea(q, zero, e3, robot, gravity=gravity)

    M = np.column_stack((M1,M2,M3))
    
    return M


def coriolis_forces(q: np.ndarray, dq: np.ndarray, robot: RobotParameters) -> np.ndarray:
    """返回 (3,) 科氏与离心力矩向量 c。"""
    # TODO 05
    zero = np.zeros(3)
    gravity = 0
    tau_c = inverse_dynamics_rnea(q, dq, zero, robot, gravity=gravity)
    return tau_c


def gravity_forces(q: np.ndarray, robot: RobotParameters) -> np.ndarray:
    """返回 (3,) 重力项 g；静止平衡时所需 tau = g(q)。"""
    # TODO 06: 注意坐标方向及动力学方程中 g 的符号。
    zero = np.zeros(3)
    gravity = robot.gravity
    tau_g = inverse_dynamics_rnea(q, zero, zero, robot, gravity=gravity)
    return tau_g


def inverse_dynamics(q: np.ndarray, dq: np.ndarray, ddq: np.ndarray,
                     robot: RobotParameters) -> np.ndarray:
    """根据 q、dq 和 ddq 返回 (3,) 关节力矩，单位 N m。"""
    # TODO 07: 使用 M(q) @ ddq + c(q, dq) + g(q) 计算关节力矩。
    gravity = robot.gravity
    tau = inverse_dynamics_rnea(q, dq, ddq, robot, gravity=gravity)
    return tau


def forward_dynamics(q: np.ndarray, dq: np.ndarray, tau: np.ndarray,
                     robot: RobotParameters) -> np.ndarray:
    """返回 (3,) 关节加速度，rad/s²。"""
    # TODO 08: 组装上述各项并求解动力学方程。
    M = mass_matrix(q, robot)
    c = coriolis_forces(q, dq, robot)
    g = gravity_forces(q, robot)
    ddq = np.linalg.solve(
        M,
        tau - c -g,
    )

    return ddq

def inverse_dynamics_rnea(q: np.ndarray, dq: np.ndarray, ddq: np.ndarray, robot: RobotParameters, *, gravity: float) -> np.ndarray:
    l1, l2, l3 = robot.lengths
    c1, c2, c3 = robot.com_offsets
    m1, m2, m3 = robot.masses
    I1, I2, I3 = robot.inertias
    q1, q2, q3 = q
    dq1, dq2, dq3 = dq
    ddq1, ddq2, ddq3 = ddq

    M01 = np.eye(4)
    M01[0,3] = c1
    M12 = np.eye(4)
    M12[0,3] = l1 - c1 + c2
    M23 = np.eye(4)
    M23[0,3] = l2 - c2 + c3
    M34 = np.eye(4)
    M34[0,3] = l3 - c3

    omega = np.array([0,0,1])
    r1 = np.array([0, 0, 0])
    r2 = np.array([l1, 0, 0])
    r3 = np.array([l2+l1,0,0])
    S1 = np.concatenate((
        omega, -np.cross(omega, r1)
    ))
    S2 = np.concatenate((
        omega, -np.cross(omega, r2)
    ))
    S3 = np.concatenate((
        omega, -np.cross(omega, r3)
    ))

    G1 = planar_spatial_inertia(m1, I1)
    G2 = planar_spatial_inertia(m2, I2)
    G3 = planar_spatial_inertia(m3, I3)

    g_vec = np.array([0,-gravity,0])

    M10 = invT(M01)
    M20 = invT(M01 @ M12)
    M30 = invT(M01 @ M12 @ M23)
    A1 = AdT(M10) @ S1
    A2 = AdT(M20) @ S2
    A3 = AdT(M30) @ S3

    T10 = expm(-hat(A1)*q1) @ invT(M01)
    T21 = expm(-hat(A2)*q2) @ invT(M12)
    T32 = expm(-hat(A3)*q3) @ invT(M23)
    T43 = invT(M34)
    
    V0 = np.zeros(6)
    dV0 = np.zeros(6)
    dV0[3:] = -g_vec

    V1 = AdT(T10) @ V0 + A1 * dq1
    V2 = AdT(T21) @ V1 + A2 * dq2
    V3 = AdT(T32) @ V2 + A3 * dq3

    dV1 = AdT(T10) @ dV0 + adV(V1) @ (A1 * dq1) + A1 * ddq1
    dV2 = AdT(T21) @ dV1 + adV(V2) @ (A2 * dq2) + A2 * ddq2
    dV3 = AdT(T32) @ dV2 + adV(V3) @ (A3 * dq3) + A3 * ddq3

    F4 = np.zeros(6)
    F3 = G3 @ dV3 - adV(V3).T @ (G3 @ V3) + AdT(T43).T @ F4
    F2 = G2 @ dV2 - adV(V2).T @ (G2 @ V2) + AdT(T32).T @ F3
    F1 = G1 @ dV1 - adV(V1).T @ (G1 @ V1) + AdT(T21).T @ F2

    tau = np.array([
        A1 @ F1,
        A2 @ F2,
        A3 @ F3,
    ])

    return tau
    
    
def planar_spatial_inertia(mass: float, inertia_zz: float):
    G = np.zeros((6,6))
    G[2,2] = inertia_zz
    G[3:,3:] = mass * np.eye(3)
    return G
    
def hat(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    w = S[:3]
    v = S[3:]
    w_hat = skew(w)
    se3_matrix = np.zeros((4,4))
    se3_matrix[:3, :3] = w_hat
    se3_matrix[:3, 3] = v
    return se3_matrix

def AdT(T: np.ndarray) -> np.ndarray:
    R = T[:3,:3]
    p = T[:3,3]
    AdT = np.zeros((6,6))
    AdT[:3,:3] = R
    AdT[3:,3:] = R
    AdT[3:,:3] = skew(p) @ R
    return AdT
    
def skew(w):
    wx, wy, wz = w
    return np.array([[0., -wz, wy], [wz, 0., -wx], [-wy, wx, 0.]])

def adV(V: np.ndarray) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    omg = V[:3]
    vel = V[3:]
    adV = np.zeros((6,6))
    adV[:3,:3] = skew(omg)
    adV[3:,:3] = skew(vel)
    adV[3:,3:] = skew(omg)
    return adV

def invT(T: np.ndarray) -> np.ndarray:
    R = T[:3,:3]
    p = T[:3,3]
    invT = np.eye(4)
    invT[:3,:3] = R.T
    invT[:3,3] = -R.T @ p
    return invT
