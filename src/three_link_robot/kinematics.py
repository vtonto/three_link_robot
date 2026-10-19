import numpy as np
from .parameters import RobotParameters
from scipy.linalg import expm


def jacobian_dot_times_dq(q, dq, robot):
    """TODO OSC-01: 返回 Jdot(q,dq) @ dq，shape (2,), m/s²。

    提示：theta=cumsum(q)，omega=cumsum(dq)。对末端位置求两次导数，
    取 ddq=0 时的部分。它不是 J @ ddq，也不是完整末端加速度。
    用沿 dq 方向的 Jacobian 中心差分验证，不要通过积分估计。
    """
    raise NotImplementedError("OSC-01: implement jacobian_dot_times_dq in kinematics.py")

def forward_kinematics(q: np.ndarray, robot: RobotParameters) -> np.ndarray:
    """返回 (4, 2)：基座、关节2、关节3、末端的世界坐标，单位 m。"""
    l1, l2, l3 = robot.lengths
    q1, q2, q3 = q
    M4_home = np.eye(4)
    M4_home[:3, 3] = np.array([
        l1+l2+l3, 0, 0,
    ])

    M3_home = np.eye(4);
    M3_home[:3, 3] = np.array([
        l1+l2, 0, 0,
    ])

    M2_home = np.eye(4);
    M2_home[:3, 3] = np.array([
        l1, 0, 0,
    ])

    wi = np.array([0,0,1])
    r1 = np.array([0, 0, 0])
    r2 = np.array([l1, 0, 0])
    r3 = np.array([l2+l1,0,0])
    v1 = -np.cross(wi, r1)
    v2 = -np.cross(wi, r2)
    v3 = -np.cross(wi, r3)
    S1 = np.concatenate((wi,v1))
    S2 = np.concatenate((wi,v2))
    S3 = np.concatenate((wi,v3))
    Slist = np.column_stack((S1,S2,S3))

    E1 = expm(hat(S1)*q1)
    E2 = expm(hat(S2)*q2)
    E3 = expm(hat(S3)*q3)

    T4 = E1 @ E2 @ E3 @ M4_home
    T3 = E1 @ E2 @ M3_home
    T2 = E1 @ M2_home

    p1 = np.array([0,0])
    p2 = T2[:2,3]
    p3 = T3[:2,3]
    p4 = T4[:2,3]

    points = np.vstack((p1,p2,p3,p4))
    return points


def jacobian(q: np.ndarray, robot: RobotParameters) -> np.ndarray:
    """返回末端位置 Jacobian，shape (2, 3)，满足 v_xy = J @ dq。"""
    l1, l2, l3 = robot.lengths
    q1, q2, q3 = q
    M4_home = np.eye(4)
    M4_home[:3, 3] = np.array([
        l1+l2+l3, 0, 0,
    ])

    wi = np.array([0,0,1])
    r1 = np.array([0, 0, 0])
    r2 = np.array([l1, 0, 0])
    r3 = np.array([l2+l1,0,0])
    v1 = -np.cross(wi, r1)
    v2 = -np.cross(wi, r2)
    v3 = -np.cross(wi, r3)
    S1 = np.concatenate((wi,v1))
    S2 = np.concatenate((wi,v2))
    S3 = np.concatenate((wi,v3)) 
    E1 = expm(hat(S1)*q1)
    E2 = expm(hat(S2)*q2)
    E3 = expm(hat(S3)*q3)
    T4 = E1 @ E2 @ E3 @ M4_home
    p = T4[:3,3]

    Js1 = S1
    Js2 = AdT(E1) @ S2;
    Js3 = AdT(E1) @ AdT(E2) @ S3
    J = np.column_stack((Js1,Js2,Js3))
    Jw = J[:3,:3]
    Jv = J[3:,:3]
    Jp = Jv - skew(p) @ Jw
    Jxy = Jp[:2,:]
    return Jxy


def inverse_kinematics(target_xy: np.ndarray, q0: np.ndarray,
                       robot: RobotParameters, *, tolerance: float = 1e-6,
                       max_iterations: int = 100) -> np.ndarray:
    """返回 shape (3,) 的关节角；失败时抛 RuntimeError，不返回伪成功。"""
    target_xy = np.asarray(target_xy, dtype=float)
    q_iter = np.asarray(q0, dtype=float).copy()
    
    if target_xy.shape != (2,):
        raise ValueError("target_xy must have shape (2,)")
    if q_iter.shape != (3,):
        raise ValueError("q0 must have shape (3,)")
    
    lambda_j = 0.03
    alpha = 0.3
    
    for i in range(max_iterations):
        joints = forward_kinematics(q_iter, robot)
        p_xy = joints[-1,:]
        error = target_xy - p_xy
        if np.linalg.norm(error) < tolerance :
            return q_iter
        Jxy = jacobian(q_iter, robot)
        A = (Jxy @ Jxy.T + lambda_j**2 * np.eye(2))
        dq = Jxy.T @ np.linalg.solve(A, error)
        q_iter = q_iter + alpha * dq
        
    raise RuntimeError('inverse kinematics did not converge')

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
