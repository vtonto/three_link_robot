"""控制器统一签名 (t, q, dq, reference, robot) -> tau (3,)。"""

import numpy as np
from .parameters import RobotParameters
from .trajectory import JointReference
from .dynamics import gravity_forces, inverse_dynamics

def zero_torque(t: float, q: np.ndarray, dq: np.ndarray,
                reference: JointReference, robot: RobotParameters) -> np.ndarray:
    """仅用于未来自由运动实验；零力矩不代表机器人静止。"""
    return np.zeros(3)


def pd_control(t: float, q: np.ndarray, dq: np.ndarray,
               reference: JointReference, robot: RobotParameters, *, kp, kd) -> np.ndarray:
    # TODO 09: 自行实现关节 PD 控制；kp、kd 约定为 shape (3,)。
    kp = np.asarray(kp, dtype=float)
    kd = np.asarray(kd, dtype=float)

    if kp.shape != (3,) or kd.shape != (3,):
        raise ValueError("kp and kd must have shape (3,)")
        
    pos_err = reference.q - q
    vel_err = reference.dq - dq
    tau = kp * pos_err + kd * vel_err
    
    return tau

def pd_gravity_control(
    t: float,
    q: np.ndarray,
    dq: np.ndarray,
    reference: JointReference,
    robot: RobotParameters,
    *,
    kp,
    kd,
) -> np.ndarray:
    kp = np.asarray(kp, dtype=float)
    kd = np.asarray(kd, dtype=float)

    if kp.shape != (3,) or kd.shape != (3,):
        raise ValueError("kp and kd must have shape (3,)")

    position_error = reference.q - q
    velocity_error = reference.dq - dq

    tau_pd = (
        kp * position_error
        + kd * velocity_error
    )

    tau_gravity = gravity_forces(q, robot)

    return tau_pd + tau_gravity


def computed_torque_control(t: float, q: np.ndarray, dq: np.ndarray,
                            reference: JointReference, robot: RobotParameters,
                            *, kp, kd) -> np.ndarray:
    # TODO 10: 动力学与 PD 通过验证之后再实现。
    pos_err = reference.q - q
    vel_err = reference.dq - dq
    ddq = reference.ddq + kp * pos_err + kd * vel_err
    tau = inverse_dynamics(q, dq, ddq, robot)
    return tau

