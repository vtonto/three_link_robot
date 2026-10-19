"""状态/控制器/动力学的连接代码。核心 TODO 未完成时直接传播异常。"""

import numpy as np
from ..dynamics import forward_dynamics


def make_robot_rhs(robot, controller, trajectory, *, dynamics=forward_dynamics):
    """返回 rhs(t, state)，state = [q1,q2,q3,dq1,dq2,dq3]。

    可注入相同签名的 dynamics(q,dq,tau,robot)，便于将来交叉验证。
    控制器在积分器每次求导时求值（连续反馈），不是离散采样控制。
    """
    def rhs(t, state):
        state = np.asarray(state, dtype=float)
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            raise ValueError("robot state must be a finite vector of shape (6,)")
        q, dq = state[:3], state[3:]
        tau = np.asarray(controller(t, q, dq, trajectory(t), robot), dtype=float)
        if tau.shape != (3,) or not np.all(np.isfinite(tau)):
            raise ValueError("controller must return three finite torques")
        ddq = np.asarray(dynamics(q, dq, tau, robot), dtype=float)
        if ddq.shape != (3,) or not np.all(np.isfinite(ddq)):
            raise ValueError("dynamics must return three finite accelerations")
        return np.concatenate((dq, ddq))

    return rhs

