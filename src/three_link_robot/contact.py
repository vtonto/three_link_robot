"""二维末端与竖直弹性墙的单边、无摩擦接触模型。"""

from dataclasses import dataclass
import numpy as np

from .dynamics import forward_dynamics
from .kinematics import forward_kinematics, jacobian
from .task_trajectory import vector


@dataclass(frozen=True)
class VerticalWall:
    x: float  # 墙面 x 坐标，机器人从左侧接近，m
    stiffness: float  # N/m
    damping: float = 0.0  # N s/m

    def __post_init__(self):
        for name in ("x", "stiffness", "damping"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.stiffness <= 0 or self.damping < 0:
            raise ValueError("wall stiffness must be positive and damping nonnegative")

    def penetration(self, xy):
        xy = vector(xy, 2, "xy")
        return max(0.0, float(xy[0] - self.x))

    def force_on_robot(self, xy, dxy):
        """返回环境作用于机器人末端的二维力；接触时 x 分量非正。"""
        xy, dxy = vector(xy, 2, "xy"), vector(dxy, 2, "dxy")
        depth = self.penetration(xy)
        if depth == 0.0:
            return np.zeros(2)
        normal = max(0.0, self.stiffness * depth + self.damping * dxy[0])
        return np.array([-normal, 0.0])

    def force_on_wall(self, xy, dxy):
        """机器人施加于墙的法向力大小，非负，单位 N。"""
        return max(0.0, float(-self.force_on_robot(xy, dxy)[0]))


def contact_state(q, dq, robot, wall):
    """由关节状态计算末端位置、速度和环境力。"""
    q, dq = vector(q, 3, "q"), vector(dq, 3, "dq")
    xy = forward_kinematics(q, robot)[-1]
    dxy = jacobian(q, robot) @ dq
    return xy, dxy, wall.force_on_robot(xy, dxy)


def make_impedance_rhs(robot, controller, trajectory, wall):
    """接触仿真的六维状态方程：M ddq + h = tau + J.T f_env。"""
    def rhs(t, state):
        state = vector(state, 6, "state")
        q, dq = state[:3], state[3:]
        _, _, f_env = contact_state(q, dq, robot, wall)
        tau = vector(controller(t, q, dq, trajectory(t), robot, f_env), 3, "tau")
        ddq = forward_dynamics(q, dq, tau + jacobian(q, robot).T @ f_env, robot)
        return np.r_[dq, ddq]

    return rhs


def make_force_rhs(robot, controller, trajectory, wall):
    """七维状态方程；第七维为法向力误差积分 z，单位 N s。"""
    def rhs(t, state):
        state = vector(state, 7, "state")
        q, dq, z = state[:3], state[3:6], state[6]
        _, _, f_env = contact_state(q, dq, robot, wall)
        tau, dz = controller(t, q, dq, trajectory(t), robot, f_env, z)
        tau = vector(tau, 3, "tau")
        dz = float(dz)
        if not np.isfinite(dz):
            raise ValueError("force-error integral derivative must be finite")
        ddq = forward_dynamics(q, dq, tau + jacobian(q, robot).T @ f_env, robot)
        return np.r_[dq, ddq, dz]

    return rhs
