"""计算力矩控制器跟踪正弦关节轨迹的演示。"""

from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from three_link_robot import RobotParameters
from three_link_robot.controllers import computed_torque_control
from three_link_robot.simulation import integrate, make_robot_rhs
from three_link_robot.trajectory import sinusoidal
from three_link_robot.kinematics import forward_kinematics
from three_link_robot.visualization import animate_robot


robot = RobotParameters()

trajectory = sinusoidal(
    q_center=np.array([0.3, -0.4, 0.2]),
    amplitude=np.array([0.2, 0.15, 0.1]),
    omega=np.array([1.0, 0.8, 1.2]),
)

controller = partial(
    computed_torque_control,
    kp=np.array([25.0, 25.0, 25.0]),
    kd=np.array([10.0, 10.0, 10.0]),
)

rhs = make_robot_rhs(robot, controller, trajectory)

times = np.linspace(0.0, 5.0, 501)
initial_reference = trajectory(times[0])
initial_state = np.concatenate((initial_reference.q, initial_reference.dq))

# 故意加入少量初始位置误差，用来观察控制器如何消除误差。
initial_state[:3] += np.array([0.1, -0.1, 0.05])

result = integrate(rhs, initial_state, times)

q = result.states[:, :3]
dq = result.states[:, 3:]
points_over_time = np.array([
    forward_kinematics(q_i, robot)
    for q_i in q
])

animation_figure, animation = animate_robot(
    points_over_time,
    reach=sum(robot.lengths),
    interval_ms=10,
)

animation_figure.axes[0].set_title(
    "Computed-torque trajectory tracking"
)

references = [trajectory(t) for t in result.t]
q_reference = np.array([reference.q for reference in references])
dq_reference = np.array([reference.dq for reference in references])
position_error = q_reference - q

print("最终实际关节角：", q[-1])
print("最终参考关节角：", q_reference[-1])
print("最终位置误差：", position_error[-1])
print("各关节最大绝对位置误差：", np.max(np.abs(position_error), axis=0))
print("最终速度误差：", dq_reference[-1] - dq[-1])

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 9))

for joint in range(3):
    axes[0].plot(result.t, q[:, joint], label=f"q{joint + 1} actual")
    axes[0].plot(
        result.t,
        q_reference[:, joint],
        "--",
        label=f"q{joint + 1} reference",
    )
    axes[1].plot(result.t, dq[:, joint], label=f"dq{joint + 1} actual")
    axes[1].plot(
        result.t,
        dq_reference[:, joint],
        "--",
        label=f"dq{joint + 1} reference",
    )
    axes[2].plot(result.t, position_error[:, joint], label=f"e{joint + 1}")

axes[0].set_ylabel("joint angle [rad]")
axes[1].set_ylabel("joint velocity [rad/s]")
axes[2].set_ylabel("position error [rad]")
axes[2].set_xlabel("time [s]")

for ax in axes:
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()
