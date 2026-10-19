"""用同一固定目标比较三种关节控制器的闭环表现。"""

from functools import partial

import numpy as np

from three_link_robot import RobotParameters
from three_link_robot.controllers import (
    computed_torque_control,
    pd_control,
    pd_gravity_control,
)
from three_link_robot.simulation import integrate, make_robot_rhs
from three_link_robot.trajectory import hold


def main():
    target = np.array([0.5, -0.4, 0.3])
    times = np.linspace(0.0, 5.0, 501)
    initial_state = np.zeros(6)
    pd_kp = np.array([40.0, 30.0, 20.0])
    pd_kd = np.array([10.0, 8.0, 6.0])
    ct_kp = np.full(3, 25.0)
    ct_kd = np.full(3, 10.0)

    experiments = (
        ("PD (g=0)", RobotParameters(gravity=0.0),
         partial(pd_control, kp=pd_kp, kd=pd_kd)),
        ("PD (g=9.81)", RobotParameters(),
         partial(pd_control, kp=pd_kp, kd=pd_kd)),
        ("PD + gravity", RobotParameters(),
         partial(pd_gravity_control, kp=pd_kp, kd=pd_kd)),
        ("Computed torque", RobotParameters(),
         partial(computed_torque_control, kp=ct_kp, kd=ct_kd)),
    )

    print("Fixed target:", target)
    print("Initial state:", initial_state)
    print("Duration: 5 s; 501 output samples; RK45")
    print("PD gains: kp=[40, 30, 20], kd=[10, 8, 6]")
    print("Computed-torque gains: kp=[25, 25, 25], kd=[10, 10, 10]")
    for name, robot, controller in experiments:
        result = integrate(
            make_robot_rhs(robot, controller, hold(target)),
            initial_state.copy(),
            times,
        )
        error = target - result.states[:, :3]
        print(f"\n{name}")
        print("  final position error [rad]:", error[-1])
        print("  final joint velocity [rad/s]:", result.states[-1, 3:])
        print("  RMS position error [rad]:", np.sqrt(np.mean(error**2, axis=0)))


if __name__ == "__main__":
    main()

