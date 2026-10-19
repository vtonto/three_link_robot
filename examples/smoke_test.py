"""验证依赖、包导入与积分器；不是机器人动力学正确性测试。"""

import numpy as np
from three_link_robot import RobotParameters
from three_link_robot.simulation import integrate


def main():
    robot = RobotParameters()
    result = integrate(lambda t, y: -y, [1.0], np.linspace(0, 1, 11))
    np.testing.assert_allclose(result.states[-1, 0], np.exp(-1), rtol=1e-6)
    print(f"Smoke test OK: {len(robot.lengths)} links; integrator matches exp(-t).")


if __name__ == "__main__":
    main()

