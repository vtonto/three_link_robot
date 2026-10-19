import numpy as np
import pytest
from three_link_robot import RobotParameters
from three_link_robot.controllers import zero_torque
from three_link_robot.simulation import integrate, make_robot_rhs
from three_link_robot.trajectory import hold
from examples.demo_kinematics import preview_fk


def test_integrator_against_known_solution():
    times = np.linspace(0, 2, 51)
    result = integrate(lambda t, y: -2 * y, [1.0, 3.0], times)
    expected = np.exp(-2 * times[:, None]) * [1.0, 3.0]
    np.testing.assert_allclose(result.states, expected, rtol=2e-6)


@pytest.mark.parametrize("times", [[0], [0, 0], [1, 0], [0, float("nan")]])
def test_bad_time_grid(times):
    with pytest.raises(ValueError):
        integrate(lambda t, y: y, [1.0], times)


def test_bad_rhs_shape():
    with pytest.raises(ValueError, match="rhs"):
        integrate(lambda t, y: [0.0, 1.0], [1.0], [0, 1])


def test_parameter_validation():
    with pytest.raises(ValueError):
        RobotParameters(lengths=(1.0, -1.0, 1.0))
    with pytest.raises(ValueError):
        RobotParameters(com_offsets=(2.0, 0.4, 0.3))


def test_hold_returns_independent_arrays():
    trajectory = hold([0.1, 0.2, 0.3])
    trajectory(0).q[:] = 99
    np.testing.assert_allclose(trajectory(1).q, [0.1, 0.2, 0.3])


def test_robot_wiring_with_explicit_fake_dynamics():
    # 只测试连接与状态顺序；fake dynamics 不是机器人模型。
    def fake_dynamics(q, dq, tau, robot):
        np.testing.assert_array_equal(tau, np.zeros(3))
        return np.zeros(3)

    rhs = make_robot_rhs(RobotParameters(), zero_torque, hold([0, 0, 0]), dynamics=fake_dynamics)
    result = integrate(rhs, [0, 0, 0, 1, 2, 3], [0, 1])
    np.testing.assert_allclose(result.states[-1], [1, 2, 3, 1, 2, 3], atol=1e-12)


@pytest.mark.parametrize("q", [np.zeros(3), [0.3, -0.5, 0.8]])
def test_preview_preserves_link_lengths(q):
    robot = RobotParameters()
    points = preview_fk(q, robot)
    np.testing.assert_allclose(np.linalg.norm(np.diff(points, axis=0), axis=1), robot.lengths)
