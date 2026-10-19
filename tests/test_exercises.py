"""TODO 未完成时显式 skip；实现后自动执行这些初步检查。

通过这些检查只是起点，不等于算法已完整验证。请继续添加自己推导的算例。
"""

import numpy as np
import pytest
from three_link_robot import RobotParameters
from three_link_robot.kinematics import forward_kinematics, jacobian
from three_link_robot.dynamics import mass_matrix, coriolis_forces, gravity_forces
from three_link_robot.dynamics import inverse_dynamics, forward_dynamics
from three_link_robot.controllers import computed_torque_control, pd_gravity_control
from three_link_robot.trajectory import JointReference, sinusoidal

pytestmark = pytest.mark.exercise


def implemented(function, *args):
    try:
        return function(*args)
    except NotImplementedError as error:
        pytest.skip(str(error))


def test_fk_zero_configuration():
    robot = RobotParameters()
    actual = implemented(forward_kinematics, np.zeros(3), robot)
    expected = np.column_stack(([0, *np.cumsum(robot.lengths)], np.zeros(4)))
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_jacobian_matches_finite_difference():
    robot = RobotParameters()
    q = np.array([0.3, -0.6, 0.4])
    actual = implemented(jacobian, q, robot)
    h = 1e-6
    columns = []
    for direction in np.eye(3):
        plus = implemented(forward_kinematics, q + h * direction, robot)[-1]
        minus = implemented(forward_kinematics, q - h * direction, robot)[-1]
        columns.append((plus - minus) / (2 * h))
    np.testing.assert_allclose(actual, np.column_stack(columns), atol=1e-7, rtol=1e-5)


def test_mass_matrix_is_symmetric_positive_definite():
    for q in (np.zeros(3), np.array([0.3, -0.6, 0.4])):
        matrix = implemented(mass_matrix, q, RobotParameters())
        assert matrix.shape == (3, 3)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-10)
        assert np.all(np.linalg.eigvalsh(matrix) > 0)


def test_coriolis_at_rest_is_zero():
    value = implemented(coriolis_forces, np.array([0.3, -0.6, 0.4]), np.zeros(3), RobotParameters())
    assert value.shape == (3,)
    np.testing.assert_allclose(value, np.zeros(3), atol=1e-12)



def test_forward_inverse_dynamics_roundtrip():
    robot = RobotParameters()

    q = np.array([0.3, -0.6, 0.4])
    dq = np.array([0.7, -0.4, 0.2])
    expected_ddq = np.array([0.5, -0.3, 0.8])

    tau = inverse_dynamics(q, dq, expected_ddq, robot)
    actual_ddq = forward_dynamics(q, dq, tau, robot)

    np.testing.assert_allclose(
        actual_ddq,
        expected_ddq,
        atol=1e-10,
    )

def test_gravity_at_zero_configuration():
    robot = RobotParameters()
    l1, l2, _ = robot.lengths
    c1, c2, c3 = robot.com_offsets
    m1, m2, m3 = robot.masses
    g = robot.gravity

    expected = g * np.array([
        m1 * c1
        + m2 * (l1 + c2)
        + m3 * (l1 + l2 + c3),

        m2 * c2
        + m3 * (l2 + c3),

        m3 * c3,
    ])

    actual = gravity_forces(np.zeros(3), robot)

    np.testing.assert_allclose(actual, expected, atol=1e-10)

def test_pd_gravity_control_at_target():
    robot = RobotParameters()

    q_target = np.array([0.5, -0.4, 0.3])
    dq = np.zeros(3)

    reference = JointReference(
        q=q_target.copy(),
        dq=np.zeros(3),
        ddq=np.zeros(3),
    )

    tau = pd_gravity_control(
        0.0,
        q_target,
        dq,
        reference,
        robot,
        kp=np.array([40.0, 30.0, 20.0]),
        kd=np.array([10.0, 8.0, 6.0]),
    )

    # 已经位于目标位置时，PD 部分为零，
    # 控制器只需输出重力补偿力矩。
    np.testing.assert_allclose(
        tau,
        gravity_forces(q_target, robot),
        atol=1e-12,
    )

    # 该力矩应该让机器人保持静止。
    ddq = forward_dynamics(
        q_target,
        dq,
        tau,
        robot,
    )

    np.testing.assert_allclose(
        ddq,
        np.zeros(3),
        atol=1e-10,
    )

def test_computed_torque_at_target():
    robot = RobotParameters()

    q = np.array([0.5, -0.4, 0.3])
    dq = np.zeros(3)

    reference = JointReference(
        q=q.copy(),
        dq=np.zeros(3),
        ddq=np.zeros(3),
    )

    tau = computed_torque_control(
        0.0,
        q,
        dq,
        reference,
        robot,
        kp=np.array([25.0, 25.0, 25.0]),
        kd=np.array([10.0, 10.0, 10.0]),
    )

    np.testing.assert_allclose(
        tau,
        gravity_forces(q, robot),
        atol=1e-10,
    )

    ddq = forward_dynamics(q, dq, tau, robot)

    np.testing.assert_allclose(
        ddq,
        np.zeros(3),
        atol=1e-10,
    )

def test_sinusoidal_trajectory_derivatives():
    trajectory = sinusoidal(
        q_center=np.array([0.3, -0.4, 0.2]),
        amplitude=np.array([0.2, 0.15, 0.1]),
        omega=np.array([1.0, 0.8, 1.2]),
    )

    t = 0.7
    h = 1e-6

    reference = trajectory(t)
    plus = trajectory(t + h)
    minus = trajectory(t - h)

    numerical_dq = (plus.q - minus.q) / (2 * h)
    numerical_ddq = (plus.dq - minus.dq) / (2 * h)

    np.testing.assert_allclose(
        reference.dq,
        numerical_dq,
        rtol=1e-6,
        atol=1e-8,
    )

    np.testing.assert_allclose(
        reference.ddq,
        numerical_ddq,
        rtol=1e-6,
        atol=1e-8,
    )

