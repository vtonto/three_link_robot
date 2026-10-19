"""外围测试立即执行；练习未完成明确 skip，错误实现会失败。"""

import numpy as np
import pytest
from three_link_robot import RobotParameters
from three_link_robot import kinematics as kin
from three_link_robot import operational_space as ops
from three_link_robot.dynamics import mass_matrix, forward_dynamics
from three_link_robot.osc_controller import osc_control
from three_link_robot.task_trajectory import TaskReference, task_hold, task_circle
from three_link_robot.simulation import make_robot_rhs

Q = np.array([.5, -1.1, .8])
DQ = np.array([.4, -.3, .2])


def implemented(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except NotImplementedError as error:
        pytest.skip(str(error))


def test_task_references_and_circle_derivatives():
    trajectory = task_circle([1., .3], radius=.08, omega=.7)
    t, step = .6, 1e-5
    ref = trajectory(t)
    plus, minus = trajectory(t+step), trajectory(t-step)
    np.testing.assert_allclose((plus.x-minus.x)/(2*step), ref.dx, atol=1e-9)
    np.testing.assert_allclose((plus.dx-minus.dx)/(2*step), ref.ddx, atol=1e-9)
    hold = task_hold([1., .5])
    hold(0).x[:] = 9
    np.testing.assert_allclose(hold(1).x, [1., .5])
    with pytest.raises(ValueError):
        TaskReference([1., 2., 3.], [0., 0.], [0., 0.])


def test_singularity_is_explicit():
    robot = RobotParameters()
    with pytest.raises(ops.TaskSingularityError):
        ops.check_task_jacobian(kin.jacobian(np.zeros(3), robot))
    assert ops.check_task_jacobian(kin.jacobian(Q, robot))[-1] > 1e-4


@pytest.mark.exercise
@pytest.mark.parametrize("dq", [np.zeros(3), DQ])
def test_jdot_dq_directional_derivative(dq):
    robot = RobotParameters()
    actual = implemented(kin.jacobian_dot_times_dq, Q, dq, robot)
    step = 1e-5
    expected = ((kin.jacobian(Q+step*dq, robot)-kin.jacobian(Q-step*dq, robot))
                /(2*step)) @ dq
    assert np.shape(actual) == (2,)
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-6)


@pytest.mark.exercise
def test_task_inertia_and_projector():
    robot = RobotParameters()
    J, M = kin.jacobian(Q, robot), mass_matrix(Q, robot)
    Lambda = implemented(ops.task_inertia, J, M)
    np.testing.assert_allclose(Lambda, Lambda.T, atol=1e-10)
    assert np.all(np.linalg.eigvalsh(Lambda) > 0)
    np.testing.assert_allclose(J @ np.linalg.solve(M, J.T) @ Lambda, np.eye(2), atol=1e-10)


@pytest.mark.exercise
def test_nullspace_torque_decoupling():
    robot = RobotParameters()
    J, M = kin.jacobian(Q, robot), mass_matrix(Q, robot)
    Lambda = implemented(ops.task_inertia, J, M)
    N = implemented(ops.nullspace_projector, J, M, Lambda)
    np.testing.assert_allclose(J @ N, np.zeros((2, 3)), atol=1e-10)
    np.testing.assert_allclose(N @ N, N, atol=1e-10)
    np.testing.assert_allclose(J @ np.linalg.solve(M, N.T), np.zeros((2, 3)), atol=1e-10)
    # This configuration has one redundant direction: catch an all-zero projector.
    assert np.linalg.matrix_rank(N, tol=1e-9) == 1


@pytest.mark.exercise
@pytest.mark.parametrize("with_posture", [False, True])
def test_controller_actual_task_acceleration(with_posture):
    robot = RobotParameters()
    J = kin.jacobian(Q, robot)
    x = kin.forward_kinematics(Q, robot)[-1]
    ref = TaskReference(x+[.02, -.01], [.1, -.05], [.03, .02])
    kp, kd = np.array([16., 25.]), np.array([8., 10.])
    posture = Q+[.2, -.1, .3] if with_posture else None
    tau = implemented(osc_control, 0., Q, DQ, ref, robot, kp=kp, kd=kd, q_posture=posture)
    ddq = forward_dynamics(Q, DQ, tau, robot)
    # Independent numerical Jdot, so matching errors in controller and helper cannot cancel.
    step = 1e-5
    bias = ((kin.jacobian(Q+step*DQ, robot)-kin.jacobian(Q-step*DQ, robot))/(2*step)) @ DQ
    expected = ref.ddx + kd*(ref.dx-J@DQ) + kp*(ref.x-x)
    np.testing.assert_allclose(J @ ddq + bias, expected, atol=1e-7)
    rhs = make_robot_rhs(robot, lambda *args: tau, lambda t: ref)
    np.testing.assert_allclose(rhs(0., np.r_[Q, DQ]), np.r_[DQ, ddq])
