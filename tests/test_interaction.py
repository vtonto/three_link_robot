"""接触外围立即验证；控制公式练习未完成时明确跳过。"""

import numpy as np
import pytest

from three_link_robot import RobotParameters
from three_link_robot.contact import (VerticalWall, contact_state,
                                      make_impedance_rhs, make_force_rhs)
from three_link_robot.dynamics import forward_dynamics, mass_matrix
from three_link_robot.interaction_controller import (impedance_wrench, impedance_control,
                                                     hybrid_force_wrench, hybrid_force_control)
from three_link_robot.interaction_trajectory import ForceReference, force_hold
from three_link_robot.kinematics import forward_kinematics, jacobian, jacobian_dot_times_dq
from three_link_robot.operational_space import task_inertia
from three_link_robot.task_trajectory import TaskReference, task_hold

Q = np.array([0.5, -1.1, 0.8])
DQ = np.array([0.4, -0.3, 0.2])


def implemented(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except NotImplementedError as error:
        pytest.skip(str(error))


def test_wall_unilateral_force_and_sign():
    wall = VerticalWall(2.0, stiffness=150.0, damping=5.0)
    np.testing.assert_array_equal(wall.force_on_robot([1.99, 0.0], [1.0, 0.0]), [0., 0.])
    np.testing.assert_allclose(wall.force_on_robot([2.01, 0.0], [0., 0.]), [-1.5, 0.])
    np.testing.assert_allclose(wall.force_on_robot([2.01, 0.0], [0.1, 0.]), [-2., 0.])
    np.testing.assert_array_equal(wall.force_on_robot([2.01, 0.0], [-1., 0.]), [0., 0.])
    assert wall.force_on_wall([2.01, 0.0], [0., 0.]) == pytest.approx(1.5)
    with pytest.raises(ValueError):
        VerticalWall(2.0, stiffness=-1.0)
    with pytest.raises(ValueError):
        wall.force_on_robot([2.01, 0.0, 0.0], [0., 0.])


def test_contact_rhs_adds_external_generalized_force():
    robot = RobotParameters()
    x = forward_kinematics(Q, robot)[-1]
    wall = VerticalWall(x[0] - .01, stiffness=150.)
    f_env = contact_state(Q, np.zeros(3), robot, wall)[-1]
    tau = np.array([1., 2., 3.])
    rhs = make_impedance_rhs(robot, lambda *args: tau, task_hold(x), wall)
    actual = rhs(0., np.r_[Q, np.zeros(3)])
    expected = forward_dynamics(Q, np.zeros(3), tau + jacobian(Q, robot).T @ f_env, robot)
    np.testing.assert_allclose(actual, np.r_[np.zeros(3), expected])
    force_rhs = make_force_rhs(robot, lambda *args: (tau, .4), force_hold(2., x[1]), wall)
    np.testing.assert_allclose(force_rhs(0., np.r_[Q, np.zeros(3), 0.]),
                               np.r_[np.zeros(3), expected, .4])
    with pytest.raises(ValueError):
        ForceReference(-1., x[1])


@pytest.mark.exercise
def test_impedance_wrench_direction_and_units():
    ref = TaskReference([1.02, .1], [0., 0.], [0., 0.])
    actual = implemented(impedance_wrench, np.array([1., .12]), np.array([.1, -.2]),
                         ref, np.array([60., 80.]), np.array([20., 10.]))
    assert np.shape(actual) == (2,)
    np.testing.assert_allclose(actual, [-.8, .4])


@pytest.mark.exercise
def test_impedance_equation_under_contact():
    robot = RobotParameters()
    x = forward_kinematics(Q, robot)[-1]
    wall = VerticalWall(x[0] - .01, stiffness=150.)
    xy, dxy, f_env = contact_state(Q, DQ, robot, wall)
    ref = TaskReference(x + [.02, -.01], [.1, 0.], [.03, -.02])
    K, D = np.array([60., 80.]), np.array([20., 10.])
    tau = implemented(impedance_control, 0., Q, DQ, ref, robot, f_env,
                      stiffness=K, damping=D)
    J, M = jacobian(Q, robot), mass_matrix(Q, robot)
    Lambda = task_inertia(J, M)
    ddq = forward_dynamics(Q, DQ, tau + J.T @ f_env, robot)
    lhs = Lambda @ (J @ ddq + jacobian_dot_times_dq(Q, DQ, robot) - ref.ddx)
    lhs += D * (dxy - ref.dx) + K * (xy - ref.x)
    np.testing.assert_allclose(lhs, f_env, atol=1e-8)


@pytest.mark.exercise
def test_hybrid_force_wrench_and_integral_gate():
    actual = implemented(hybrid_force_wrench, 2.0, 1.5, .2, .21, .01, .3,
                         kp_force=.5, ki_force=1., kp_y=80., kd_y=16.)
    assert np.shape(actual) == (2,)
    np.testing.assert_allclose(actual, [2.55, -.96])
    robot = RobotParameters()
    x = forward_kinematics(Q, robot)[-1]
    ref = ForceReference(2., x[1])
    force = np.array([-1.5, 0.])
    tau, dz = implemented(hybrid_force_control, 0., Q, np.zeros(3), ref,
                          robot, force, .3)
    expected = hybrid_force_wrench(2., 1.5, x[1], x[1], 0., .3,
                                   kp_force=.5, ki_force=1., kp_y=80., kd_y=16.)
    from three_link_robot.dynamics import coriolis_forces, gravity_forces
    h = coriolis_forces(Q, np.zeros(3), robot) + gravity_forces(Q, robot)
    np.testing.assert_allclose(tau, h + jacobian(Q, robot).T @ expected)
    assert dz == pytest.approx(.5)
    _, dz_free = hybrid_force_control(0., Q, np.zeros(3), ref, robot, np.zeros(2), .3)
    assert dz_free == 0.


@pytest.mark.exercise
def test_impedance_free_space_reaches_target_without_contact():
    from examples.demo_interaction import run_experiment
    data = implemented(run_experiment, "free", duration=2.0, samples=61)
    np.testing.assert_allclose(data["force"], np.zeros_like(data["force"]))
    assert np.linalg.norm(data["xy"][-1] - data["target_xy"][-1]) < .002


@pytest.mark.exercise
def test_impedance_wall_static_force_prediction():
    from examples.demo_interaction import run_experiment
    data = implemented(run_experiment, "wall", duration=2.0, samples=61)
    # K=60 N/m, wall k=150 N/m, desired position is 3 cm beyond the wall.
    expected = 60. * 150. * .03 / (60. + 150.)
    assert data["force"][-1] == pytest.approx(expected, abs=.1)
    assert data["xy"][-1, 0] > data["wall"].x


@pytest.mark.exercise
def test_hybrid_force_reaches_target_and_holds_tangent():
    from examples.demo_interaction import run_experiment
    data = implemented(run_experiment, "force", duration=2.0, samples=61)
    assert data["force"][-1] == pytest.approx(2.0, abs=.1)
    assert abs(data["xy"][-1, 1] - data["reference"](2.0).y) < .005
