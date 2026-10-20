"""阻抗、弹性墙接触与法向控力/切向控位三个渐进实验。

python -m examples.demo_interaction --task free|wall|force [--show]
"""

import argparse
from functools import partial
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from three_link_robot import RobotParameters
from three_link_robot.contact import (VerticalWall, contact_state,
                                      make_impedance_rhs, make_force_rhs)
from three_link_robot.interaction_controller import impedance_control, hybrid_force_control
from three_link_robot.interaction_trajectory import force_hold
from three_link_robot.kinematics import forward_kinematics
from three_link_robot.operational_space import TaskSingularityError
from three_link_robot.simulation import integrate
from three_link_robot.task_trajectory import task_hold


def make_experiment(task="free"):
    if task not in ("free", "wall", "force"):
        raise ValueError("task must be free, wall or force")
    robot = RobotParameters()
    q_start = np.array([0.5, -1.1, 0.8])
    start_xy = forward_kinematics(q_start, robot)[-1]
    if task == "free":
        wall = VerticalWall(start_xy[0] + 1.0, stiffness=150., damping=5.)
        reference = task_hold(start_xy + [0.02, 0.0])
        controller = partial(impedance_control, stiffness=(60., 60.), damping=(20., 20.))
        initial = np.r_[q_start, np.zeros(3)]
        rhs = make_impedance_rhs(robot, controller, reference, wall)
    elif task == "wall":
        wall = VerticalWall(start_xy[0] + 0.01, stiffness=150., damping=5.)
        reference = task_hold(start_xy + [0.04, 0.0])
        controller = partial(impedance_control, stiffness=(60., 60.), damping=(20., 20.))
        initial = np.r_[q_start, np.zeros(3)]
        rhs = make_impedance_rhs(robot, controller, reference, wall)
    else:
        wall = VerticalWall(start_xy[0] - 0.01, stiffness=150., damping=5.)
        reference = force_hold(2.0, start_xy[1])
        controller = partial(hybrid_force_control, kp_force=0.5, ki_force=1.0,
                             kp_y=80., kd_y=16.)
        initial = np.r_[q_start, np.zeros(3), 0.0]
        rhs = make_force_rhs(robot, controller, reference, wall)
    return robot, wall, initial, reference, rhs


def run_experiment(task="free", *, duration=5.0, samples=151):
    if not np.isfinite(duration) or duration <= 0 or samples < 2:
        raise ValueError("duration must be positive and samples >= 2")
    robot, wall, initial, reference, rhs = make_experiment(task)
    rhs(0.0, initial)  # 未填练习在积分前明确报错。
    result = integrate(rhs, initial, np.linspace(0.0, duration, samples))
    samples_at_wall = [contact_state(s[:3], s[3:6], robot, wall)
                       for s in result.states]
    xy = np.array([sample[0] for sample in samples_at_wall])
    force = np.array([max(0.0, float(-sample[2][0])) for sample in samples_at_wall])
    target_xy = None if task == "force" else np.array([reference(t).x for t in result.t])
    target_force = (np.array([reference(t).force_n for t in result.t])
                    if task == "force" else None)
    return {"task": task, "robot": robot, "wall": wall, "result": result,
            "xy": xy, "force": force, "target_xy": target_xy,
            "target_force": target_force, "reference": reference}


def plot_experiment(data):
    task, result = data["task"], data["result"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    t, xy, force = result.t, data["xy"], data["force"]
    axes[0, 0].plot(xy[:, 0], xy[:, 1], label="Actual endpoint")
    if task != "free":
        axes[0, 0].axvline(data["wall"].x, color="black", linestyle="--", label="Wall")
    if data["target_xy"] is not None:
        axes[0, 0].plot(data["target_xy"][:, 0], data["target_xy"][:, 1], ":",
                        label="Position reference")
    axes[0, 0].set(xlabel="x [m]", ylabel="y [m]", title=task)
    axes[0, 0].legend()
    axes[0, 1].plot(t, force, label="Measured normal force")
    if data["target_force"] is not None:
        axes[0, 1].plot(t, data["target_force"], "--", label="Desired force")
    axes[0, 1].set(xlabel="Time [s]", ylabel="Force on wall [N]")
    axes[0, 1].legend()
    axes[1, 0].plot(t, xy[:, 0] - data["wall"].x)
    axes[1, 0].set(xlabel="Time [s]", ylabel="x - wall [m]",
                   title="Positive means wall compression")
    axes[1, 1].plot(t, xy[:, 1], label="Actual y")
    if task == "force":
        y_ref = data["reference"](0.0).y
        axes[1, 1].axhline(y_ref, linestyle="--", label="Desired y")
    axes[1, 1].set(xlabel="Time [s]", ylabel="Tangential y [m]")
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.grid(alpha=.25)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("free", "wall", "force"), default="free")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    try:
        data = run_experiment(args.task)
    except (NotImplementedError, TaskSingularityError) as error:
        parser.exit(2, f"{error}\nSee docs/interaction_learning.md; no plot was generated.\n")
    figure = plot_experiment(data)
    output = Path("artifacts") / f"interaction_{args.task}.png"
    output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=150)
    print(f"Saved {output}; final measured force={data['force'][-1]:.3g} N")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
