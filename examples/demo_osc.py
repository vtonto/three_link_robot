"""三个渐进实验；未完成公式时明确报错，不生成伪结果。

python -m examples.demo_osc --task hold|circle|nullspace [--show]
"""

import argparse
from functools import partial
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from three_link_robot import RobotParameters
from three_link_robot.kinematics import forward_kinematics
from three_link_robot.operational_space import task_diagnostics, TaskSingularityError
from three_link_robot.osc_controller import osc_control
from three_link_robot.task_trajectory import task_hold, task_circle
from three_link_robot.simulation import make_robot_rhs, integrate


def make_experiment(task="hold", *, robot=None):
    """Notebook 和命令行共用同一实验；不硬编码本机路径。"""
    if task not in ("hold", "circle", "nullspace"):
        raise ValueError("task must be hold, circle or nullspace")
    robot = RobotParameters() if robot is None else robot
    q_start = np.array([0.5, -1.1, 0.8])
    start = forward_kinematics(q_start, robot)[-1]
    if task == "circle":
        trajectory = task_circle(start - [0.05, 0.0], radius=0.05, omega=0.5)
    else:
        trajectory = task_hold(start + ([0.02, 0.02] if task == "hold" else [0., 0.]))
    controller = partial(osc_control, q_posture=(q_start + [0.0, 0.0, 0.6])
                         if task == "nullspace" else None)
    return robot, np.r_[q_start, np.zeros(3)], trajectory, controller


def run_experiment(task="hold", *, duration=5.0, samples=151):
    if not np.isfinite(duration) or duration <= 0 or samples < 2:
        raise ValueError("duration must be positive and samples >= 2")
    robot, initial, trajectory, controller = make_experiment(task)
    rhs = make_robot_rhs(robot, controller, trajectory)
    # Probe before integration: missing exercise errors remain visible.
    rhs(0.0, initial)
    result = integrate(rhs, initial, np.linspace(0., duration, samples))
    points = np.array([forward_kinematics(q, robot) for q in result.states[:, :3]])
    target = np.array([trajectory(t).x for t in result.t])
    error = np.linalg.norm(target - points[:, -1], axis=1)
    sigma = np.array([task_diagnostics(q, robot)["singular_values"][-1]
                      for q in result.states[:, :3]])
    return {"robot": robot, "result": result, "points": points, "target": target,
            "error": error, "sigma_min": sigma, "task": task}


def plot_experiment(data):
    result, points, target = data["result"], data["points"], data["target"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(target[:, 0], target[:, 1], "--x", markevery=[0], label="Target")
    ax.plot(points[:, -1, 0], points[:, -1, 1], label="Actual")
    ax.plot(points[0, :, 0], points[0, :, 1], "o-", alpha=.35, label="Initial arm")
    ax.plot(points[-1, :, 0], points[-1, :, 1], "o-", alpha=.7, label="Final arm")
    ax.set(xlabel="x [m]", ylabel="y [m]", title=data["task"])
    ax.set_aspect("equal"); ax.legend()
    axes[0, 1].plot(result.t, data["error"])
    axes[0, 1].set(xlabel="Time [s]", ylabel="Position error norm [m]")
    for i in range(3):
        axes[1, 0].plot(result.t, result.states[:, i], label=f"q{i+1}")
    axes[1, 0].set(xlabel="Time [s]", ylabel="Joint angle [rad]")
    axes[1, 0].legend()
    axes[1, 1].plot(result.t, data["sigma_min"])
    axes[1, 1].set(xlabel="Time [s]", ylabel="Smallest singular value [m/rad]")
    for ax in axes.flat:
        ax.grid(alpha=.25)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("hold", "circle", "nullspace"), default="hold")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    try:
        data = run_experiment(args.task)
    except (NotImplementedError, TaskSingularityError) as error:
        parser.exit(2, f"{error}\nSee docs/osc_learning.md; no simulation output was generated.\n")
    figure = plot_experiment(data)
    output = Path("artifacts") / f"osc_{args.task}.png"
    output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=150)
    print(f"Saved {output}; final error={data['error'][-1]:.3g} m")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
