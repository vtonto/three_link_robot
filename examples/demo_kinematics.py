"""预设关节运动的简单 FK 演示；不调用学习区，不计算力矩或动力学。"""

import argparse
from pathlib import Path
import numpy as np
from three_link_robot import RobotParameters


def preview_fk(q, robot):
    """仅供演示的二维几何 FK；正式学习入口在 kinematics.py。"""
    angles = np.cumsum(q)
    segments = np.asarray(robot.lengths)[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
    return np.vstack((np.zeros((1, 2)), np.cumsum(segments, axis=0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="另外打开交互动画窗口")
    parser.add_argument("--output", type=Path, default=Path("artifacts/kinematic_demo.png"))
    args = parser.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from three_link_robot.visualization import plot_robot, animate_robot

    robot = RobotParameters()
    times = np.linspace(0, 6, 201)
    q = np.column_stack((0.6 * np.sin(times), 0.8 * np.sin(times + 0.6), 0.5 * np.cos(times)))
    frames = np.array([preview_fk(angles, robot) for angles in q])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    plot_robot(frames[0], ax=axes[0], reach=sum(robot.lengths))
    axes[0].plot(frames[:, -1, 0], frames[:, -1, 1], "--", alpha=0.6, label="end-effector path")
    axes[0].set_title("Simple FK preview (prescribed motion)")
    axes[0].legend()
    axes[1].plot(times, q, label=["q1", "q2", "q3"])
    axes[1].set(xlabel="time [s]", ylabel="angle [rad]", title="Prescribed joint angles")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"Kinematic preview saved: {args.output.resolve()}")
    if args.show:
        figure, animation = animate_robot(frames, reach=sum(robot.lengths),
                                          interval_ms=1000 * (times[1] - times[0]))
        plt.show()


if __name__ == "__main__":
    main()

