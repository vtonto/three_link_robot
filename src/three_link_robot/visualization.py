"""只接收坐标，不依赖具体 FK 或动力学实现。"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_robot(points, *, ax=None, reach=None):
    points = np.asarray(points, dtype=float)
    if points.shape != (4, 2) or not np.all(np.isfinite(points)):
        raise ValueError("points must have shape (4, 2) and be finite")
    if ax is None:
        _, ax = plt.subplots()
    line, = ax.plot(points[:, 0], points[:, 1], "o-", linewidth=3)
    limit = reach if reach is not None else max(1.0, np.linalg.norm(points, axis=1).max())
    ax.set(xlim=(-1.1 * limit, 1.1 * limit), ylim=(-1.1 * limit, 1.1 * limit),
           xlabel="x [m]", ylabel="y [m]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    return ax, line


def animate_robot(points_over_time, *, reach, interval_ms=30):
    frames = np.asarray(points_over_time, dtype=float)
    if frames.ndim != 3 or frames.shape[1:] != (4, 2) or len(frames) == 0 or not np.all(np.isfinite(frames)):
        raise ValueError("frames must have finite shape (N, 4, 2), N > 0")
    ax, line = plot_robot(frames[0], reach=reach)
    ax.set_title("Kinematic preview (prescribed motion)")

    def update(index):
        line.set_data(frames[index, :, 0], frames[index, :, 1])
        return (line,)

    animation = FuncAnimation(ax.figure, update, frames=len(frames), interval=interval_ms, blit=True)
    return ax.figure, animation  # 调用者需保留 animation 的引用
