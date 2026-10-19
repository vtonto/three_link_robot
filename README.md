# three_link_robot

学习优先的固定基座平面 3R 机器人项目，使用 Python、NumPy、SciPy、Matplotlib 和 pytest。

已实现 FK、二维位置 Jacobian、阻尼最小二乘 IK、六维 RNEA、M/c/g 提取、正向动力学、关节 PD、重力补偿与计算力矩控制。
验证范围和局限见 [技术总结](docs/technical_summary.md)。

**新增 OSC 学习框架：外围连接与实验入口已完成，四个核心公式留给学习者填写。**
未完成公式显式抛出 `NotImplementedError`，不会返回零值或生成虚构的闭环结果。

## 安装和运行

Python >=3.10，建议 3.12。在包含本文件的工程根目录运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q -rs
python -m examples.compare_controllers
python -m examples.demo_pd_control
```

最后一个脚本演示计算力矩控制，需桌面绘图环境。虚拟环境不能直接从 Mac 复制到其他平台。
当前积分器为自适应 RK45、连续反馈；输出时间网格不是控制采样周期。

## 操作空间控制练习

完整顺序、公式、提示、验证标准和 iPad 说明见 [OSC 学习指南](docs/osc_learning.md)。

| 练习 | 文件 / 函数 | 完成后能做什么 |
| --- | --- | --- |
| OSC-01 | `kinematics.py / jacobian_dot_times_dq` | 计算任务加速度中的速度偏置 |
| OSC-02 | `operational_space.py / task_inertia` | 计算二维任务空间惯量 |
| OSC-03 | `operational_space.py / task_torque` | 运行末端定点、圆轨迹实验 |
| OSC-04 | `operational_space.py / nullspace_projector` | 运行零空间姿态实验 |

```bash
python -m pytest tests/test_osc.py -q -rs
python -m examples.demo_osc --task hold
python -m examples.demo_osc --task circle
python -m examples.demo_osc --task nullspace --show
```

TODO 未完成时测试明确 skipped，实验报错退出；skipped 不等于验证通过。
实验默认无窗口输出到 `artifacts/osc_<task>.png`。不要从完全伸直的零构型开始未阻尼 OSC。

## Jupyter / iPad

```bash
python -m pip install -e '.[dev,notebook]'
python -m jupyterlab
```

打开 [01_osc_lab.ipynb](notebooks/01_osc_lab.ipynb)。公式未填时可以观察初始构型、参考轨迹和奇异值，
填完后可以直接运行闭环并查看内嵌曲线与可选 HTML 动画。核心实现仍保存在 src/，无需复制到单元格。

iPad 可作为浏览器客户端连接经过认证的 Mac/云端 Jupyter，Python 由服务器执行；
Mac 需要保持唤醒和网络可达。此项目尚未在真实 iPad 上实测，连接配置见 OSC 学习指南。

## 文件地图

- `src/three_link_robot/kinematics.py`：FK、Jacobian、IK、新增 Jdot dq 练习。
- `src/three_link_robot/dynamics.py`：RNEA、M/c/g、正逆动力学。
- `src/three_link_robot/controllers.py`：已有的关节空间控制器。
- `src/three_link_robot/trajectory.py`：关节参考；`task_trajectory.py`：新增二维末端参考。
- `src/three_link_robot/operational_space.py`：OSC 数学练习、奇异构型检查。
- `src/three_link_robot/osc_controller.py`：任务反馈、模型补偿和零空间的外围连接。
- `src/three_link_robot/simulation/`：通用积分器和机器人闭环连接。
- `examples/demo_osc.py`、`notebooks/01_osc_lab.ipynb`：共用实验函数的两种入口。
- `tests/test_osc.py`：轨迹导数、奇异性、任务加速度与零空间解耦验证。

## 统一约定与范围

世界 xy 竖直平面，重力沿 -y，关节绕 +z 逆时针为正；零位沿 +x 完全伸直。
关节向量 `(3,)`，二维任务向量 `(2,)`，Jacobian `(2,3)`，状态 `[q,dq]` 为 `(6,)`。
采用 SI 单位与 `M(q) @ ddq + c(q,dq) + g(q) = tau`；c 是力矩向量。

当前不含摩擦、碰撞、关节限位、力矩饱和、模型失配或离散控制周期。
OSC 使用完整 h=c+g 补偿，零空间力矩用 N.T 投影；精确解耦依赖精确模型和 Jacobian 满行秩。
这不是通用三维多体引擎，也不代表真机鲁棒性或全局稳定性已得到验证。
