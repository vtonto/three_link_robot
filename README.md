# three_link_robot

一个学习优先的平面三连杆机器人仿真工程。用 Python、NumPy、SciPy、Matplotlib 和 pytest，从几何关系与动力学方程逐步实现自己的机器人模型。

**当前阶段：工程框架与运动学预览。真实机器人动力学仿真尚未实现。** 未完成的核心函数抛出 `NotImplementedError`，不会悄悄返回零矩阵或虚构的物理结果。没有 MuJoCo、Pinocchio 或其他机器人算法库依赖。

## 在这台 Mac 上运行

在终端进入包含本文件的 `three_link_robot` 目录：

```bash
source .venv/bin/activate
python -m pytest -q -rs
python -m examples.smoke_test
python -m examples.demo_kinematics
```

最后一条命令无须图形窗口，会生成 `artifacts/kinematic_demo.png`：左侧是机械臂初始姿态和末端轨迹，右侧是预设关节角。

若希望在 Mac 桌面上播放动画：

```bash
python -m examples.demo_kinematics --show
```

关闭窗口即可结束。默认无窗口示例与 `--show` 使用相同数据；`--show` 需要可用的桌面图形后端。

当前 `.venv` 基于 Homebrew Python 3.12 创建，依赖安装在项目环境内，不修改 macOS 自带的系统 Python。虚拟环境不可直接复制到另一台电脑；换电脑或更换基础解释器后需要重建。

## 在新环境中安装

要求 Python **3.10 或更新版本**；推荐使用 3.12。本机已通过 Homebrew 安装 Python 3.12，新终端中的 `python3` 会优先使用该版本。macOS 自带的 `/usr/bin/python3` 仍保留为 3.9，请不要修改或删除它。重建环境时：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q -rs
```

`pyproject.toml` 统一定义依赖，`requirements.txt` 只是便捷安装入口。首次安装需要网络。可编辑安装意味着修改 `src/` 中的代码后无需重新安装。

## 文件地图

```text
three_link_robot/
├── src/three_link_robot/
│   ├── parameters.py          # 连杆长度、质量、质心、惯量、重力；已完成
│   ├── kinematics.py          # FK、Jacobian、IK；TODO
│   ├── dynamics.py            # M、c、g、逆/正向动力学；TODO
│   ├── controllers.py         # 零力矩已完成；PD、计算力矩控制 TODO
│   ├── trajectory.py          # 参考状态接口、静止目标；已完成
│   ├── simulation/
│   │   ├── integrator.py      # SciPy 通用积分封装；已完成
│   │   └── system.py          # 状态、控制器、动力学连接；已完成
│   └── visualization.py       # 坐标绘图、动画；已完成
├── examples/
│   ├── smoke_test.py          # 环境、导入和通用积分器检查
│   └── demo_kinematics.py     # 独立简单 FK 预览
├── tests/
│   ├── test_infrastructure.py # 可立即运行的外围测试
│   └── test_exercises.py      # 算法实现后自动启用的初步检查
├── artifacts/                # 示例输出，不纳入版本管理
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 统一约定

- 固定基座位于世界原点，机械臂在 **x-y 竖直平面**运动；x 向右、y 向上、z 指向屏幕外。
- 三个关节均绕 z 轴旋转。`q` 是相对上一根连杆的角度，逆时针为正。`q = [0, 0, 0]` 时所有连杆沿 +x 方向伸直。
- 使用 SI 单位：m、kg、s、rad、N·m。关节向量是 `(3,)`，不是 `(3, 1)`；暂不自动将角度折回 ±π。
- `inertias` 是各连杆**绕自身质心 z 轴**的转动惯量。默认值对应均匀细杆；如果修改长度或质量，需自己同步修改质心和惯量，参数对象不会自动推断。
- `gravity` 是非负重力加速度大小，方向为 -y；允许设置为 0。暂不建模摩擦、碰撞、关节限位。
- 动力学采用 `M(q) @ ddq + c(q, dq) + g(q) = tau`。`c` 是 `(3,)` 的速度相关力矩向量，不是 `(3, 3)` 的科氏矩阵。静止平衡需要 `tau = g(q)`。
- `forward_kinematics` 返回 `(4, 2)` 的基座及三个连杆端点坐标；`jacobian` 只描述末端 x、y，返回 `(2, 3)`。
- 仿真状态为 `[q1, q2, q3, dq1, dq2, dq3]`；仿真结果按时间放在行上，`states.shape == (N, 6)`。

## 建议的学习顺序

1. **FK**：从 `src/three_link_robot/kinematics.py` 的 TODO 01 开始；先验证零位，再添加自己选取的直角和折叠构型算例。
2. **Jacobian 与 IK**：完成 TODO 02–03；Jacobian 初步测试用有限差分交叉验证，IK 需要自行增加可达、不可达和奇异附近的算例。
3. **动力学**：完成 TODO 04–08；先推导，再实现。对称正定和静止时 `c=0` 只是必要检查，还应添加势能/重力一致性、正逆动力学一致性与无外力能量检查。
4. **闭环控制**：实现 TODO 09 的 PD，随后尝试重力补偿和 TODO 10 的计算力矩控制；之后再扩展轨迹与操作空间控制。

查看未完成入口：在编辑器中搜索 `TODO`。独立运行学习测试：

```bash
python -m pytest -m exercise -q -rs
```

当前 4 个学习测试会显示 **skipped** 和对应 TODO；这不代表算法通过验证。函数完成并不再抛 `NotImplementedError` 后，测试自动执行断言。请不要用零值临时消除异常，也不要将预览函数直接接入正式模型以绕过练习。

`examples/demo_kinematics.py` 中的 `preview_fk` 是唯一已完成的机器人几何计算，用于让第一天就有可观察的结果。它与正式学习入口分开。演示角度来自预设正弦序列，**没有力矩输入、重力响应或闭环控制**。

## 动力学完成后如何接起来

以下片段展示已准备好的连接方式；TODO 08 完成前会明确报错：

```python
import numpy as np
from three_link_robot import RobotParameters
from three_link_robot.controllers import zero_torque
from three_link_robot.trajectory import hold
from three_link_robot.simulation import integrate, make_robot_rhs

robot = RobotParameters()
rhs = make_robot_rhs(robot, zero_torque, hold([0, 0, 0]))
result = integrate(rhs, np.zeros(6), np.linspace(0, 2, 201))
q_history = result.states[:, :3]
```

这将是零力矩下的自由运动；有重力时不会保持初始姿态。`make_robot_rhs` 也接受 `dynamics=` 参数，未来可以替换同签名的动力学实现，不必改控制器。PD 完成后可用 `functools.partial(pd_control, kp=..., kd=...)` 绑定增益。

积分器采用 SciPy 的自适应 RK45。传入的时间网格只控制输出采样，不代表固定内部步长；控制器在每次求导时调用。学习离散采样控制时再扩展这一层。

第一阶段不需要 GitHub、远程服务器或 iPad 配置。后续 iPad 小实验可以独立整理成 Notebook，Mac 仍维护完整工程。
