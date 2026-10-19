# 平面三连杆机器人：项目技术总结

更新日期：2026-10-18。本文对应当前代码与本次重新运行的实验；数值是仿真结果，不代表真机精度。

## 1. 目标、范围与约定

本项目以学习机器人学算法为目的，使用 Python、NumPy、SciPy、Matplotlib 和 pytest，从零实现平面三个转动关节（3R）的运动学、动力学与关节空间控制。MuJoCo 和 Pinocchio 没有参与核心计算。连杆在世界坐标系的 `xy` 平面运动，正关节角绕 `+z` 轴；实际重力沿世界 `-y` 方向。默认连杆长度为 `(1.0, 0.8, 0.6)` m，质量为 `(1.0, 0.8, 0.6)` kg，质心距各自近端关节为 `(0.5, 0.4, 0.3)` m，重力加速度大小为 `9.81 m/s²`。

动力学采用

\[
M(q)\ddot q+c(q,\dot q)+g(q)=\tau.
\]

这里 `c` 是三维科氏与离心**力矩向量**，不是一个 `3×3` 的科氏矩阵。`q`、`dq`、`ddq` 和 `tau` 都采用 NumPy 一维数组，形状 `(3,)`；仿真状态为 `[q1,q2,q3,dq1,dq2,dq3]`，形状 `(6,)`。

## 2. 代码架构和数据流

| 文件 | 职责与主要接口 |
| --- | --- |
| [`parameters.py`](../src/three_link_robot/parameters.py) | `RobotParameters` 保存连杆长度、质量、质心位置、绕质心 `z` 轴转动惯量和重力，并检查形状、有限性及参数范围。 |
| [`kinematics.py`](../src/three_link_robot/kinematics.py) | `forward_kinematics` 返回基座、两个中间关节和末端的 `(4,2)` 坐标；`jacobian` 返回末端位置的 `(2,3)` Jacobian；`inverse_kinematics` 用阻尼最小二乘迭代求解给定末端位置。 |
| [`dynamics.py`](../src/three_link_robot/dynamics.py) | 六维空间 RNEA、`M/c/g` 提取、逆动力学和正动力学；包含齐次变换、伴随矩阵、空间速度叉乘矩阵等辅助函数。 |
| [`trajectory.py`](../src/three_link_robot/trajectory.py) | `JointReference(q,dq,ddq)`、静止目标 `hold` 和三关节正弦参考 `sinusoidal`。 |
| [`controllers.py`](../src/three_link_robot/controllers.py) | 零力矩、关节 PD、PD 加重力补偿、计算力矩控制；统一接口为 `(t,q,dq,reference,robot) -> tau`。 |
| [`simulation/system.py`](../src/three_link_robot/simulation/system.py) | `make_robot_rhs` 将轨迹、控制器和正动力学连成 `dx/dt=[dq,ddq]`，并检查输入输出形状及有限性。 |
| [`simulation/integrator.py`](../src/three_link_robot/simulation/integrator.py) | `solve_ivp` 的 RK45 包装；输出指定时刻的状态，内部使用自适应步长。 |
| [`visualization.py`](../src/three_link_robot/visualization.py) | 静态绘图和基于 `(N,4,2)` 关节点序列的 Matplotlib 动画。 |
| [`tests/`](../tests) | 基础设施、运动学、动力学、控制器平衡点和轨迹导数的自动检查。 |
| [`examples/demo_pd_control.py`](../examples/demo_pd_control.py) | 当前实际演示的是计算力矩控制跟踪正弦轨迹，绘制位置、速度、误差并生成机械臂动画；文件名沿用早期 PD 演示。 |
| [`examples/compare_controllers.py`](../examples/compare_controllers.py) | 在固定目标和统一仿真条件下重现下文的四组控制器对比。 |

闭环调用关系是：`trajectory(t)` 产生 `q_d,dq_d,ddq_d`；控制器结合当前 `q,dq` 生成 `tau`；`forward_dynamics` 生成 `ddq`；积分器更新 `q,dq`。控制器由积分器每次评估右端函数时调用，因此当前模型是**连续反馈仿真**，不是固定采样周期的数字控制器。

## 3. RNEA 的实现

### 3.1 坐标系、轴和惯量

每根连杆的局部坐标系位于该连杆质心，零位时各局部 `x` 轴沿世界 `+x`。零位相邻坐标系的位移为

\[
M_{01,x}=c_1,\quad
M_{12,x}=l_1-c_1+c_2,\quad
M_{23,x}=l_2-c_2+c_3,\quad
M_{34,x}=l_3-c_3.
\]

`M34` 把第三连杆质心接到末端工具坐标系；目前末端外力 `F4=0`。空间速度约定为 `V=[ωx,ωy,ωz,vx,vy,vz]`。第 `i` 个关节的空间螺旋轴在零位世界坐标系中为 `S_i=[ω;-ω×r_i]`，其中 `ω=(0,0,1)`，`r_i` 分别为三个关节轴上的点。代码使用累积零位变换的逆伴随，将 `S_i` 变到各质心坐标系，得到 `A_i`。

质心坐标系的六维惯量 `G_i` 把绕 `z` 轴的转动惯量放在旋转分块的 `G[2,2]`，把质量放在线速度分块 `G[3:,3:]`。这正对应 `[ω;v]` 的排序；该构造只用于此平面 3R 模型，不是任意三维刚体的完整惯量描述。

### 3.2 前向速度与加速度递推

代码通过 `expm(-hat(A_i)*q_i)` 和零位相邻变换构造 `T10,T21,T32`。`AdT(T)` 返回运动向量的六维伴随变换，`adV(V)` 返回运动向量的叉乘算子。以前一坐标系的量表示递推关系：

\[
V_i=\operatorname{Ad}(T_{i,i-1})V_{i-1}+A_i\dot q_i,
\]

\[
\dot V_i=\operatorname{Ad}(T_{i,i-1})\dot V_{i-1}
+\operatorname{ad}(V_i)A_i\dot q_i+A_i\ddot q_i.
\]

实际重力向量设为 `g_vec=(0,-gravity,0)`，基座的空间加速度初始化为 `dV0[3:] = -g_vec`。这个“负重力”是 RNEA 的伪加速度约定，使静止时递推出保持姿态所需的正向抗重力力矩。

### 3.3 反向力与关节力矩递推

从第三连杆向第一连杆传播空间力：

\[
F_i=G_i\dot V_i-\operatorname{ad}(V_i)^\mathsf T G_iV_i
+\operatorname{Ad}(T_{i+1,i})^\mathsf T F_{i+1},
\qquad \tau_i=A_i^\mathsf T F_i.
\]

与运动向量不同，传播力时用伴随矩阵的**转置**。代码从 `F4=zeros(6)` 出发，求得 `F3,F2,F1`，最后组成 `(3,)` 的 `tau`。不建通用符号模型，所有矩阵和递推均由 NumPy/SciPy 直接计算。

### 3.4 从 RNEA 提取动力学各项

`inverse_dynamics_rnea(q,dq,ddq,robot,gravity=...)` 是统一计算核心：

\[
M_{:,j}=\operatorname{RNEA}(q,0,e_j,0),\quad
c(q,\dot q)=\operatorname{RNEA}(q,\dot q,0,0),\quad
g(q)=\operatorname{RNEA}(q,0,0,g_0).
\]

对三个单位加速度各运行一次 RNEA，并用 `np.column_stack` 组装质量矩阵。`inverse_dynamics` 用真实重力直接调用 RNEA；`forward_dynamics` 使用 `np.linalg.solve(M, tau-c-g)`，不显式求逆矩阵。

## 4. 动力学验证：方法与本次结果

在项目虚拟环境中运行 `python -m pytest -q`，本次结果为 **20 passed**。自动测试包括：零位 FK、Jacobian 与 FK 中心差分的一致性、两个姿态下质量矩阵对称正定、静止时 `c=0`、正逆动力学回环、水平伸直时手算重力力矩、两种含模型补偿控制器的目标点静态平衡，以及正弦轨迹的速度和加速度数值导数检查。积分器、参数检查和模块连接也有测试。

为了避免只用 RNEA 验证 RNEA，本次还在 `q=(0.3,-0.6,0.4)`、`dq=(0.7,-0.4,0.2)` 做了**独立数值交叉验证**。对每个连杆质心直接求位置 Jacobian `Jv_i` 和角速度 Jacobian `Jw_i`，构造

\[
M_{\rm ref}=\sum_i\left(m_iJ_{v_i}^{\mathsf T}J_{v_i}
+I_{zz,i}J_{w_i}^{\mathsf T}J_{w_i}\right),\qquad
g_{\rm ref}=\sum_i m_i g_0\,\frac{\partial y_{c_i}}{\partial q}.
\]

再对这个独立构造的 `M_ref` 做中心差分，利用 Christoffel 公式得到 `c_ref`。结果如下：

| 检查 | 本次结果 |
| --- | ---: |
| `max(abs(M-M.T))` | `2.22e-16` |
| `eigvalsh(M)` | `[0.02299882, 0.18028851, 5.05927142]`，全为正 |
| `max(abs(M-M_ref))` | `5.00e-16` |
| `max(abs(g-g_ref))` | `3.55e-15` |
| `max(abs(c-c_ref))` | `1.35e-10`，包含差分误差 |
| 零位静态重力力矩 | `[28.2528, 9.6138, 1.7658]` N·m，与手算值一致 |
| `max(abs(RNEA-(M@ddq+c+g)))` | `7.11e-15` |
| 正逆动力学回环的最大加速度误差 | `8.77e-15` rad/s² |

最后两项是很有用的**内部一致性**检查，但各项来自同一个动力学实现，不能单独证明模型符合真实机械臂；上面的质心 Jacobian、势能梯度和数值微分比较更独立。独立交叉验证是本次审阅时运行的计算，尚未纳入项目 pytest 测试集。

## 5. 三种控制器及测试结果

误差统一定义为 `e=q_d-q`、`de=dq_d-dq`。固定目标比较采用 `q_d=(0.5,-0.4,0.3)` rad，初始 `q=dq=0`，仿真 `0–5 s`，输出 `501` 个等间隔时刻，内部为自适应 RK45。普通 PD 和 PD 加重力补偿使用 `kp=(40,30,20)`、`kd=(10,8,6)`；计算力矩控制使用 `kp=(25,25,25)`、`kd=(10,10,10)`。有重力时 `g0=9.81 m/s²`。复现命令：`.venv/bin/python -m examples.compare_controllers`。

| 控制器 | 力矩公式 | `t=5 s` 的位置误差 `q_d-q`，rad | `t=5 s` 的速度 `dq`，rad/s |
| --- | --- | --- | --- |
| 普通 PD，有重力 | `kp*e + kd*de` | `[0.641135, 0.237924, 0.075532]` | `[-0.004178, -0.001333, -0.000767]` |
| PD 加重力补偿 | `kp*e + kd*de + g(q)` | `[0.003133, 0.001723, 0.000532]` | `[0.000544, 0.000087, -0.000096]` |
| 计算力矩控制 | `v=ddq_d+kd*de+kp*e`; `tau=RNEA(q,dq,v)` | `[1.99e-10, -1.59e-10, 1.20e-10]` | `[9.56e-10, -7.65e-10, 5.74e-10]` |

补充基线：把普通 PD 的重力设为零后，其 5 秒位置误差同 PD 加精确重力补偿，约为 `[0.003133,0.001723,0.000532]` rad。这说明在当前理想模型里，重力项被精确抵消。对普通 PD，有重力静止平衡满足 `kp*e_ss=g(q_ss)`，因此需要非零位置误差维持抗重力力矩。PD 加重力补偿的理论平衡误差为零；表中的毫弧度量级是**5 秒时的剩余瞬态误差**，不能当作稳态误差。计算力矩控制令理想系统 `ddq=v`，误差满足 `d²e/dt²+kd*de/dt+kp*e=0`；接近机器精度的结果依赖控制器与仿真对象使用同一动力学模型。

另外，[正弦轨迹演示](../examples/demo_pd_control.py) 以初始位置偏差 `(0.1,-0.1,0.05)` rad 启动计算力矩控制。5 秒后位置误差为 `[3.01e-10,1.87e-10,2.40e-12]` rad；最大绝对误差恰为初始偏差的大小。动画逐帧用实际 `q(t)` 调用 FK，因此展示的是动力学闭环生成的姿态序列。

三类增益不能直接按数字比较：普通 PD 的 `kp/kd` 把位置和速度误差变成力矩，而计算力矩控制中的 `kp/kd` 把误差变成目标加速度。本表展示的是给定参数下的行为，不是公平调参后的控制器排行榜。

## 6. 开发中遇到的问题与修复

| 问题 | 原因、影响与处理 |
| --- | --- |
| MATLAB 写法混入 Python | 列表中的分号、`np.zeros(6,6)`、函数注解中的分号、4×4 变换的第 `4` 列索引都造成语法或运行错误；改为逗号、`np.zeros((6,6))`、`:` 和平移列索引 `3`。 |
| 六维向量维度混用 | `(6,)` 与 `(6,1)` 相加会触发 NumPy 广播，可能悄悄变为 `(6,6)`；空间速度、加速度、力统一采用 `(6,)`。 |
| 空间惯量索引和反向递推 | `[ω;v]` 排序要求 `Izz` 在 `G[2,2]`；反向传播力需要 `Ad(T).T`；第二连杆应使用 `dV2`。这些改动是 RNEA 结果正确的关键。 |
| 重力符号 | 实际重力为世界 `-y`，RNEA 基座加速度需初始化为其相反数；用水平零位的静止抗重力力矩核对符号。 |
| Python 调用与拼写 | `gravity` 被声明为仅限关键字参数，调用时必须写 `gravity=...`；质量矩阵列拼接应为 `np.column_stack((M1,M2,M3))`。 |
| 导入与缩进导致测试无法收集 | `kinematics.py` 的 `RobotParameters` 导入曾被注释、`JointReference` 曾从未导出的包顶层导入、`controllers.py` 首行曾多一个空格；修正后 pytest 才能进入实际测试。 |
| PD 速度误差笔误 | 曾将 `reference.dq` 赋为当前 `dq`，把负速度反馈变成错误的正反馈；现为 `reference.dq-dq`。现有早期测试未覆盖此控制器，因而错误由代码检查发现。 |
| 演示代码顺序 | 正弦轨迹曾在定义前调用、初始轨迹状态曾被之后的 `zeros(6)` 覆盖；现先构造轨迹，再构造初态，并保留 `FuncAnimation` 引用。 |

## 7. 适用范围、现存缺口与运行方法

模型目前没有摩擦、关节限位、力矩饱和、传感器噪声、控制周期、外部接触或参数辨识。六维递推用于平面三连杆，不能据此声称已实现通用三维多体动力学。控制器和被控对象共用完全相同的参数；理想轨迹误差极小不构成实际机器人鲁棒性的证据。IK 目前采用固定阻尼和步长；可达边界、奇异邻域、关节限位及失败模式仍需更系统的测试。当前代码中还有已完成函数的旧 `TODO` 注释，README 的阶段说明也仍停留在初始工程壳子；本次检查时项目目录尚不是 Git 仓库。

在项目根目录执行：

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m examples.compare_controllers
.venv/bin/python -m examples.demo_pd_control
```

前两项分别运行自动测试和命令行数值对比；最后一项会打开曲线及动画窗口，需要图形桌面环境。接下来的工程验证优先做控制器与被控对象的参数失配、力矩限幅及离散采样实验，再考虑接入真实硬件或更完整的仿真平台。

