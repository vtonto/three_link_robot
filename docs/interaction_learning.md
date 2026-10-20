# 阻抗控制与接触力控制：学习路线

本练习接在已经完成的平面 3R 运动学、RNEA、正向动力学和二维 OSC 之后。先使用一面**竖直、无摩擦、有限刚度**的墙，让末端沿世界坐标系 `+x` 方向接近；再练习机器人对墙的法向力以及沿墙的 `y` 位置。它是理想数值模型，不代表刚性接触求解器或真机控制。

## 你需要填写的两个函数

| 顺序 | 文件 / 函数 | 目标 |
| --- | --- | --- |
| INT-01 | `src/three_link_robot/interaction_controller.py / impedance_wrench` | 把二维位置和速度误差变成任务空间力 |
| INT-02 | 同文件 `hybrid_force_wrench` | 组合接触法向 PI 控力与切向 PD 控位 |

外围已完成：墙的力模型、把环境力加到关节动力学的状态方程、积分状态、三组实验、曲线与测试。未填函数会抛 `NotImplementedError`；对应测试会标记 `skipped`，不会给出伪造的闭环结果。请按顺序填函数，不必改 RNEA 或已有 OSC 实现。

## 1. 先弄清力的方向

墙面在 `x = x_wall`。当末端 `x > x_wall` 时，压入量为 `δ = x - x_wall > 0`。墙作用于机器人的力为

$$
f_{env}=\begin{bmatrix}-f_n\\0\end{bmatrix},\qquad
f_n=\max(0,k_w\delta+b_w\dot x),
$$

其中仅在 `δ > 0` 时计算上式；否则 `f_env = 0`。`f_n` 是**机器人施加到墙上的力的大小**，单位 N，永远非负。退离墙面时阻尼可能使弹簧与阻尼之和小于零，此时也取零，避免墙“拉住”末端。模型没有摩擦，所以 `y` 方向的环境力为零。

环境力进入机器人动力学时是外力，不是控制器力矩：

$$
M(q)\ddot q+h(q,\dot q)=\tau_{ctrl}+J(q)^Tf_{env},\qquad h=c+g.
$$

先运行外围测试，检查 `x` 在墙左边时无接触、压入时力指向 `-x`、以及外力经 `J.T` 转成关节广义力。接触边界是非光滑的，当前 RK45 只适合低刚度的学习实验；不要把它当成高刚度碰撞模拟。

## 2. INT-01：先做自由空间阻抗，再碰墙

`TaskReference` 已给出期望 $x_d,\dot x_d,\ddot x_d$。`impedance_wrench` 接收末端位置、速度以及两轴的刚度 $K$（N/m）和阻尼 $D$（N·s/m），返回二维任务力：

$$
F_{imp}=K(x_d-x)+D(\dot x_d-\dot x).
$$

`impedance_control` 的外围代码已经组装重力/速度偏置与 OSC 惯量：

$$
\tau_{ctrl}=h+J^T\{\Lambda(\ddot x_d-\dot J\dot q)+F_{imp}\}.
$$

代入含接触的动力学，可得到本练习**实际实现的阻抗**。令 $e=x-x_d$：

$$
\Lambda(q)\ddot e+D\dot e+Ke=f_{env}.
$$

因此环境施力后末端会让位。这里的表观惯量是随构型变化的 $\Lambda(q)$；本练习**没有**实现任意指定的常数 $M_d$ 惯量整形。不要把外力额外加进 `impedance_wrench`，它已经从被控对象动力学进入了等式。

先运行 `free`：墙在工作空间以外，末端应靠近目标、墙力始终为零。再运行 `wall`：目标在墙后方 3 cm，末端无法同时到达目标并保持零接触力。忽略稳态速度后，法向力可由两个弹簧平衡预测：

$$
K_x(x_d-x)=k_w(x-x_{wall}),\qquad
f_{n,\infty}=\frac{K_xk_w}{K_x+k_w}(x_d-x_{wall}).
$$

当前参数 $K_x=60$ N/m、$k_w=150$ N/m、$x_d-x_{wall}=0.03$ m，预测约 **1.286 N**。提高 $K_x$ 会使末端更少让位、接触力更大；降低 $K_x$ 则反之。先只改一个参数，观察曲线后再解释。

## 3. INT-02：接触时法向控力、切向控位

第三组实验从**已经轻微接触**的状态开始，省去“自由空间接近 → 识别接触 → 切换控制模式”的问题。设 $f_d\geq0$ 是期望施加到墙上的法向力，$f_m=-f_{env,x}\geq0$ 是理想测量值，$z=\int(f_d-f_m)dt$，单位 N·s。你填写的二维任务力应为

$$
u_n=f_d+K_{pf}(f_d-f_m)+K_{if}z,\qquad
u_y=K_{py}(y_d-y)-K_{dy}\dot y.
$$

外围代码将它映射成 $\tau_{ctrl}=h+J^T[u_n,u_y]^T$；`z` 是仿真状态的第七维。失去接触、测得 $f_m=0$ 时，本练习冻结积分，避免积分项无限积累；但还没有力矩饱和或完整抗积分饱和策略。法向正方向指向墙，切向目标沿 `y`。不要把机器人**受到**的 `f_env,x < 0` 直接当成期望的正法向力。

这是**有限刚度墙上的简化混合控制练习**。法向和切向任务在力的坐标上分开，但任务惯量可能有非对角耦合；它不是严格刚性约束下、用约束投影构造的精确混合位置/力控制。先验证恒定 2 N 的接触力和切向位置，再思考模型失配、传感器噪声与接触丢失时会发生什么。

## 运行与验收

在工程根目录执行：

```bash
python -m pytest tests/test_interaction.py -q -rs
python -m examples.demo_interaction --task free
python -m examples.demo_interaction --task wall
python -m examples.demo_interaction --task force
```

或打开 `notebooks/02_interaction_lab.ipynb`，按单元格观察墙、末端和参考值；核心公式仍需写在 `src/` 中。示例成功后会写出 `artifacts/interaction_<task>.png`。未完成时示例退出码为 2、不会输出图。

验收顺序：

1. 墙模型的力方向、无接触零力、`J.T @ f_env` 与正向动力学一致。
2. INT-01 的任务力符号和单位正确；外力存在时满足上面的阻抗方程。
3. `free` 末端接近目标、`wall` 的稳态力接近两个弹簧的预测值。
4. INT-02 的法向前馈 + PI 与切向 PD 正确；失去接触时积分冻结。
5. `force` 的法向力接近 2 N，沿墙 `y` 位置保持；同时检查整段轨迹是否经过奇异构型。

这些检查只验证指定参数下的理想模型，不构成接触稳定性或真机安全性证明。

## 延伸问题

- 把墙刚度加倍，阻抗实验的稳态力是否仍符合两个弹簧的公式？
- 令 `ki_force=0`，在重力模型故意偏差时观察力误差；再恢复积分。需要先明确模型误差从哪里注入，避免控制器与被控对象仍使用同一参数。
- 把 `f_d` 设为零或让末端脱离墙：法向力还能被“跟踪”吗？如何设计接触状态机？
- 若要指定常数表观质量 $M_d$，所需的加速度命令如何补偿真实 $\Lambda^{-1}f_{env}$？这应作为独立练习验证，不能直接把 $\Lambda$ 在公式中改成 $M_d$。

参考阅读：[Modern Robotics：力控制](https://modernrobotics.northwestern.edu/nu-gm-book-resource/11-5-force-control/)、[混合运动/力控制](https://modernrobotics.northwestern.edu/nu-gm-book-resource/11-6-hybrid-motion-force-control/)、[MIT OCW：Interaction Control](https://ocw.mit.edu/courses/2-141-modeling-and-simulation-of-dynamic-systems-fall-2006/4bd3caf3e58b45aa68ddc5ae239bbd8f_interaction_cont.pdf)。本仓库采用了为学习而简化的弹性墙模型，上述资料讨论的完整约束或真实交互控制范围更广。
