# OSC 学习框架

固定基座平面 3R，只控制末端世界坐标系二维位置。没有环境接触、关节限位、力矩饱和或离散控制周期。

## 当前完成与待完成

外围包括二维参考轨迹、控制器连接、奇异构型检查、三组实验、曲线输出和 Notebook。
以下四个练习目前已由学习者实现并通过测试；保留顺序、公式和验收方法供复习。若重新留空练习，`NotImplementedError` 对应的 skipped 只表示未完成。

1. **OSC-01**：`kinematics.py` 中 `jacobian_dot_times_dq`。
2. **OSC-02**：`operational_space.py` 中 `task_inertia`。
3. **OSC-03**：同文件 `task_torque`。完成前三项后运行定点和圆轨迹实验。
4. **OSC-04**：同文件 `nullspace_projector`。完成后运行零空间实验。

保留原有动力学和关节控制器，`simulation/system.py` 无需修改，它本来就能传递任意参考对象。

## 公式约定

$$
\ddot x=J\ddot q+\dot J\dot q,\quad h=c+g,\quad
a_{cmd}=\ddot x_d+K_d(\dot x_d-J\dot q)+K_p(x_d-x).
$$

$$
\Lambda=(JM^{-1}J^T)^{-1},\quad
\bar J=M^{-1}J^T\Lambda,\quad N=I-\bar JJ.
$$

$$
\tau=h+J^T\Lambda(a_{cmd}-\dot J\dot q)+N^T\tau_0,\quad
\tau_0=K_{p0}(q_0-q)-K_{d0}\dot q.
$$

此版本补偿**完整关节偏置 h**。之前的任务方向补偿写法使用
$\eta=\Lambda JM^{-1}h-\Lambda\dot J\dot q$，两种写法的主任务加速度相同，
但完整补偿额外包含 $N^Th$，因此冗余运动不同，不能重复添加 h。
姿态参考不一定与主任务兼容，不能要求三个关节同时收敛到任意 q0。

可展开提示：

<details><summary>OSC-01：如何直接求 Jdot dq？</summary>

令 theta=cumsum(q)、omega=cumsum(dq)，则

$$
\dot J\dot q=-\sum_{i=1}^3 l_i\omega_i^2
\begin{bmatrix}\cos\theta_i\\\sin\theta_i\end{bmatrix}.
$$

这里用到的是每根连杆的绝对角速度，而不是单个关节相对角速度。
</details>

<details><summary>OSC-02/04：如何避免显式矩阵求逆？</summary>

先计算 `X = np.linalg.solve(M, J.T)`，再解 `(J @ X) @ Lambda = I2`。
广义逆为 `X @ Lambda`；速度投影 N 通常不对称，力矩用 N.T。
</details>

## 运行与验证

工程根目录运行：

```bash
python -m pip install -e '.[dev,notebook]'
python -m pytest -q -rs
python -m pytest tests/test_osc.py -q -rs
python -m examples.demo_osc --task hold
python -m examples.demo_osc --task circle
python -m examples.demo_osc --task nullspace --show
```

若练习函数未填写，实验以退出码 2 提示对应 TODO，不会输出轨迹图。成功时默认图保存在 artifacts；Notebook 直接内嵌显示。
初始构型为 `[0.5,-1.1,0.8]` rad，避免伸直奇异点；圆半径 0.05 m。
默认最小奇异值保护阈值为 1e-4 m/rad。它只检测当前评估构型，不保证全轨迹可达或不越过奇异点。
目前没有阻尼广义逆；加入阻尼后不能继续声称严格的零空间解耦。

验证顺序：Jdot 的方向差分 → Lambda 对称正定与逆关系 → 真实正向动力学计算的末端加速度 → 零空间投影 → 闭环轨迹。
测试覆盖未填公式会跳过；错误数值、维度或投影会失败。单次轨迹成功不构成全局稳定性证明。

## iPad / Jupyter

```bash
# Mac 工程根目录；默认仅本机访问
python -m jupyterlab
```

打开 `notebooks/01_osc_lab.ipynb`。Notebook 查找当前目录的工程根，不使用 Mac 绝对路径。
核心算法仍在 src/ 中；Notebook 启用 autoreload，保存模块后下一次运行会重新加载。
从上到下运行，未填公式时显示待完成提示并跳过仿真，不能把“Notebook 无异常”理解成 OSC 已实现。

iPad 可以通过浏览器访问 Mac 上经过认证的 Jupyter 服务；Mac 须保持在线、唤醒且网络可达。
同一可信局域网可绑定 Mac 的局域网地址，例如 `python -m jupyterlab --ip=<Mac局域网IP> --no-browser`，保留默认 token 认证。
外网访问请使用受保护的私有网络连接，不直接把 8888 端口暴露到公网。
这时文件保存在 Mac 上，iPad 是编辑与显示端。

也可以将仓库检出到云端 Jupyter 后安装同样的依赖。云端是独立副本，修改需通过 Git 同步；
当前 Notebook 未在真实 iPad、Safari 或 Colab 中实测。无需 GPU。

## 框架交付时的验证记录

原有 20 项测试通过；含新增外围测试的学习版本为 22 passed、6 skipped。
临时在内存注入参考公式（不写入练习文件）后为 28 passed，并运行了三组 5 秒闭环实验、检查曲线。
这些数字是当时的外围框架验证记录；四个 OSC 练习后来已实现并通过现有测试。

当前验证环境禁止 Jupyter 内核的 TCP/IPC 通信，因此未完成独立内核测试。
Notebook 的全部代码单元已使用进程内 IPython 按顺序执行，并检查初始构型图；
在 Mac 安装 notebook 依赖后仍需用 JupyterLab 执行一次“Restart Kernel and Run All”。
