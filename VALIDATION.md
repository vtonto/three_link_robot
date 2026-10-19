# 第一阶段验证记录

验证日期：2026-10-13（Asia/Shanghai），本机项目 `.venv`。

| 环境 | 版本 |
| --- | --- |
| Python | 3.12.14 |
| NumPy | 2.5.4 |
| SciPy | 1.18.1 |
| Matplotlib | 3.11.2 |
| pytest | 9.1.1 |

实际执行结果：

- `python -m pytest -q -rs`：**11 passed, 4 skipped**。跳过的是 FK、Jacobian、质量矩阵和科氏项的学习测试；原因均为对应算法尚未实现。
- `python -m examples.smoke_test`：通过。通用积分器对已知指数衰减解的数值结果符合容差。
- `python -m examples.demo_kinematics`：成功生成 `artifacts/kinematic_demo.png`。已查看图像，坐标、关节角曲线和标签正常显示。
- `python -m pip check`：未发现依赖冲突。

绘图验证时将 `MPLCONFIGDIR` 设为项目内的 `.mplconfig`，让字体缓存写入可写目录。

演示是预设关节运动的几何预览，尚未验证真实机器人动力学、闭环控制或桌面交互动画。学习函数仍保留 TODO；上述测试通过不意味着这些算法已实现。

安装过程中 SciPy 的官方源下载遇到中断，最终使用清华镜像下载同一 wheel，并与 PyPI 官方元数据的 SHA256 比对成功后安装。未修改全局 pip 配置或项目默认软件源。

## Python 环境升级验证

验证日期：2026-10-17（Asia/Shanghai）。

- 通过 Homebrew 安装 Python 3.12.15，并在用户 shell PATH 中优先使用它；macOS 自带的 `/usr/bin/python3` 3.9.6 保持不变。
- 用 Homebrew Python 3.12.15 重新创建项目 `.venv` 并安装项目依赖。
- `python -m pytest -q -rs`：**11 passed, 4 skipped**。
- `python -m examples.smoke_test`：通过。
- `python -m examples.demo_kinematics`：成功生成运动学预览图。
- `python -m pip check`：未发现依赖冲突。

Homebrew 在预发布 macOS 27 上执行 Python bottle 的最后权限整理步骤时报告 `unknown install step: set_permissions`。安装后的解释器、pip、SSL、SQLite 和 readline 均已单独验证可用；项目环境和全部既有测试也已通过。
