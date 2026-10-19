"""SI 单位；三个转动关节，连杆质心位于各连杆局部 x 轴。"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RobotParameters:
    lengths: tuple[float, float, float] = (1.0, 0.8, 0.6)  # m
    masses: tuple[float, float, float] = (1.0, 0.8, 0.6)  # kg
    com_offsets: tuple[float, float, float] = (0.5, 0.4, 0.3)  # m from proximal joint
    inertias: tuple[float, float, float] = (0.083333333333, 0.042666666667, 0.018)  # kg m² about COM z
    gravity: float = 9.81  # positive magnitude; world gravity points along -y

    def __post_init__(self):
        for name in ("lengths", "masses", "com_offsets", "inertias"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain three finite values")
            if name != "com_offsets" and np.any(values <= 0):
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, tuple(float(v) for v in values))
        if np.any(np.asarray(self.com_offsets) < 0) or np.any(
            np.asarray(self.com_offsets) > np.asarray(self.lengths)
        ):
            raise ValueError("com_offsets must lie within each link")
        if not np.isfinite(self.gravity) or self.gravity < 0:
            raise ValueError("gravity must be finite and nonnegative")
