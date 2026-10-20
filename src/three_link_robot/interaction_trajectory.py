"""接触法向力与切向位置的参考值。"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ForceReference:
    force_n: float  # 机器人施加于墙的法向力大小，N，非负
    y: float  # 切向位置，m

    def __post_init__(self):
        force_n, y = float(self.force_n), float(self.y)
        if not np.isfinite(force_n) or force_n < 0 or not np.isfinite(y):
            raise ValueError("force_n must be finite and nonnegative; y must be finite")
        object.__setattr__(self, "force_n", force_n)
        object.__setattr__(self, "y", y)


def force_hold(force_n, y):
    reference = ForceReference(force_n, y)
    return lambda t: reference
