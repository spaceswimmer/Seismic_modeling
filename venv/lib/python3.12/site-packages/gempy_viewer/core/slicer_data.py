from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(init=True)
class SlicerData:
    x: str
    y: str
    Gx: str
    Gy: str
    select_projected_p: np.ndarray
    select_projected_o: np.ndarray
    regular_grid_x_idx: Optional[int] = None
    regular_grid_y_idx: Optional[int] = None
    regular_grid_z_idx: Optional[int] = None


