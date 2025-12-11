import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .plant import ControlInput
from .rotors import RotorModel


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class StepLogger:
    """
    CSV logger for simulation steps.
    Creates logs/<timestamp>/<name>.csv and appends one row per step with state + control.
    """

    rotors: List[RotorModel]
    name: str = "log"
    base_dir: Path = Path("logs")

    def __post_init__(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = _ensure_dir(self.base_dir / ts)
        self.path = self.log_dir / f"{self.name}.csv"
        self._header_written = False

    def _header(self) -> List[str]:
        hdr = [
            "t",
            "px",
            "py",
            "pz",
            "vx",
            "vy",
            "vz",
            "qw",
            "qx",
            "qy",
            "qz",
            "wx",
            "wy",
            "wz",
        ]
        for i, rotor in enumerate(self.rotors):
            hdr.append(f"omega_{i}")
            for j, _ in enumerate(rotor.config.tilt_axes()):
                hdr.append(f"servo_{i}_{j}")
        return hdr

    def _row(self, t: float, x: np.ndarray, u: ControlInput | None):
        vals = [
            float(t),
            float(x[0]),
            float(x[1]),
            float(x[2]),
            float(x[3]),
            float(x[4]),
            float(x[5]),
            float(x[6]),
            float(x[7]),
            float(x[8]),
            float(x[9]),
            float(x[10]),
            float(x[11]),
            float(x[12]),
        ]
        if u is None:
            return vals
        for i in range(len(self.rotors)):
            omega = u.omegas[i] if i < len(u.omegas) else 0.0
            vals.append(float(omega))
            servo_list = u.servo_angles[i] if i < len(u.servo_angles) else []
            for j, _ in enumerate(self.rotors[i].config.tilt_axes()):
                ang = servo_list[j] if j < len(servo_list) else 0.0
                vals.append(float(ang))
        return vals

    def log(self, t: float, x: np.ndarray, u: ControlInput | None):
        mode = "a"
        with open(self.path, mode, newline="") as f:
            writer = csv.writer(f)
            if not self._header_written:
                writer.writerow(self._header())
                self._header_written = True
            writer.writerow(self._row(t, x, u))
