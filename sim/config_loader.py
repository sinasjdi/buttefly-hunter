import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .rotors import RotorConfig, RotorModel


@dataclass
class VehicleConfig:
    name: str
    mass: float
    inertia: np.ndarray  # 3x3
    gravity: float = 9.81


def _parse_inertia(values):
    if len(values) != 9:
        raise ValueError("Inertia must have 9 elements (row-major 3x3).")
    return np.array(values, dtype=float).reshape((3, 3))


def load_vehicle_config(path: Path) -> Tuple[VehicleConfig, List[RotorModel]]:
    data = json.loads(Path(path).read_text())
    vehicle = data.get("vehicle", {})
    vehicle_cfg = VehicleConfig(
        name=vehicle.get("name", "uav"),
        mass=float(vehicle["mass"]),
        inertia=_parse_inertia(vehicle["inertia"]),
        gravity=float(vehicle.get("gravity", 9.81)),
    )

    def _parse_pose(entry):
        if "pose_baselink" in entry:
            pose = np.array(entry["pose_baselink"], dtype=float)
            if pose.size == 16:
                pose = pose.reshape((4, 4))
            if pose.shape != (4, 4):
                raise ValueError("pose_baselink must be 4x4 homogeneous transform.")
            position = pose[:3, 3]
            orientation = pose[:3, :3]
        else:
            position = np.array(entry.get("position_baselink", entry.get("position_body")), dtype=float)
            base_orientation = entry.get("base_orientation_baselink", entry.get("base_orientation_body", "identity"))
            if isinstance(base_orientation, str) and base_orientation == "identity":
                orientation = np.eye(3)
            else:
                orientation = np.array(base_orientation, dtype=float)
                if orientation.size == 9:
                    orientation = orientation.reshape((3, 3))
        return position, orientation

    rotors = []
    for entry in data["rotors"]:
        position_baselink, base_orientation_baselink = _parse_pose(entry)
        tilt_axes_baselink_list = entry.get("tilt_axes_baselink", entry.get("tilt_axes_body", None))
        tilt_axes_rotor_list = entry.get("tilt_axes_rotor", None)
        tilt_axes_baselink = [np.array(ax, dtype=float) for ax in tilt_axes_baselink_list] if tilt_axes_baselink_list else None
        tilt_axes_rotor = [np.array(ax, dtype=float) for ax in tilt_axes_rotor_list] if tilt_axes_rotor_list else None
        cfg = RotorConfig(
            position_baselink=position_baselink,
            base_orientation_baselink=base_orientation_baselink,
            tilt_axes_baselink=tilt_axes_baselink,
            tilt_axes_rotor=tilt_axes_rotor,
            k_thrust=float(entry["k_thrust"]),
            k_drag=float(entry["k_drag"]),
            spin_dir=int(entry.get("spin_dir", 1)),
        )
        rotors.append(RotorModel(cfg))

    return vehicle_cfg, rotors
