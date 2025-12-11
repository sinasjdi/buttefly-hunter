import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .rigid_body import quat_normalize, rotmat_to_quat


def axis_angle_to_rotmat(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


@dataclass
class RotorConfig:
    # All positions/orientations are expressed in the baselink/body frame.
    position_baselink: np.ndarray  # 3
    base_orientation_baselink: np.ndarray  # 3x3
    tilt_axes_baselink: List[np.ndarray] | None  # axes in baselink frame
    tilt_axes_rotor: List[np.ndarray] | None  # axes in rotor base frame (+X,+Y,+Z before tilts)
    k_thrust: float
    k_drag: float
    spin_dir: int = 1  # +1 or -1
    rotor_inertia: float = 0.0  # around rotor +Z axis
    tilt_thrust_power: float = 0.0  # 0 -> no tilt coupling, 1 -> scale thrust by cos(tilt)

    def tilt_axes(self) -> List[np.ndarray]:
        if self.tilt_axes_rotor is not None:
            return self.tilt_axes_rotor
        return self.tilt_axes_baselink or []


class RotorModel:
    def __init__(self, config: RotorConfig):
        self.config = config

    def rotation_body_from_servos(self, servo_angles: Sequence[float]):
        R = self.config.base_orientation_baselink
        axes = self.config.tilt_axes()
        for axis, ang in zip(axes, servo_angles):
            ax = np.asarray(axis, dtype=float)
            if self.config.tilt_axes_rotor is not None:
                # Axis expressed in rotor base frame; transform to baselink
                ax_bl = self.config.base_orientation_baselink @ ax
            else:
                ax_bl = ax
            # Hinge is fixed in baselink -> pre-multiply
            R = axis_angle_to_rotmat(ax_bl, ang) @ R
        return R

    def thrust_and_torque_body(self, omega: float, servo_angles: Sequence[float], w_body: np.ndarray | None = None):
        R = self.rotation_body_from_servos(servo_angles)
        rotor_axis_body = R @ np.array([0.0, 0.0, 1.0])
        tilt_cos = max(0.0, rotor_axis_body[2])
        tilt_scale = tilt_cos ** self.config.tilt_thrust_power if self.config.tilt_thrust_power > 0.0 else 1.0
        thrust_mag = self.config.k_thrust * omega * omega * tilt_scale
        thrust_rotor = np.array([0.0, 0.0, thrust_mag])
        F_body = R @ thrust_rotor
        torque_drag_body = R @ np.array([0.0, 0.0, self.config.k_drag * omega * omega * self.config.spin_dir])
        tau_gyro = np.zeros(3)
        if w_body is not None and self.config.rotor_inertia > 0.0:
            # Gyroscopic torque from rotor angular momentum when body rotates.
            tau_gyro = self.config.rotor_inertia * omega * np.cross(w_body, rotor_axis_body)
        torque_arm = np.cross(self.config.position_baselink, F_body)
        tau_body = torque_arm + torque_drag_body + tau_gyro
        return F_body, tau_body, R


def rotors_from_config(path: Path) -> List[RotorModel]:
    data = json.loads(Path(path).read_text())
    rotors = []

    def parse_pose(entry):
        if "pose_baselink" in entry:
            pose = np.array(entry["pose_baselink"], dtype=float)
            if pose.size == 16:
                pose = pose.reshape((4, 4))
            if pose.shape != (4, 4):
                raise ValueError("pose_baselink must be 4x4.")
            position = pose[:3, 3]
            orientation = pose[:3, :3]
        else:
            position = np.array(entry.get("position_baselink", entry.get("position_body")), dtype=float)
            base_orientation_entry = entry.get("base_orientation_baselink", entry.get("base_orientation_body", "identity"))
        if isinstance(base_orientation_entry, str) and base_orientation_entry == "identity":
            orientation = np.eye(3)
        else:
            orientation = np.array(base_orientation_entry, dtype=float)
            if orientation.size == 9:
                orientation = orientation.reshape((3, 3))
        return position, orientation

    for entry in data["rotors"]:
        position_baselink, base_orientation_baselink = parse_pose(entry)

        tilt_axes_baselink_entry = entry.get("tilt_axes_baselink", entry.get("tilt_axes_body", None))
        tilt_axes_rotor_entry = entry.get("tilt_axes_rotor", None)
        tilt_axes_baselink = [np.array(ax, dtype=float) for ax in tilt_axes_baselink_entry] if tilt_axes_baselink_entry else None
        tilt_axes_rotor = [np.array(ax, dtype=float) for ax in tilt_axes_rotor_entry] if tilt_axes_rotor_entry else None
        k_thrust = float(entry["k_thrust"])
        k_drag = float(entry["k_drag"])
        spin_dir = int(entry.get("spin_dir", 1))
        rotor_inertia = float(entry.get("rotor_inertia", 0.0))
        tilt_thrust_power = float(entry.get("tilt_thrust_power", 0.0))
        cfg = RotorConfig(
            position_baselink=position_baselink,
            base_orientation_baselink=base_orientation_baselink,
            tilt_axes_baselink=tilt_axes_baselink,
            tilt_axes_rotor=tilt_axes_rotor,
            k_thrust=k_thrust,
            k_drag=k_drag,
            spin_dir=spin_dir,
            rotor_inertia=rotor_inertia,
            tilt_thrust_power=tilt_thrust_power,
        )
        rotors.append(RotorModel(cfg))
    return rotors


def rotor_pose_world(rb_position, rb_rotation, rotor: RotorModel, servo_angles):
    R_body = rb_rotation
    R_rotor_body = rotor.rotation_body_from_servos(servo_angles)
    p_world = rb_position + R_body @ rotor.config.position_baselink
    R_world_rotor = R_body @ R_rotor_body
    q_world = rotmat_to_quat(R_world_rotor)
    return p_world, q_world, R_world_rotor
