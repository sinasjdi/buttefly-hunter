from typing import List, Sequence, Tuple

import numpy as np

from .rigid_body import quat_normalize, quat_to_rotmat, rotmat_to_quat
from .rotors import RotorModel


def state_to_body_pose_world(x) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract body pose from full state vector."""
    p = x[0:3]
    q = quat_normalize(x[6:10])
    R = quat_to_rotmat(q)
    return p, R, q


class GeometryModel:
    """
    Pure geometry/kinematics layer for the rotor layout.
    Given body pose + servo angles, computes rotor world poses and thrust directions.
    Includes getters/setters to inspect or edit geometry parameters in baselink.
    """

    def __init__(self, rotors: List[RotorModel]):
        self._rotors = rotors

    @property
    def rotors(self) -> List[RotorModel]:
        return self._rotors

    def rotor_count(self) -> int:
        return len(self._rotors)

    def get_rotor_config(self, idx: int) -> RotorModel:
        return self._rotors[idx]

    def get_pose_baselink(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (position, orientation 3x3) in baselink/body frame."""
        r = self._rotors[idx].config
        return r.position_baselink.copy(), r.base_orientation_baselink.copy()

    def set_pose_baselink(
        self,
        idx: int,
        position: np.ndarray | None = None,
        orientation: np.ndarray | None = None,
        pose_4x4: np.ndarray | None = None,
    ):
        """Set rotor pose in baselink/body frame using position/orientation or a 4x4 homogeneous pose."""
        cfg = self._rotors[idx].config
        if pose_4x4 is not None:
            pose = np.asarray(pose_4x4, dtype=float)
            if pose.size == 16:
                pose = pose.reshape((4, 4))
            if pose.shape != (4, 4):
                raise ValueError("pose_4x4 must be 4x4.")
            cfg.position_baselink = pose[:3, 3]
            cfg.base_orientation_baselink = pose[:3, :3]
            return
        if position is not None:
            cfg.position_baselink = np.asarray(position, dtype=float)
        if orientation is not None:
            ori = np.asarray(orientation, dtype=float)
            if ori.size == 9:
                ori = ori.reshape((3, 3))
            if ori.shape != (3, 3):
                raise ValueError("orientation must be 3x3.")
            cfg.base_orientation_baselink = ori

    def set_tilt_axes(self, idx: int, tilt_axes: Sequence[Sequence[float]]):
        """Replace tilt axes list for rotor idx (expressed in baselink/body frame)."""
        self._rotors[idx].config.tilt_axes_baselink = [np.asarray(ax, dtype=float) for ax in tilt_axes]

    def set_motor_constants(self, idx: int, k_thrust: float | None = None, k_drag: float | None = None, spin_dir: int | None = None):
        """Update thrust/drag constants or spin direction for rotor idx."""
        cfg = self._rotors[idx].config
        if k_thrust is not None:
            cfg.k_thrust = float(k_thrust)
        if k_drag is not None:
            cfg.k_drag = float(k_drag)
        if spin_dir is not None:
            cfg.spin_dir = int(spin_dir)

    def rotor_world_pose(
        self, body_position: np.ndarray, body_rotation: np.ndarray, servo_angles: Sequence[Sequence[float]]
    ):
        """Return list of (p_world, q_world, R_world_rotor) for each rotor."""
        out = []
        for idx, rotor in enumerate(self._rotors):
            angles = servo_angles[idx] if idx < len(servo_angles) else []
            R_rotor_body = rotor.rotation_body_from_servos(angles)
            p_world = body_position + body_rotation @ rotor.config.position_baselink
            R_world_rotor = body_rotation @ R_rotor_body
            q_world_rotor = rotmat_to_quat(R_world_rotor)
            out.append((p_world, q_world_rotor, R_world_rotor))
        return out

    def thrust_directions_world(self, rotor_world_rotations: List[np.ndarray]) -> List[np.ndarray]:
        """Unit thrust directions in world frame (assuming rotor thrust along +Z in rotor frame)."""
        dirs = []
        for R_world_rotor in rotor_world_rotations:
            dirs.append(R_world_rotor @ np.array([0.0, 0.0, 1.0]))
        return dirs

    def thrust_vectors_world(
        self,
        rotor_world_rotations: List[np.ndarray],
        omegas: Sequence[float],
    ) -> List[np.ndarray]:
        """
        Thrust vectors (direction and magnitude) in world frame using k_thrust * omega^2.
        Length encodes thrust magnitude; direction follows rotor +Z rotated into world.
        """
        thrusts = []
        for idx, R_world_rotor in enumerate(rotor_world_rotations):
            omega = omegas[idx] if idx < len(omegas) else 0.0
            mag = self._rotors[idx].config.k_thrust * omega * omega
            thrusts.append(R_world_rotor @ np.array([0.0, 0.0, mag]))
        return thrusts

    def thrust_vectors_body(
        self,
        rotor_rotations_body: List[np.ndarray],
        omegas: Sequence[float],
    ) -> List[np.ndarray]:
        """Thrust vectors in body frame (k_thrust * omega^2 along rotor +Z rotated into body)."""
        thrusts = []
        for idx, R_rotor_body in enumerate(rotor_rotations_body):
            omega = omegas[idx] if idx < len(omegas) else 0.0
            mag = self._rotors[idx].config.k_thrust * omega * omega
            thrusts.append(R_rotor_body @ np.array([0.0, 0.0, mag]))
        return thrusts

    def rotor_outputs(
        self,
        body_position: np.ndarray,
        body_rotation: np.ndarray,
        servo_angles: Sequence[Sequence[float]],
        omegas: Sequence[float],
    ):
        """
        Compute per-rotor pose and thrust in both body and world frames.
        Returns list of dicts with keys: position_world, quaternion_world, R_world_rotor,
        thrust_body, thrust_world, torque_body, torque_world.
        """
        outputs = []
        for idx, rotor in enumerate(self._rotors):
            omega = omegas[idx] if idx < len(omegas) else 0.0
            angles = servo_angles[idx] if idx < len(servo_angles) else []
            F_body, tau_body, R_rotor_body = rotor.thrust_and_torque_body(omega, angles)
            p_world = body_position + body_rotation @ rotor.config.position_baselink
            R_world_rotor = body_rotation @ R_rotor_body
            q_world_rotor = rotmat_to_quat(R_world_rotor)
            outputs.append(
                {
                    "position_world": p_world,
                    "quaternion_world": q_world_rotor,
                    "R_world_rotor": R_world_rotor,
                    "thrust_body": F_body,
                    "thrust_world": body_rotation @ F_body,
                    "torque_body": tau_body,
                    "torque_world": body_rotation @ tau_body,
                }
            )
        return outputs

    def from_state(self, x, servo_angles: Sequence[Sequence[float]]):
        """Convenience: given full state and servo angles, return per-rotor world pose and thrust dir."""
        p_body, R_body, q_body = state_to_body_pose_world(x)
        rotor_poses = self.rotor_world_pose(p_body, R_body, servo_angles)
        thrust_dirs = self.thrust_directions_world([R for *_ , R in rotor_poses])
        return p_body, q_body, rotor_poses, thrust_dirs
