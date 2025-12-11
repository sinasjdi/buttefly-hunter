"""
Lightweight control helpers: tilt-aware wrench allocator for overactuated
tilt-rotor vehicles. Uses simple geometry to pick servo angles and a regularized
least-squares solve to distribute thrusts.

Usage:
    from sim.lib_control import TiltAllocator
    allocator = TiltAllocator(rotors, thrust_min=0.0, thrust_max=15.0, tilt_limit=np.deg2rad(35))
    u = allocator.allocate(wrench_des=np.hstack([F_des, tau_des]), desired_body_dir=None)
    # u is ControlInput (omegas, servo_angles) ready for the plant
"""

from dataclasses import dataclass
from math import atan2
from typing import List, Sequence

import numpy as np

from .plant import ControlInput
from .rotors import RotorModel, axis_angle_to_rotmat


def _normalize(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-9)


def _tilt_angle_to_align(axis_body: np.ndarray, rotor_axis_body: np.ndarray, desired_dir_body: np.ndarray) -> float:
    """
    Find the rotation angle about axis_body that best aligns rotor_axis_body to desired_dir_body.
    Works by projecting both vectors into the plane orthogonal to the hinge axis.
    """
    a = _normalize(axis_body)
    r_proj = rotor_axis_body - a * np.dot(a, rotor_axis_body)
    d_proj = desired_dir_body - a * np.dot(a, desired_dir_body)
    if np.linalg.norm(r_proj) < 1e-9 or np.linalg.norm(d_proj) < 1e-9:
        return 0.0
    r_proj = _normalize(r_proj)
    d_proj = _normalize(d_proj)
    sin_term = np.dot(a, np.cross(r_proj, d_proj))
    cos_term = np.dot(r_proj, d_proj)
    return atan2(sin_term, cos_term)


@dataclass
class TiltAllocator:
    rotors: List[RotorModel]
    thrust_min: float = 0.0
    thrust_max: float = 30.0
    tilt_limit: float = np.deg2rad(35.0)
    reg: float = 1e-4  # regularization to pick a unique solution in the nullspace
    yaw_drag_scale: float = 1.0  # scale k_drag contribution; set 0 to ignore

    def _aim_servo_single_axis(self, rotor: RotorModel, desired_dir_body: np.ndarray) -> List[float]:
        """
        Compute a single hinge angle to steer rotor +Z toward desired_dir_body.
        If the rotor lacks tilt axes, returns [].
        """
        axes = rotor.config.tilt_axes()
        if not axes:
            return []
        # Use first axis only for now
        axis = np.asarray(axes[0], dtype=float)
        if rotor.config.tilt_axes_rotor is not None:
            axis_body = rotor.config.base_orientation_baselink @ axis
        else:
            axis_body = axis
        R0 = rotor.config.base_orientation_baselink
        rotor_axis_body = R0 @ np.array([0.0, 0.0, 1.0])
        theta = _tilt_angle_to_align(axis_body, rotor_axis_body, desired_dir_body)
        theta = np.clip(theta, -self.tilt_limit, self.tilt_limit)
        return [theta]

    def _thrust_direction_and_tau_col(self, rotor: RotorModel, servo_angles: Sequence[float]):
        R = rotor.rotation_body_from_servos(servo_angles)
        dir_body = R @ np.array([0.0, 0.0, 1.0])
        tau_arm = np.cross(rotor.config.position_baselink, dir_body)
        yaw_gain = 0.0
        if rotor.config.k_thrust > 1e-9:
            yaw_gain = (rotor.config.k_drag / rotor.config.k_thrust) * rotor.config.spin_dir * self.yaw_drag_scale
        tau_col = tau_arm + yaw_gain * dir_body
        return dir_body, tau_col

    def allocate(
        self,
        wrench_des: np.ndarray,
        desired_dir_body: np.ndarray | None = None,
    ) -> ControlInput:
        """
        Allocate thrusts + tilt angles to track a desired wrench [Fx,Fy,Fz,Tx,Ty,Tz].
        desired_dir_body: optional preferred thrust direction for tilts. If None, uses
        the desired force direction (from wrench_des) falling back to body +Z.
        Returns ControlInput with per-rotor omegas and servo angles.
        """
        wrench_des = np.asarray(wrench_des, dtype=float).reshape(-1)
        if wrench_des.size != 6:
            raise ValueError("wrench_des must have 6 elements (Fx,Fy,Fz,Tx,Ty,Tz).")
        if desired_dir_body is not None:
            dir_pref = _normalize(desired_dir_body)
        else:
            F_des = wrench_des[0:3]
            if np.linalg.norm(F_des) > 1e-6:
                dir_pref = _normalize(F_des)
            else:
                dir_pref = np.array([0.0, 0.0, 1.0])

        servo_angles_cmd: List[List[float]] = []
        for rotor in self.rotors:
            servo_angles_cmd.append(self._aim_servo_single_axis(rotor, dir_pref))

        n = len(self.rotors)
        B = np.zeros((6, n))
        for i, rotor in enumerate(self.rotors):
            dir_body, tau_col = self._thrust_direction_and_tau_col(rotor, servo_angles_cmd[i])
            B[0:3, i] = dir_body
            B[3:6, i] = tau_col

        # Regularized least squares: minimize ||B f - w||^2 + reg * ||f||^2
        BtB = B.T @ B + self.reg * np.eye(n)
        btw = B.T @ wrench_des
        try:
            f = np.linalg.solve(BtB, btw)
        except np.linalg.LinAlgError:
            f, *_ = np.linalg.lstsq(BtB, btw, rcond=None)

        # Clamp thrusts
        f = np.clip(f, self.thrust_min, self.thrust_max)

        omegas: List[float] = []
        for thrust, rotor in zip(f, self.rotors):
            k = rotor.config.k_thrust if rotor.config.k_thrust > 1e-9 else 1.0
            omega = np.sqrt(max(thrust, 0.0) / k)
            omegas.append(omega)

        return ControlInput(omegas=omegas, servo_angles=servo_angles_cmd)
