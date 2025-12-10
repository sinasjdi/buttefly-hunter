from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .rigid_body import quat_normalize, quat_to_rotmat
from .rotors import RotorModel, rotor_pose_world


@dataclass
class ControlInput:
    omegas: Sequence[float]
    servo_angles: Sequence[Sequence[float]]


def state_to_body_pose_world(x):
    p = x[0:3]
    q = quat_normalize(x[6:10])
    return p, quat_to_rotmat(q), q


class MultiTiltRotorPlant:
    def __init__(self, rotors: List[RotorModel]):
        self.rotors = rotors

    def force_torque(self, t, x, u: ControlInput):
        p_body, R_body, _ = state_to_body_pose_world(x)
        total_F = np.zeros(3)
        total_tau = np.zeros(3)
        rotor_world_poses: List[Tuple[np.ndarray, np.ndarray]] = []

        for idx, rotor in enumerate(self.rotors):
            omega = u.omegas[idx] if idx < len(u.omegas) else 0.0
            servo_angles = u.servo_angles[idx] if idx < len(u.servo_angles) else []
            F_body, tau_body, R_rotor_body = rotor.thrust_and_torque_body(omega, servo_angles)
            total_F += F_body
            total_tau += tau_body

            p_world, q_world, R_world_rotor = rotor_pose_world(p_body, R_body, rotor, servo_angles)
            rotor_world_poses.append((p_world, q_world, R_world_rotor, F_body, tau_body, R_rotor_body))

        return total_F, total_tau, rotor_world_poses
