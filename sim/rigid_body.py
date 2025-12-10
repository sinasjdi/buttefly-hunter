import numpy as np
from dataclasses import dataclass


@dataclass
class RigidBodyParams:
    mass: float
    inertia: np.ndarray  # 3x3
    gravity: float = 9.81


def quat_mult(q, r):
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_from_omega(omega):
    return np.array([0.0, omega[0], omega[1], omega[2]])


def quat_normalize(q):
    return q / np.linalg.norm(q)


def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotmat_to_quat(R):
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return quat_normalize(q)


class RigidBody6DOF:
    """
    Generic 6-DOF rigid body in 3D space.
    State x = [p(3), v(3), q(4), w(3)].
    All forces/torques are provided in body frame.
    """

    def __init__(self, params: RigidBodyParams):
        self.params = params
        self.inv_inertia = np.linalg.inv(params.inertia)

    def dynamics(self, t, x, force_torque_fn):
        """
        Compute x_dot = f(t, x) given a force/torque provider.
        force_torque_fn(t, x) -> (F_body[3], tau_body[3])
        """
        m = self.params.mass
        g = self.params.gravity

        p = x[0:3]
        v = x[3:6]
        q = quat_normalize(x[6:10])
        w_body = x[10:13]

        F_body, tau_body = force_torque_fn(t, x)
        R = quat_to_rotmat(q)

        p_dot = v
        v_dot = np.array([0.0, 0.0, -g]) + (1.0 / m) * (R @ F_body)
        q_dot = 0.5 * quat_mult(q, quat_from_omega(w_body))

        I = self.params.inertia
        w_dot = self.inv_inertia @ (tau_body - np.cross(w_body, I @ w_body))

        x_dot = np.zeros_like(x)
        x_dot[0:3] = p_dot
        x_dot[3:6] = v_dot
        x_dot[6:10] = q_dot
        x_dot[10:13] = w_dot
        return x_dot

