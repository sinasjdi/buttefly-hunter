"""
CasADi-based nonlinear MPC for tilt-rotor.

State: x = [p(3), v(3), q(4), w(3), servos(ns)]
Control: u = [thrusts(nr), servo_angles(ns)]
Dynamics: integrates rigid-body with thrust/torque from rotors using servo angles.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import casadi as ca
import numpy as np

from .rotors import RotorModel


def rodrigues(axis, angle):
    axis = axis / (ca.norm_2(axis) + 1e-9)
    x, y, z = axis[0], axis[1], axis[2]
    c = ca.cos(angle)
    s = ca.sin(angle)
    C = 1 - c
    return ca.vertcat(
        ca.hcat([c + x * x * C, x * y * C - z * s, x * z * C + y * s]),
        ca.hcat([y * x * C + z * s, c + y * y * C, y * z * C - x * s]),
        ca.hcat([z * x * C - y * s, z * y * C + x * s, c + z * z * C]),
    )


def quat_mult(q, r):
    w1, x1, y1, z1 = q[0], q[1], q[2], q[3]
    w2, x2, y2, z2 = r[0], r[1], r[2], r[3]
    return ca.vertcat(
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def quat_from_omega(w):
    return ca.vertcat(0, w[0], w[1], w[2])


def quat_to_rotmat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return ca.vertcat(
        ca.hcat([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
        ca.hcat([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
        ca.hcat([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
    )


def quat_normalize(q):
    return q / (ca.norm_2(q) + 1e-9)


def quat_error_cost(q, q_ref):
    dot = ca.dot(q, q_ref)
    return 1 - dot * dot


@dataclass
class NMPCConfig:
    dt: float = 0.01
    horizon: int = 15
    servo_rate: float = 15.0
    q_pos: float = 20.0
    q_vel: float = 6.0
    q_att: float = 5.0
    q_rate: float = 1.0
    r_thrust: float = 0.1
    r_servo: float = 0.05
    thrust_min: float = 0.0
    thrust_max: float = 50.0
    servo_min: float = -np.pi / 2
    servo_max: float = np.pi / 2


class NMPCController:
    def __init__(self, mass: float, inertia: np.ndarray, gravity: float, rotors: List[RotorModel], cfg: NMPCConfig):
        self.mass = mass
        self.inertia = inertia
        self.inv_inertia = np.linalg.inv(inertia)
        self.g = gravity
        self.rotors = rotors
        self.cfg = cfg
        self.nr = len(rotors)
        self.ns = sum(len(r.config.tilt_axes()) for r in rotors)
        self.nx = 13 + self.ns
        self.nu = self.nr + self.ns
        self._build_solver()

    def _dynamics(self, x, u):
        nr = self.nr
        ns = self.ns
        p = x[0:3]
        v = x[3:6]
        q = quat_normalize(x[6:10])
        w = x[10:13]
        servos = x[13 : 13 + ns]

        thrusts = u[0:nr]
        servo_cmds = u[nr : nr + ns]

        R_body = quat_to_rotmat(q)
        total_F = ca.SX.zeros(3, 1)
        total_tau = ca.SX.zeros(3, 1)
        s_idx = 0
        for i, r in enumerate(self.rotors):
            R = ca.SX(r.config.base_orientation_baselink)
            axes = r.config.tilt_axes() or []
            for ax in axes:
                ax_vec = np.asarray(ax, dtype=float)
                if r.config.tilt_axes_rotor is not None:
                    ax_body = r.config.base_orientation_baselink @ ax_vec
                else:
                    ax_body = ax_vec
                R = rodrigues(ca.SX(ax_body), servos[s_idx]) @ R
                s_idx += 1
            thrust_dir = R @ ca.SX([0.0, 0.0, 1.0])
            F_body = thrusts[i] * thrust_dir
            torque_arm = ca.cross(ca.SX(r.config.position_baselink), F_body)
            yaw_gain = (r.config.k_drag / r.config.k_thrust) * r.config.spin_dir if r.config.k_thrust > 1e-9 else 0.0
            tau_body = torque_arm + yaw_gain * F_body
            total_F += F_body
            total_tau += tau_body

        p_dot = v
        v_dot = ca.vertcat(0, 0, -self.g) + (1.0 / self.mass) * (R_body @ total_F)
        q_dot = 0.5 * quat_mult(q, quat_from_omega(w))
        w_dot = ca.mtimes(self.inv_inertia, (total_tau - ca.cross(w, ca.mtimes(self.inertia, w))))
        servo_rate = float(self.cfg.servo_rate) if hasattr(self.cfg, "servo_rate") else 0.0
        servos_dot = servo_rate * (servo_cmds - servos) if servo_rate > 0 else servo_cmds - servos

        return ca.vertcat(p_dot, v_dot, q_dot, w_dot, servos_dot)

    def _integrate(self, x, u):
        return x + self.cfg.dt * self._dynamics(x, u)

    def _build_solver(self):
        cfg = self.cfg
        nx = self.nx
        nu = self.nu
        N = cfg.horizon

        X = ca.SX.sym("X", nx, N + 1)
        U = ca.SX.sym("U", nu, N)
        Xref = ca.SX.sym("Xref", nx, N + 1)

        cost = 0
        g = []
        lbg = []
        ubg = []

        # initial state param
        X0 = ca.SX.sym("X0", nx)
        g.append(X[:, 0] - X0)
        lbg.extend([0.0] * nx)
        ubg.extend([0.0] * nx)

        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]
            xref_k = Xref[:, k]

            e_pos = xk[0:3] - xref_k[0:3]
            e_vel = xk[3:6] - xref_k[3:6]
            qk = quat_normalize(xk[6:10])
            qref = quat_normalize(xref_k[6:10])
            att_cost = quat_error_cost(qk, qref)
            e_rate = xk[10:13] - xref_k[10:13]
            thrusts = uk[0:self.nr]
            servos = uk[self.nr : self.nr + self.ns]

            cost += cfg.q_pos * ca.dot(e_pos, e_pos)
            cost += cfg.q_vel * ca.dot(e_vel, e_vel)
            cost += cfg.q_att * att_cost
            cost += cfg.q_rate * ca.dot(e_rate, e_rate)
            cost += cfg.r_thrust * ca.dot(thrusts, thrusts)
            cost += cfg.r_servo * ca.dot(servos, servos)

            x_next = X[:, k + 1]
            x_pred = self._integrate(xk, uk)
            g.append(x_next - x_pred)
            lbg.extend([0.0] * nx)
            ubg.extend([0.0] * nx)

            # bounds
            g.append(thrusts)
            lbg.extend([cfg.thrust_min] * self.nr)
            ubg.extend([cfg.thrust_max] * self.nr)
            if self.ns > 0:
                g.append(servos)
                lbg.extend([cfg.servo_min] * self.ns)
                ubg.extend([cfg.servo_max] * self.ns)

        # terminal cost
        e_pos_t = X[0:3, N] - Xref[0:3, N]
        e_vel_t = X[3:6, N] - Xref[3:6, N]
        q_t = quat_normalize(X[6:10, N])
        qref_t = quat_normalize(Xref[6:10, N])
        att_cost_t = quat_error_cost(q_t, qref_t)
        e_rate_t = X[10:13, N] - Xref[10:13, N]
        cost += cfg.q_pos * ca.dot(e_pos_t, e_pos_t)
        cost += cfg.q_vel * ca.dot(e_vel_t, e_vel_t)
        cost += cfg.q_att * att_cost_t
        cost += cfg.q_rate * ca.dot(e_rate_t, e_rate_t)

        g = ca.vertcat(*g)
        P = ca.vertcat(ca.reshape(Xref, -1, 1), X0)
        nlp = {"x": ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)), "f": cost, "g": g, "p": P}
        opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": 30}
        self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)
        self.lbg = np.array(lbg, dtype=float)
        self.ubg = np.array(ubg, dtype=float)

    def solve(self, x0: np.ndarray, x_refs: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        X0_guess = np.tile(x0.reshape(-1, 1), (1, cfg.horizon + 1))
        U_guess = np.zeros((self.nu, cfg.horizon))
        pref = np.concatenate([xr.reshape(-1, 1) for xr in x_refs], axis=1)
        p_vec = np.concatenate([pref.reshape((-1, 1), order="F"), x0.reshape((-1, 1))], axis=0)
        sol = self.solver(
            x0=np.concatenate([X0_guess.reshape((-1, 1), order="F"), U_guess.reshape((-1, 1), order="F")], axis=0),
            p=p_vec,
            lbg=self.lbg,
            ubg=self.ubg,
        )
        z = np.array(sol["x"]).flatten()
        X_opt = z[: self.nx * (cfg.horizon + 1)].reshape((self.nx, cfg.horizon + 1), order="F")
        U_opt = z[self.nx * (cfg.horizon + 1) :].reshape((self.nu, cfg.horizon), order="F")
        return X_opt, U_opt
