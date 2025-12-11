"""
Lightweight dynamics helper for multi-tilt-rotor vehicles.

Usage (direct omegas):
    from sim.lib_dynamics import build_dynamics, ControlInput
    dyn = build_dynamics("sim/config/vehicle_example.json", dt=0.005)
    u = ControlInput(omegas=[omega0, ...], servo_angles=[[ang0], ...])
    dyn.step(u)
    snap = dyn.snapshot()  # dict compatible with viewer/server payloads

Usage (first-order rotor dynamics):
    dyn = build_dynamics("sim/config/vehicle_example.json", dt=0.005, rotor_time_constant=0.05)
    dyn.step_with_commands(omega_cmds=[cmd0, ...], servo_angles=[[ang0], ...])

Usage (with servo dynamics, damped 2nd order):
    dyn = build_dynamics("sim/config/vehicle_example.json", dt=0.005,
                         rotor_time_constant=0.05, servo_omega_n=50.0, servo_zeta=0.7)
    dyn.step_with_commands(omega_cmds=[cmd0, ...], servo_angles=[[cmd_ang0], ...])

The vehicle/rotor layout comes from the JSON config, so rotor count/poses/tilt
axes are all configurable without code changes.
"""

from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np

from .config_loader import VehicleConfig, load_vehicle_config
from .plant import ControlInput, MultiTiltRotorPlant, state_to_body_pose_world
from .rigid_body import RigidBody6DOF, RigidBodyParams
from .rotors import RotorModel


def _default_state() -> np.ndarray:
    p0 = np.array([0.0, 0.0, 0.0])
    v0 = np.zeros(3)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    w0 = np.zeros(3)
    return np.concatenate([p0, v0, q0, w0])


class RotorFirstOrderActuator:
    """First-order rotor speed dynamics: d(omega)/dt = (cmd - omega) / tau."""

    def __init__(
        self,
        rotor_count: int,
        tau: float | Sequence[float],
        omega_min: float | Sequence[float] = 0.0,
        omega_max: float | Sequence[float] | None = None,
        initial_omega: float | Sequence[float] = 0.0,
    ):
        self.rotor_count = rotor_count
        self.tau = self._broadcast(tau, fill=0.05)
        self.omega_min = self._broadcast(omega_min, fill=0.0)
        self.omega_max = self._broadcast(omega_max, fill=None) if omega_max is not None else None
        self.omegas = self._broadcast(initial_omega, fill=0.0)

    def _broadcast(self, val, fill):
        if isinstance(val, (list, tuple, np.ndarray)):
            arr = np.asarray(val, dtype=float)
            if arr.size != self.rotor_count:
                raise ValueError(f"Expected {self.rotor_count} elements, got {arr.size}")
            return arr.copy()
        if val is None:
            arr = np.array([fill] * self.rotor_count, dtype=float)
        else:
            arr = np.array([val] * self.rotor_count, dtype=float)
        return arr

    def reset(self, omegas: Sequence[float] | None = None):
        self.omegas = self._broadcast(omegas, fill=0.0) if omegas is not None else self.omegas * 0.0

    def step(self, omega_cmds: Sequence[float], dt: float) -> np.ndarray:
        cmds = np.asarray(self._broadcast(omega_cmds, fill=0.0), dtype=float)
        # Explicit Euler for first-order lag
        self.omegas = self.omegas + (cmds - self.omegas) * (dt / self.tau)
        # Clamp
        self.omegas = np.maximum(self.omegas, self.omega_min)
        if self.omega_max is not None:
            self.omegas = np.minimum(self.omegas, self.omega_max)
        return self.omegas.copy()


class ServoSecondOrderActuator:
    """
    Damped second-order servo dynamics for tilt axes (per rotor).
    theta_ddot = omega_n^2 * (cmd - theta) - 2 * zeta * omega_n * theta_dot
    """

    def __init__(
        self,
        servo_counts: Sequence[int],
        omega_n: float | Sequence[float] = 50.0,
        zeta: float | Sequence[float] = 0.7,
        angle_min: float | Sequence[float] | None = None,
        angle_max: float | Sequence[float] | None = None,
        initial_angle: float | Sequence[float] = 0.0,
    ):
        self.servo_counts = list(servo_counts)
        self.total = sum(self.servo_counts)
        self._slices = []
        start = 0
        for count in self.servo_counts:
            self._slices.append(slice(start, start + count))
            start += count

        self.omega_n = self._broadcast(omega_n, default=50.0)
        self.zeta = self._broadcast(zeta, default=0.7)
        self.angle_min = self._broadcast(angle_min, default=None) if angle_min is not None else None
        self.angle_max = self._broadcast(angle_max, default=None) if angle_max is not None else None
        self.angle = self._broadcast(initial_angle, default=0.0)
        self.rate = np.zeros_like(self.angle)

    def _broadcast(self, val, default):
        if isinstance(val, (list, tuple, np.ndarray)):
            arr = np.asarray(val, dtype=float)
            if arr.size != self.total:
                raise ValueError(f"Expected {self.total} elements, got {arr.size}")
            return arr.copy()
        if val is None:
            return np.array([default] * self.total, dtype=float)
        return np.array([val] * self.total, dtype=float)

    def reset(self, angles: Sequence[float] | None = None, rates: Sequence[float] | None = None):
        if angles is not None:
            self.angle = self._broadcast(angles, default=0.0)
        else:
            self.angle = self.angle * 0.0
        if rates is not None:
            self.rate = self._broadcast(rates, default=0.0)
        else:
            self.rate = self.rate * 0.0

    def _flatten_cmds(self, servo_cmds: Sequence[Sequence[float]] | None):
        flat = np.zeros(self.total, dtype=float)
        if servo_cmds is None:
            return flat
        idx = 0
        for rotor_idx, count in enumerate(self.servo_counts):
            cmds = servo_cmds[rotor_idx] if rotor_idx < len(servo_cmds) else []
            for j in range(count):
                flat[idx] = cmds[j] if j < len(cmds) else 0.0
                idx += 1
        return flat

    def _unflatten(self, flat: np.ndarray) -> List[List[float]]:
        out: List[List[float]] = []
        for sl in self._slices:
            out.append(flat[sl].tolist())
        return out

    def step(self, servo_cmds: Sequence[Sequence[float]] | None, dt: float) -> List[List[float]]:
        cmd_flat = self._flatten_cmds(servo_cmds)
        theta_ddot = (self.omega_n ** 2) * (cmd_flat - self.angle) - 2.0 * self.zeta * self.omega_n * self.rate
        # Semi-implicit Euler: update rate, then angle
        self.rate = self.rate + theta_ddot * dt
        self.angle = self.angle + self.rate * dt
        # Clamp angles
        if self.angle_min is not None:
            self.angle = np.maximum(self.angle, self.angle_min)
        if self.angle_max is not None:
            self.angle = np.minimum(self.angle, self.angle_max)
        return self._unflatten(self.angle)


class DynamicsModel:
    """
    Forward dynamics for a configured multi-tilt-rotor UAV.
    Uses RigidBody6DOF + MultiTiltRotorPlant and integrates with RK4.
    """

    def __init__(
        self,
        vehicle_cfg: VehicleConfig,
        rotors: List[RotorModel],
        dt: float = 0.005,
        actuator: RotorFirstOrderActuator | None = None,
        servo_actuator: ServoSecondOrderActuator | None = None,
        ground_height: float | None = None,
        ground_k: float = 0.0,
        ground_d: float = 0.0,
    ):
        self.vehicle_cfg = vehicle_cfg
        self.rotors = rotors
        self.dt = dt
        self.actuator = actuator
        self.servo_actuator = servo_actuator
        self.rb = RigidBody6DOF(
            RigidBodyParams(
                mass=vehicle_cfg.mass,
                inertia=vehicle_cfg.inertia,
                gravity=vehicle_cfg.gravity,
            )
        )
        self.plant = MultiTiltRotorPlant(
            rotors,
            ground_height=ground_height,
            ground_k=ground_k,
            ground_d=ground_d,
        )
        self.reset()

    def reset(self, state: np.ndarray | None = None, t: float = 0.0):
        """Reset simulation state and time."""
        self.x = state.copy() if state is not None else _default_state()
        self.t = float(t)
        self._last_rotor_world = []
        self._last_R_body = np.eye(3)
        self.last_u: ControlInput | None = None
        if self.actuator is not None:
            self.actuator.reset()
        if self.servo_actuator is not None:
            self.servo_actuator.reset()

    def _derivative(self, t: float, x: np.ndarray, u: ControlInput):
        """Compute x_dot and cache rotor/world data for visualization."""
        F_body, tau_body, rotor_world = self.plant.force_torque(t, x, u)
        self._last_rotor_world = rotor_world
        _, R_body, _ = state_to_body_pose_world(x)
        self._last_R_body = R_body
        return self.rb.dynamics(t, x, lambda _t, _x: (F_body, tau_body))

    def step(self, u: ControlInput, dt: float | None = None) -> np.ndarray:
        """Advance dynamics by dt (default self.dt) using RK4."""
        h = self.dt if dt is None else dt
        x = self.x
        t = self.t
        f = self._derivative

        k1 = f(t, x, u)
        k2 = f(t + h / 2, x + h * k1 / 2, u)
        k3 = f(t + h / 2, x + h * k2 / 2, u)
        k4 = f(t + h, x + h * k3, u)

        self.x = x + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.t += h
        self.last_u = u
        return self.x

    def step_with_commands(
        self,
        omega_cmds: Sequence[float],
        servo_angles: Sequence[Sequence[float]] | None = None,
        dt: float | None = None,
    ):
        """
        Advance dynamics by dt using rotor speed commands. If an actuator model is
        configured, it integrates rotor speeds with first-order lag; otherwise it
        applies commands as-is. If a servo actuator is configured, servo_angles
        are treated as commands and integrated with second-order dynamics.
        """
        h = self.dt if dt is None else dt
        if self.actuator is None:
            omegas = omega_cmds
        else:
            omegas = self.actuator.step(omega_cmds, h)
        if self.servo_actuator is not None:
            servo_applied = self.servo_actuator.step(servo_angles, h)
        else:
            servo_applied = servo_angles if servo_angles is not None else [[] for _ in self.rotors]
        u = ControlInput(omegas=omegas, servo_angles=servo_applied)
        return self.step(u, dt=h)

    def propagate(self, controller: Callable[[float, np.ndarray], ControlInput], steps: int, dt: float | None = None):
        """
        Convenience loop: run N steps using a controller(t, x) -> ControlInput.
        Returns final state.
        """
        h = self.dt if dt is None else dt
        for _ in range(steps):
            u = controller(self.t, self.x)
            self.step(u, dt=h)
        return self.x

    def snapshot(self):
        """
        Snapshot compatible with viewer/server payloads.
        Includes body pose and per-rotor thrust/torque in both frames.
        """
        p_body, _, q_body = state_to_body_pose_world(self.x)
        rotors_out = []
        R_body = self._last_R_body
        for i, (p_world, q_world, _R_world_rotor, F_body, tau_body, _R_rotor_body) in enumerate(
            self._last_rotor_world
        ):
            F_world = R_body @ F_body
            tau_world = R_body @ tau_body
            rotors_out.append(
                {
                    "id": i,
                    "position_world": p_world.tolist(),
                    "quaternion_world": q_world.tolist(),
                    "thrust_world": F_world.tolist(),
                    "thrust_body": F_body.tolist(),
                    "torque_world": tau_world.tolist(),
                    "torque_body": tau_body.tolist(),
                }
            )

        return {
            "t": self.t,
            "body": {"position": p_body.tolist(), "quaternion": q_body.tolist()},
            "rotors": rotors_out,
        }


def build_dynamics(
    config_path: str | Path,
    dt: float = 0.005,
    rotor_time_constant: float | Sequence[float] | None = None,
    omega_min: float | Sequence[float] = 0.0,
    omega_max: float | Sequence[float] | None = None,
    servo_omega_n: float | Sequence[float] | None = None,
    servo_zeta: float | Sequence[float] = 0.7,
    servo_angle_min: float | Sequence[float] | None = None,
    servo_angle_max: float | Sequence[float] | None = None,
    ground_height: float | None = None,
    ground_k: float = 0.0,
    ground_d: float = 0.0,
) -> DynamicsModel:
    """
    Construct a DynamicsModel from a JSON config path.
    - If rotor_time_constant is provided, a first-order actuator model will be
      used to integrate rotor speeds in step_with_commands.
    - If servo_omega_n is provided, a damped second-order actuator will integrate
      servo angles in step_with_commands.
    """
    vehicle_cfg, rotors = load_vehicle_config(Path(config_path))
    actuator = None
    if rotor_time_constant is not None:
        actuator = RotorFirstOrderActuator(
            rotor_count=len(rotors),
            tau=rotor_time_constant,
            omega_min=omega_min,
            omega_max=omega_max,
        )
    servo_actuator = None
    if servo_omega_n is not None:
        servo_counts = [len(r.config.tilt_axes()) for r in rotors]
        total_servos = sum(servo_counts)
        if total_servos > 0:
            servo_actuator = ServoSecondOrderActuator(
                servo_counts=servo_counts,
                omega_n=servo_omega_n,
                zeta=servo_zeta,
                angle_min=servo_angle_min,
                angle_max=servo_angle_max,
            )
    return DynamicsModel(
        vehicle_cfg,
        rotors,
        dt=dt,
        actuator=actuator,
        servo_actuator=servo_actuator,
        ground_height=ground_height,
        ground_k=ground_k,
        ground_d=ground_d,
    )
