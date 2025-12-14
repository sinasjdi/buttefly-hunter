"""
Closed-loop demo using a simple MPC (double-integrator) for position and
tilt-aware allocation for motors/servos. More stable fallback while SQP/NMPC
is being developed.
"""

import asyncio
import json
from pathlib import Path
import time

import numpy as np
import websockets

from .config_loader import load_vehicle_config
from .lib_control import TiltAllocator
from .lib_dynamics import build_dynamics
from .lib_nmpc import NMPCConfig, NMPCController
from .logging_utils import StepLogger
from .rigid_body import quat_normalize, quat_to_rotmat

CONFIG_PATH = Path(__file__).parent / "config" / "vehicle_example.json"
CONTROL_PATH = Path(__file__).parent / "config" / "demo_controls.json"
WS_PORT = 8766
DT = 0.01
ROTOR_TIME_CONSTANT = 0.05
SERVO_OMEGA_N = 50.0
SERVO_ZETA = 0.7
GROUND_HEIGHT = 0.0
GROUND_K = 2000.0
GROUND_D = 80.0

ATT_KR = np.array([6.0, 6.0, 3.0])
ATT_KW = np.array([1.0, 1.0, 0.6])
ATT_KI = np.array([0.8, 0.8, 0.5])
POS_KI = np.array([1.5, 1.5, 2.5])
POS_INT_CLAMP = 2.0


def load_controls():
    if not CONTROL_PATH.exists():
        return None
    try:
        return json.loads(CONTROL_PATH.read_text())
    except Exception:
        return None


def body_rotation_from_quat(q):
    return quat_to_rotmat(quat_normalize(q))


def rotation_from_rpy(roll, pitch, yaw):
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def desired_rotation_from_force(F_world, yaw_des):
    b3 = F_world / (np.linalg.norm(F_world) + 1e-9)
    b1d = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])
    b2 = np.cross(b3, b1d)
    if np.linalg.norm(b2) < 1e-6:
        b2 = np.array([0.0, 1.0, 0.0])
    b2 /= np.linalg.norm(b2)
    b1 = np.cross(b2, b3)
    return np.stack([b1, b2, b3], axis=1)


def attitude_control(R_body, w_body, R_des, e_att_int, dt):
    e_R_mat = 0.5 * (R_des.T @ R_body - R_body.T @ R_des)
    e_R = np.array([e_R_mat[2, 1], e_R_mat[0, 2], e_R_mat[1, 0]])
    e_att_int += e_R * dt
    tau = -ATT_KR * e_R - ATT_KW * w_body - ATT_KI * e_att_int
    return tau, e_att_int


def clamp_vec(v, limit):
    v = np.asarray(v, dtype=float)
    return np.clip(v, -limit, limit)


def quat_from_rpy(roll, pitch, yaw):
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    q = np.array([w, x, y, z], dtype=float)
    q = q / (np.linalg.norm(q) + 1e-9)
    return q


class SetpointManager:
    def __init__(self):
        self.pos = np.array([0.0, 0.0, 1.0])
        self.yaw = 0.0
        self.rpy = None
        self.e_att_int = np.zeros(3)
        self.e_pos_int = np.zeros(3)
        self._last_sig = None

    def update_from_ctrl(self, ctrl, reset_int=False):
        changed = False
        if not ctrl:
            return changed
        sig = json.dumps(ctrl, sort_keys=True)
        if sig == self._last_sig and not reset_int:
            return False
        self._last_sig = sig
        if "target_pos" in ctrl:
            self.pos = np.array(ctrl["target_pos"], dtype=float)
            changed = True
        if "target_yaw_deg" in ctrl:
            self.yaw = float(ctrl["target_yaw_deg"]) * np.pi / 180.0
            changed = True
        if "target_rpy_deg" in ctrl:
            rpy = ctrl["target_rpy_deg"]
            if rpy is None:
                self.rpy = None
            else:
                self.rpy = [float(r) * np.pi / 180.0 for r in rpy]
            changed = True
        if changed or reset_int:
            self.e_att_int[:] = 0.0
            self.e_pos_int[:] = 0.0
        if changed:
            print(f"[demo_mpc] applied setpoint: pos={self.pos}, yaw={self.yaw}, rpy={self.rpy}", flush=True)
        return changed


async def run_server(config_path: Path, port: int = WS_PORT, dt: float = DT):
    vehicle_cfg, rotors = load_vehicle_config(config_path)
    dyn = build_dynamics(
        config_path,
        dt=dt,
        rotor_time_constant=ROTOR_TIME_CONSTANT,
        servo_omega_n=SERVO_OMEGA_N,
        servo_zeta=SERVO_ZETA,
        ground_height=GROUND_HEIGHT,
        ground_k=GROUND_K,
        ground_d=GROUND_D,
    )
    nmpc_cfg = NMPCConfig(
        dt=dt,
        horizon=5,
        servo_rate=80.0,
        q_pos=450.0,
        q_vel=140.0,
        q_att=14.0,
        q_rate=4.0,
        r_thrust=0.003,
        r_servo=0.003,
        thrust_max=110.0,
        thrust_min=0.0,
        servo_min=-np.pi / 2,
        servo_max=np.pi / 2,
    )
    nmpc = NMPCController(
        mass=vehicle_cfg.mass, inertia=vehicle_cfg.inertia, gravity=vehicle_cfg.gravity, rotors=rotors, cfg=nmpc_cfg
    )
    logger = StepLogger(rotors, name="mpc")
    sp = SetpointManager()

    async def handler(websocket):
        solve_times_ms = []
        while True:
            ctrl = load_controls()
            sp.update_from_ctrl(ctrl)

            x = dyn.x
            x_nmpc = np.zeros(nmpc.nx)
            x_nmpc[0:13] = x
            if dyn.servo_actuator is not None and nmpc.ns > 0:
                angles = getattr(dyn.servo_actuator, "angle", None)
                if angles is not None and len(angles) == nmpc.ns:
                    x_nmpc[13 : 13 + nmpc.ns] = angles

            x_ref = np.zeros(nmpc.nx)
            x_ref[0:3] = sp.pos
            q_ref = quat_from_rpy(*(sp.rpy if sp.rpy is not None else [0.0, 0.0, sp.yaw]))
            x_ref[6:10] = q_ref
            x_refs = [x_ref.copy() for _ in range(nmpc.cfg.horizon + 1)]

            try:
                t0 = time.perf_counter()
                X_opt, U_opt = nmpc.solve(x_nmpc, x_refs)
                solve_ms = (time.perf_counter() - t0) * 1000.0
                solve_times_ms.append(solve_ms)
                u0 = U_opt[:, 0]
                thrusts = u0[0:nmpc.nr]
                servo_cmds = u0[nmpc.nr : nmpc.nr + nmpc.ns]
                servo_split = []
                idx = 0
                for r in rotors:
                    count = len(r.config.tilt_axes())
                    servo_split.append([float(servo_cmds[idx + j]) for j in range(count)])
                    idx += count
                omegas = []
                for thrust, rotor in zip(thrusts, rotors):
                    k = rotor.config.k_thrust if rotor.config.k_thrust > 1e-9 else 1.0
                    omegas.append(float(np.sqrt(max(thrust, 0.0) / k)))
            except Exception as e:
                print(f"[NMPC] solver failed: {e}")
                omegas = [0.0 for _ in rotors]
                servo_split = [[0.0 for _ in r.config.tilt_axes()] for r in rotors]

            dyn.step_with_commands(omegas, servo_split, dt=dt)
            if len(solve_times_ms) % 50 == 0 and solve_times_ms:
                avg_ms = sum(solve_times_ms) / len(solve_times_ms)
                print(f"[NMPC] solve avg={avg_ms:.2f} ms (n={len(solve_times_ms)})", flush=True)
            logger.log(dyn.t, dyn.x, dyn.last_u)
            await websocket.send(json.dumps(dyn.snapshot()))
            await asyncio.sleep(dt)

    async with websockets.serve(handler, "localhost", port):
        print(f"MPC demo at ws://localhost:{port} (config={config_path})", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(run_server(CONFIG_PATH))
