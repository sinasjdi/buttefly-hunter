"""
Closed-loop geometric-ish hover/position demo with tilt-aware allocation.
Runs the dynamics with rotor lag + servo dynamics, uses a simple position/velocity
controller to produce a desired force in body frame, and a rate PID to align attitude.
Then a tilt-aware allocator maps desired wrench -> per-rotor thrust/servo.

Connect the Three.js viewer to ws://localhost:8765 to visualize.
Logs states/controls to logs/<timestamp>/closedloop_geometric.csv.
"""

import asyncio
import json
from pathlib import Path

import numpy as np
import websockets

from .config_loader import load_vehicle_config
from .lib_control import TiltAllocator
from .lib_dynamics import build_dynamics
from .logging_utils import StepLogger
from .rigid_body import quat_normalize, quat_to_rotmat

# Configuration
CONFIG_PATH = Path(__file__).parent / "config" / "vehicle_example.json"
CONTROL_PATH = Path(__file__).parent / "config" / "demo_controls.json"
WS_PORT = 8765
DT = 0.01
ROTOR_TIME_CONSTANT = 0.05
SERVO_OMEGA_N = 50.0
SERVO_ZETA = 0.7
GROUND_HEIGHT = 0.0
GROUND_K = 2000.0
GROUND_D = 80.0

POS_KP = np.array([5.0, 5.0, 10.0])
VEL_KD = np.array([4.0, 4.0, 8.0])
POS_KI = np.array([1.5, 1.5, 2.5])
ATT_KR = np.array([8.0, 8.0, 4.0])  # attitude error gains
ATT_KW = np.array([1.2, 1.2, 0.7])  # rate damping
ATT_KI = np.array([0.5, 0.5, 0.3])  # attitude integral gains

DEFAULT_POS = np.array([0.0, 0.0, 1.0])
DEFAULT_YAW = 0.0  # rad


def _normalize(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-9)


def body_rotation_from_quat(q):
    return quat_to_rotmat(quat_normalize(q))


def control_outer(p, v, R_body, mass, gravity, target_pos, e_int):
    """Position/velocity PID -> desired force in world."""
    e_p = target_pos - p
    e_v = -v
    F_world = POS_KP * e_p + VEL_KD * e_v + POS_KI * e_int + np.array([0.0, 0.0, mass * gravity])
    return F_world, e_p


def desired_rotation_from_force(F_world, yaw_des):
    """Construct desired rotation: body z aligns with force; yaw about world z is yaw_des."""
    b3 = _normalize(F_world)
    # Desired heading vector in world XY
    b1d = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])
    b2 = _normalize(np.cross(b3, b1d))
    if np.linalg.norm(b2) < 1e-6:
        # F nearly vertical and aligned with b1d; pick orthogonal
        b2 = np.array([0.0, 1.0, 0.0])
    b1 = np.cross(b2, b3)
    R_des = np.stack([b1, b2, b3], axis=1)
    return R_des


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


def attitude_control(R_body, w_body, R_des):
    """Geometric attitude PD on SO(3)."""
    e_R_mat = 0.5 * (R_des.T @ R_body - R_body.T @ R_des)
    e_R = np.array([e_R_mat[2, 1], e_R_mat[0, 2], e_R_mat[1, 0]])
    tau = -ATT_KR * e_R - ATT_KW * w_body
    return tau, e_R


def clamp_vec(v, limit):
    """Clamp each element of v to +/- limit."""
    v = np.asarray(v, dtype=float)
    return np.clip(v, -limit, limit)


def load_controls():
    """Optional override from CONTROL_PATH: target position under key 'target_pos'."""
    if not CONTROL_PATH.exists():
        return None
    try:
        return json.loads(CONTROL_PATH.read_text())
    except Exception:
        return None


class SetpointManager:
    """
    Handles setpoint scheduling from control file.
    Supports:
      - Single target_pos / target_yaw_deg
      - Single target_rpy_deg (overrides yaw)
      - Waypoints: list of {pos:[x,y,z], yaw_deg:float, rpy_deg:[r,p,y], duration:float}, optional loop
    """

    def __init__(self):
        self.waypoints = [{"pos": DEFAULT_POS.tolist(), "yaw": DEFAULT_YAW, "rpy": None, "duration": 1e9}]
        self.loop = False
        self.start_t = 0.0
        self.e_int = np.zeros(3)
        self.e_att_int = np.zeros(3)

    def load_from_ctrl(self, ctrl, t_now: float):
        if not ctrl:
            return
        if "waypoints" in ctrl and isinstance(ctrl["waypoints"], list):
            wps = []
            for wp in ctrl["waypoints"]:
                pos = np.array(wp.get("pos", DEFAULT_POS), dtype=float)
                yaw = float(wp.get("yaw_deg", DEFAULT_YAW * 180.0 / np.pi)) * np.pi / 180.0
                rpy = wp.get("rpy_deg", None)
                if rpy is not None:
                    rpy = [float(r) * np.pi / 180.0 for r in rpy]
                dur = float(wp.get("duration", 3.0))
                wps.append({"pos": pos, "yaw": yaw, "rpy": rpy, "duration": dur})
            if wps:
                self.waypoints = wps
                self.loop = bool(ctrl.get("loop_waypoints", False))
                self.start_t = t_now
                self.e_int[:] = 0.0
                self.e_att_int[:] = 0.0
                return
        # Fallback to single target
        if "target_pos" in ctrl or "target_yaw_deg" in ctrl or "target_rpy_deg" in ctrl:
            pos = np.array(ctrl.get("target_pos", DEFAULT_POS), dtype=float)
            yaw = float(ctrl.get("target_yaw_deg", DEFAULT_YAW * 180.0 / np.pi)) * np.pi / 180.0
            rpy = ctrl.get("target_rpy_deg", None)
            if rpy is not None:
                rpy = [float(r) * np.pi / 180.0 for r in rpy]
            self.waypoints = [{"pos": pos, "yaw": yaw, "rpy": rpy, "duration": 1e9}]
            self.loop = False
            self.start_t = t_now
            self.e_int[:] = 0.0
            self.e_att_int[:] = 0.0

    def current(self, t_now: float):
        elapsed = t_now - self.start_t
        idx = 0
        acc = 0.0
        while idx < len(self.waypoints):
            dur = self.waypoints[idx]["duration"]
            if elapsed < acc + dur:
                wp = self.waypoints[idx]
                return wp["pos"], wp["yaw"], wp["rpy"]
            acc += dur
            idx += 1
        # End of list: hold last or loop
        if self.loop and self.waypoints:
            self.start_t = t_now
            wp = self.waypoints[0]
            return wp["pos"], wp["yaw"], wp["rpy"]
        wp = self.waypoints[-1]
        return wp["pos"], wp["yaw"], wp["rpy"]


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
    alloc = TiltAllocator(rotors, thrust_max=30.0, tilt_limit=np.deg2rad(45), reg=1e-6, yaw_drag_scale=2.0)
    logger = StepLogger(rotors, name="closedloop_geometric")
    sp_manager = SetpointManager()

    async def handler(websocket):
        while True:
            ctrl = load_controls()
            sp_manager.load_from_ctrl(ctrl, dyn.t)
            target_pos, target_yaw, target_rpy = sp_manager.current(dyn.t)

            x = dyn.x
            p = x[0:3]
            v = x[3:6]
            q = x[6:10]
            R_body = body_rotation_from_quat(q)

            # Outer loop: desired world force
            F_world_des, e_p = control_outer(p, v, R_body, dyn.vehicle_cfg.mass, dyn.vehicle_cfg.gravity, target_pos, sp_manager.e_int)
            # Integrate position error with clamp
            sp_manager.e_int += e_p * dt
            sp_manager.e_int = clamp_vec(sp_manager.e_int, limit=5.0)
            # Desired orientation aligns body z with F_world_des and sets yaw unless rpy target provided
            if target_rpy is not None:
                R_des = rotation_from_rpy(*target_rpy)
            else:
                R_des = desired_rotation_from_force(F_world_des, target_yaw)
            tau_body_des, e_R = attitude_control(R_body, x[10:13], R_des)
            sp_manager.e_att_int += e_R * dt
            sp_manager.e_att_int = clamp_vec(sp_manager.e_att_int, limit=0.5)
            tau_body_des += -ATT_KI * sp_manager.e_att_int
            F_body_des = R_body.T @ F_world_des

            wrench_des = np.concatenate([F_body_des, tau_body_des])
            u = alloc.allocate(wrench_des=wrench_des, desired_dir_body=None)

            dyn.step_with_commands(u.omegas, u.servo_angles, dt=dt)
            logger.log(dyn.t, dyn.x, dyn.last_u)
            await websocket.send(json.dumps(dyn.snapshot()))
            await asyncio.sleep(dt)

    async with websockets.serve(handler, "localhost", port):
        print(f"Closed-loop geometric demo at ws://localhost:{port} (config={config_path})")
        print("Edit target_pos/target_yaw_deg/target_rpy_deg or waypoints in sim/config/demo_controls.json to move the setpoint.")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(run_server(CONFIG_PATH))
