"""
Geometry-only demo: uses lib_geometry to compute rotor world poses/thrust vectors
and streams them over WebSocket. No physics integration; body pose and servo/thrust
inputs are defined here so you can drive the viewer without client-side overrides.
"""

import asyncio
import json
from math import cos, sin
from pathlib import Path
from typing import List

import numpy as np
import websockets

from .config_loader import load_vehicle_config
from .lib_geometry import GeometryModel
from .rigid_body import rotmat_to_quat

# --- Configurable inputs ---

# Start static by default; set to True to enable circular path
ENABLE_BODY_TRAJ = False
BODY_RADIUS = 0.6
BODY_ALTITUDE = 5
BODY_YAW_RATE = 0.5  # rad/s
BODY_TRAJ_RATE = 0.6  # rad/s for x/y circle

# Control file to override body/servo/thrust at runtime (edit this JSON while running)
CONTROL_PATH = Path(__file__).parent / "config" / "demo_controls.json"

# Servo tilts (per rotor) in radians; if None, uses control file (or zeros)
MANUAL_SERVO_ANGLES: List[List[float]] | None = None  # e.g., [[0.2], [-0.2], [0.1], [0.0]]

# Normalized thrust commands (0..1.5); if None, uses control file (or zeros)
MANUAL_THRUST_CMDS: List[float] | None = None

# WebSocket port
WS_PORT = 8765

# --- Helpers ---


def load_controls():
    """Load override controls from CONTROL_PATH if it exists."""
    if not CONTROL_PATH.exists():
        return None
    try:
        return json.loads(CONTROL_PATH.read_text())
    except Exception:
        return None


def body_pose(t: float):
    """Body pose in world. Returns p (3), q (4), R (3x3)."""
    if not ENABLE_BODY_TRAJ:
        p = np.array([0.0, 0.0, BODY_ALTITUDE])
        q = np.array([1.0, 0.0, 0.0, 0.0])
        R = np.eye(3)
        return p, q, R

    x = BODY_RADIUS * cos(BODY_TRAJ_RATE * t)
    y = BODY_RADIUS * sin(BODY_TRAJ_RATE * t)
    z = BODY_ALTITUDE
    yaw = BODY_YAW_RATE * t

    R = np.array(
        [
            [cos(yaw), -sin(yaw), 0.0],
            [sin(yaw), cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    q = rotmat_to_quat(R)
    return np.array([x, y, z]), q, R


def snapshot_at(t: float, geom: GeometryModel):
    ctrl = load_controls()
    # Body pose
    if ctrl and "body" in ctrl:
        bp = ctrl["body"]
        p_body = np.array(bp.get("position", [0.0, 0.0, BODY_ALTITUDE]), dtype=float)
        rpy = bp.get("rpy_deg", [0.0, 0.0, 0.0])
        roll, pitch, yaw = [v * np.pi / 180.0 for v in rpy]
        R_body = np.array(
            [
                [np.cos(yaw) * np.cos(pitch), np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll), np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)],
                [np.sin(yaw) * np.cos(pitch), np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll), np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)],
                [-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(roll)],
            ]
        )
        q_body = rotmat_to_quat(R_body)
    else:
        p_body, q_body, R_body = body_pose(t)

    # Servo angles
    if ctrl and "servo_angles_deg" in ctrl:
        servo_angles = [[ang * np.pi / 180.0 for ang in angles] for angles in ctrl["servo_angles_deg"]]
    elif MANUAL_SERVO_ANGLES is not None:
        servo_angles = MANUAL_SERVO_ANGLES
    else:
        servo_angles = [[0.0 for _ in rotor.config.tilt_axes()] for rotor in geom.rotors]

    # Thrust commands -> omegas
    if ctrl and "thrust_norm" in ctrl:
        thrust_cmds = ctrl["thrust_norm"]
    elif MANUAL_THRUST_CMDS is not None:
        thrust_cmds = MANUAL_THRUST_CMDS
    else:
        thrust_cmds = [0.0 for _ in geom.rotors]
    omegas = []
    for cmd, rotor in zip(thrust_cmds, geom.rotors):
        k = rotor.config.k_thrust if rotor.config.k_thrust > 1e-6 else 1.0
        omegas.append((cmd / k) ** 0.5)

    outputs = geom.rotor_outputs(p_body, R_body, servo_angles, omegas)

    rotors_out = []
    for i, out in enumerate(outputs):
        rotors_out.append(
            {
                "id": i,
                "position_world": out["position_world"].tolist(),
                "quaternion_world": out["quaternion_world"].tolist(),
                "thrust_world": out["thrust_world"].tolist(),
                "thrust_body": out["thrust_body"].tolist(),
                "torque_world": out["torque_world"].tolist(),
                "torque_body": out["torque_body"].tolist(),
            }
        )

    return {
        "t": t,
        "body": {"position": p_body.tolist(), "quaternion": q_body.tolist()},
        "rotors": rotors_out,
    }


def control_signature(ctrl):
    if ctrl is None:
        return "none"
    return json.dumps(ctrl, sort_keys=True)


async def geometry_server(config_path: Path, port: int = WS_PORT, dt: float = 0.02):
    _, rotors = load_vehicle_config(config_path)
    geom = GeometryModel(rotors)
    t = 0.0
    last_sig = None

    async def handler(websocket):
        nonlocal t, last_sig
        try:
            while True:
                ctrl = load_controls()
                sig = control_signature(ctrl)
                if sig != last_sig:
                    print(f"[demo_geometry] applied controls: {sig}")
                    last_sig = sig
                snap = snapshot_at(t, geom)
                await websocket.send(json.dumps(snap))
                t += dt
                await asyncio.sleep(dt)
        except websockets.ConnectionClosed:
            return

    async with websockets.serve(handler, "localhost", port):
        print(f"Geometry demo running at ws://localhost:{port} (config={config_path})")
        print(f"Edit controls in {CONTROL_PATH} (body position/RPY deg, servo_angles_deg, thrust_norm)")
        await asyncio.Future()


if __name__ == "__main__":
    config = Path(__file__).parent / "config" / "vehicle_example.json"
    asyncio.run(geometry_server(config))
def load_controls():
    """Load override controls from CONTROL_PATH if it exists."""
    if not CONTROL_PATH.exists():
        return None
    try:
        return json.loads(CONTROL_PATH.read_text())
    except Exception:
        return None
