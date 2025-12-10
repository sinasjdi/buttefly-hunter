"""
Kinematic demo: move the UAV on a simple trajectory (no dynamics) and stream poses
so you can verify visualization updates. This does not integrate forces; it just
sets position/orientation over time.
"""

import asyncio
import json
from math import cos, sin
from pathlib import Path

import numpy as np
import websockets

from .config_loader import load_vehicle_config
from .lib_geometry import GeometryModel
from .plant import MultiTiltRotorPlant
from .rigid_body import rotmat_to_quat

# Optional manual overrides (edit for quick tests)
MANUAL_SERVO_ANGLES = None  # e.g., [[0.2], [-0.1], [0.3], [0.0]]
MANUAL_THRUST_CMDS = None   # e.g., [0.8, 0.8, 0.8, 0.8]


def body_pose(t: float):
    """Simple circular path with slow yaw, Z-up world."""
    radius = 1.0
    altitude = 0.3
    yaw_rate = 0.8  # rad/s

    x = radius * cos(0.8 * t)
    y = radius * sin(0.8 * t)
    z = altitude
    yaw = yaw_rate * t

    R = np.array(
        [
            [cos(yaw), -sin(yaw), 0.0],
            [sin(yaw), cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    q = rotmat_to_quat(R)
    return np.array([x, y, z]), q, R


def snapshot_at(t: float, plant: MultiTiltRotorPlant):
    p_body, q_body, R_body = body_pose(t)
    # Servo angles: manual override or oscillating demo
    if MANUAL_SERVO_ANGLES is not None:
        servo_angles = MANUAL_SERVO_ANGLES
    else:
        servo_angles = []
        for rotor_idx, rotor in enumerate(plant.rotors):
            angles = []
            for axis_idx, _ in enumerate(rotor.config.tilt_axes()):
                angles.append(0.6 * sin(0.6 * t + 0.4 * rotor_idx + 0.2 * axis_idx))
            servo_angles.append(angles)

    # Thrust commands: manual override or oscillating demo
    if MANUAL_THRUST_CMDS is not None:
        thrust_cmds = MANUAL_THRUST_CMDS
    else:
        thrust_cmds = [0.5 + 0.5 * sin(0.5 * t + 0.3 * i) for i in range(len(plant.rotors))]

    # Map normalized thrust command to omega via k_thrust * omega^2 = cmd (unitless scaling)
    omegas = []
    for cmd, rotor in zip(thrust_cmds, plant.rotors):
        k = rotor.config.k_thrust if rotor.config.k_thrust > 1e-6 else 1.0
        omegas.append((cmd / k) ** 0.5)
    geom = GeometryModel(plant.rotors)
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


async def kinematic_server(config_path: Path, port: int = 8765, dt: float = 0.02):
    _, rotors = load_vehicle_config(config_path)
    plant = MultiTiltRotorPlant(rotors)
    t = 0.0

    async def handler(websocket):
        nonlocal t
        try:
            while True:
                snap = snapshot_at(t, plant)
                await websocket.send(json.dumps(snap))
                t += dt
                await asyncio.sleep(dt)
        except websockets.ConnectionClosed:
            return

    async with websockets.serve(handler, "localhost", port):
        print(f"Kinematic demo running at ws://localhost:{port} (config={config_path})")
        await asyncio.Future()


if __name__ == "__main__":
    config = Path(__file__).parent / "config" / "vehicle_example.json"
    asyncio.run(kinematic_server(config))
