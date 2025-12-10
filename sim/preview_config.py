"""
Static preview server: publishes a single rigid-body pose with rotor gizmos derived
from the config file. Useful to verify geometry/tilt axes in the Three.js viewer
without running the full dynamics loop.
"""

import asyncio
import json
from pathlib import Path

import numpy as np
import websockets

from .config_loader import load_vehicle_config
from .lib_geometry import GeometryModel
from .plant import MultiTiltRotorPlant, state_to_body_pose_world
from .rigid_body import rotmat_to_quat


def make_snapshot(plant: MultiTiltRotorPlant, rb_pos=None, rb_quat=None):
    rb_pos = np.zeros(3) if rb_pos is None else np.asarray(rb_pos, dtype=float)
    rb_quat = np.array([1.0, 0.0, 0.0, 0.0]) if rb_quat is None else np.asarray(rb_quat, dtype=float)
    R_body = state_to_body_pose_world(
        np.concatenate([rb_pos, np.zeros(3), rb_quat, np.zeros(3)])
    )[1]

    rotors = []
    for i, rotor in enumerate(plant.rotors):
        servo_angles = [0.0 for _ in rotor.config.tilt_axes_baselink]
        p_world = rb_pos + R_body @ rotor.config.position_baselink
        R_rotor_body = rotor.rotation_body_from_servos(servo_angles)
        R_world_rotor = R_body @ R_rotor_body
        q_world_rotor = rotmat_to_quat(R_world_rotor)

        # Unit thrust vector in world frame to visualize rotor orientation.
        F_world = R_world_rotor @ np.array([0.0, 0.0, 1.0])

        rotors.append(
            {
                "id": i,
                "position_world": p_world.tolist(),
                "quaternion_world": q_world_rotor.tolist(),
                "thrust_world": F_world.tolist(),
            }
        )

    return {
        "t": 0.0,
        "body": {"position": rb_pos.tolist(), "quaternion": rb_quat.tolist()},
        "rotors": rotors,
    }


async def preview_server(config_path: Path, port: int = 8765):
    _, rotors = load_vehicle_config(config_path)
    plant = MultiTiltRotorPlant(rotors)
    geometry = GeometryModel(rotors)
    snapshot = make_snapshot(plant)
    payload = json.dumps(snapshot)

    async def handler(websocket):
        while True:
            await websocket.send(payload)
            await asyncio.sleep(0.03)  # ~33 Hz for smoother visualization with less load

    async with websockets.serve(handler, "localhost", port):
        print(f"Preview server running at ws://localhost:{port} (config={config_path})")
        await asyncio.Future()


if __name__ == "__main__":
    config = Path(__file__).parent / "config" / "vehicle_example.json"
    asyncio.run(preview_server(config))
