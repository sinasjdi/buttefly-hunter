"""
Open-loop drop demo: start the vehicle at 2 m altitude with zero thrust and
stream the falling motion over WebSocket. Ground contact uses the plant's
spring-damper model to catch the drop.

Connect the Three.js viewer to ws://localhost:8765 to visualize.
"""

import asyncio
import json
from pathlib import Path

import numpy as np
import websockets

from .lib_dynamics import build_dynamics
from .logging_utils import StepLogger

# Configuration
CONFIG_PATH = Path(__file__).parent / "config" / "vehicle_example.json"
WS_PORT = 8765
DT = 0.01
ROTOR_TIME_CONSTANT = 0.05
SERVO_OMEGA_N = 50.0
SERVO_ZETA = 0.7
GROUND_HEIGHT = 0.0
GROUND_K = 2000.0
GROUND_D = 80.0


def initial_state(z0: float = 2.0):
    p0 = np.array([0.0, 0.0, z0])
    v0 = np.zeros(3)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    w0 = np.zeros(3)
    return np.concatenate([p0, v0, q0, w0])


async def drop_server(config_path: Path, port: int = WS_PORT, dt: float = DT):
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
    dyn.reset(state=initial_state())
    logger = StepLogger(dyn.rotors, name="drop")

    async def handler(websocket):
        while True:
            # zero commands -> free fall
            dyn.step_with_commands(
                omega_cmds=[0.0 for _ in dyn.rotors],
                servo_angles=[[0.0 for _ in r.config.tilt_axes()] for r in dyn.rotors],
                dt=dt,
            )
            logger.log(dyn.t, dyn.x, dyn.last_u)
            await websocket.send(json.dumps(dyn.snapshot()))
            await asyncio.sleep(dt)

    async with websockets.serve(handler, "localhost", port):
        print(f"Drop demo running at ws://localhost:{port} (config={config_path})")
        print("Initial altitude 2 m, zero thrust; ground contact enabled.")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(drop_server(CONFIG_PATH))
