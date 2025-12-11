"""
Open-loop dynamics demo: runs the full rigid-body dynamics with a first-order
rotor actuator model and streams snapshots over WebSocket for the viewer.

Controls are read live from a JSON file (edit while running):
  - "omega_cmd": list of commanded rotor speeds (rad/s)
  - or "thrust_norm": list of normalized thrust commands (maps to omega via k_thrust)
  - "servo_angles_deg": per-rotor servo angles in degrees
  - "body": optional initial pose override with "position" and "rpy_deg"

Defaults: hover-like command based on mg / (n * k_thrust) with zero servo angles.
"""

import asyncio
import json
from math import sqrt
from pathlib import Path
from typing import List, Sequence

import numpy as np
import websockets

from .config_loader import load_vehicle_config
from .lib_dynamics import build_dynamics
from .logging_utils import StepLogger

# Configuration
CONFIG_PATH = Path(__file__).parent / "config" / "vehicle_example.json"
CONTROL_PATH = Path(__file__).parent / "config" / "demo_controls.json"
WS_PORT = 8765
DT = 0.01  # integration and streaming step (s)
ROTOR_TIME_CONSTANT = 0.05  # first-order rotor lag (s)
SERVO_OMEGA_N = 50.0  # rad/s natural frequency (set None to disable servo dynamics)
SERVO_ZETA = 0.7      # damping ratio
GROUND_HEIGHT = 0.0   # meters; set None to disable
GROUND_K = 2000.0     # N/m spring
GROUND_D = 80.0       # N/(m/s) damping


def load_controls():
    """Load live controls from CONTROL_PATH if present, else None."""
    if not CONTROL_PATH.exists():
        return None
    try:
        return json.loads(CONTROL_PATH.read_text())
    except Exception:
        return None


def default_hover_cmd(vehicle_cfg, rotors):
    """Compute a naive hover omega per rotor using k_thrust."""
    if not rotors:
        return []
    # Assume identical k_thrust; fall back to 1.0 if zero/invalid
    k = rotors[0].config.k_thrust if rotors[0].config.k_thrust > 1e-6 else 1.0
    thrust_per_rotor = vehicle_cfg.mass * vehicle_cfg.gravity / len(rotors)
    omega_hover = sqrt(thrust_per_rotor / k)
    return [omega_hover for _ in rotors]


def servo_angles_from_rotors(rotors: Sequence, servo_angles_deg: List[List[float]] | None):
    """Normalize servo angles list to match rotor count/tilt axes (radians)."""
    angles = []
    for idx, rotor in enumerate(rotors):
        if servo_angles_deg and idx < len(servo_angles_deg):
            angles.append([a * np.pi / 180.0 for a in servo_angles_deg[idx]])
        else:
            angles.append([0.0 for _ in rotor.config.tilt_axes()])
    return angles


def omega_commands(rotors, ctrl):
    """Resolve omega commands from control file (omega_cmd or thrust_norm)."""
    if ctrl and "omega_cmd" in ctrl:
        return ctrl["omega_cmd"]
    if ctrl and "thrust_norm" in ctrl:
        cmds = ctrl["thrust_norm"]
        omegas = []
        for cmd, rotor in zip(cmds, rotors):
            k = rotor.config.k_thrust if rotor.config.k_thrust > 1e-6 else 1.0
            omegas.append(sqrt(cmd / k))
        return omegas
    return None


async def openloop_server(config_path: Path, port: int = WS_PORT, dt: float = DT):
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
    hover_omegas = default_hover_cmd(vehicle_cfg, rotors)
    logger = StepLogger(rotors, name="openloopdyn")

    async def handler(websocket):
        while True:
            ctrl = load_controls()
            # Servo angles
            servo_angles = servo_angles_from_rotors(rotors, ctrl.get("servo_angles_deg") if ctrl else None)
            # Omega commands
            omega_cmds = omega_commands(rotors, ctrl)
            if omega_cmds is None:
                omega_cmds = hover_omegas

            dyn.step_with_commands(omega_cmds, servo_angles, dt=dt)
            logger.log(dyn.t, dyn.x, dyn.last_u)
            await websocket.send(json.dumps(dyn.snapshot()))
            await asyncio.sleep(dt)

    async with websockets.serve(handler, "localhost", port):
        print(f"Open-loop dynamics running at ws://localhost:{port} (config={config_path})")
        print(f"Edit controls in {CONTROL_PATH} (omega_cmd, thrust_norm, servo_angles_deg)")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(openloop_server(CONFIG_PATH))
