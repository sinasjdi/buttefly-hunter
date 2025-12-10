import asyncio
import json
from pathlib import Path

import numpy as np
import websockets

from .config_loader import load_vehicle_config
from .lib_geometry import GeometryModel
from .plant import ControlInput, MultiTiltRotorPlant, state_to_body_pose_world
from .rigid_body import RigidBody6DOF, RigidBodyParams

DT = 0.005
VIS_EVERY = 5  # stream every 5 steps -> ~40 Hz at DT=5 ms for smoother viz, lighter on the viewer


def hover_controller(mass, rotor_count):
    def _ctrl(t, x):
        thrust_per_rotor = (mass * 9.81) / rotor_count
        omega = np.sqrt(thrust_per_rotor / 1.0)
        return ControlInput(
            omegas=[omega] * rotor_count,
            servo_angles=[[0.0] * 0] * rotor_count,  # empty servo list per rotor
        )

    return _ctrl


class Simulator:
    def __init__(self, rb: RigidBody6DOF, plant: MultiTiltRotorPlant, controller):
        self.rb = rb
        self.plant = plant
        self.controller = controller
        self.x = self.initial_state()
        self.t = 0.0
        self.last_rotor_world = []
        self.last_R_body = np.eye(3)

    def initial_state(self):
        p0 = np.array([0.0, 0.0, 0.0])
        v0 = np.zeros(3)
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        w0 = np.zeros(3)
        return np.concatenate([p0, v0, q0, w0])

    def _derivative(self, t, x):
        u = self.controller(t, x)
        F_body, tau_body, rotor_world = self.plant.force_torque(t, x, u)
        self.last_rotor_world = rotor_world
        _, R_body, _ = state_to_body_pose_world(x)
        self.last_R_body = R_body
        return self.rb.dynamics(t, x, lambda _t, _x: (F_body, tau_body))

    def step(self, dt):
        x = self.x
        t = self.t
        f = self._derivative

        k1 = f(t, x)
        k2 = f(t + dt / 2, x + dt * k1 / 2)
        k3 = f(t + dt / 2, x + dt * k2 / 2)
        k4 = f(t + dt, x + dt * k3)
        self.x = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.t += dt

    def snapshot(self):
        p_body, _, q_body = state_to_body_pose_world(self.x)
        rotors = []
        R_body = self.last_R_body
        for i, (p_world, q_world, _R_world_rotor, F_body, tau_body, R_rotor_body) in enumerate(
            self.last_rotor_world
        ):
            F_world = R_body @ F_body
            tau_world = R_body @ tau_body
            rotors.append(
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
            "body": {
                "position": p_body.tolist(),
                "quaternion": q_body.tolist(),
            },
            "rotors": rotors,
        }


async def sim_loop(websocket):
    config_path = Path(__file__).parent / "config" / "vehicle_example.json"
    vehicle_cfg, rotors = load_vehicle_config(config_path)
    rb = RigidBody6DOF(
        RigidBodyParams(
            mass=vehicle_cfg.mass,
            inertia=vehicle_cfg.inertia,
            gravity=vehicle_cfg.gravity,
        )
    )
    plant = MultiTiltRotorPlant(rotors)
    geometry = GeometryModel(rotors)
    controller = hover_controller(rb.params.mass, rotor_count=len(rotors))
    sim = Simulator(rb, plant, controller)

    step_count = 0
    while True:
        sim.step(DT)
        step_count += 1
        if step_count % VIS_EVERY == 0:
            try:
                await websocket.send(json.dumps(sim.snapshot()))
            except websockets.ConnectionClosed:
                break
        await asyncio.sleep(DT)


async def main():
    async with websockets.serve(lambda ws, path=None: sim_loop(ws), "localhost", 8765):
        print("Sim server running at ws://localhost:8765")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
