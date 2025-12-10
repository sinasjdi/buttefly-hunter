# Tilt-Rotor UAV sandbox

Physics core in Python, Three.js viewer via Vite. The physics publishes state snapshots over WebSocket; the viewer only consumes them and draws gizmos.

## Python simulation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m sim.server
```

- WebSocket endpoint: `ws://localhost:8765`
- Configuration: `sim/config/vehicle_example.json` (baselink/body frame; rotor positions/orientations are expressed in this frame, world is Z-up)
- Rigid-body model: `sim/rigid_body.py`
- Rotor layout / force aggregation: `sim/rotors.py`, `sim/plant.py`

## Viewer (Three.js + Vite)

```bash
cd viewer
npm install
npm run dev -- --host
```

Open the printed local URL; it will connect to `ws://localhost:8765` automatically.

## How it fits together

- `sim.server` runs the dynamics at 200 Hz and streams snapshots every 20 steps (10 Hz).
- `sim/config/vehicle_example.json` describes vehicle mass/inertia/gravity and the rotor layout. Each rotor can be defined with a full 4x4 `pose_baselink` (homogeneous transform) plus tilt axes. Rotor positions/orientations/tilt axes are expressed in the baselink/body frame; world frame is Z-up.
- `sim/rotors.py` allows arbitrary rotor positions and tilt axes; extend the JSON to add/remove rotors or gimbals.
- `viewer/src/main.js` renders body axes, rotor axes, and thrust arrows (no meshes) in world coordinates.

## Next steps

- Add real controllers by implementing functions that map `(t, x)` to `ControlInput` (omegas + servo angles).
- Enforce servo limits and rotor thrust curves in `RotorModel`.
- Extend the JSON protocol if you want velocities, net forces, or desired setpoints in the viewer.

## Previewing configuration only (no dynamics)

To quickly verify a rotor layout/tilt axes without running dynamics, start the static preview server (reads `sim/config/vehicle_example.json` by default):

```bash
python -m sim.preview_config
```

It streams a fixed snapshot (body at origin, zero attitude) with rotor gizmos and unit thrust arrows along each rotor frame. Keep the Three.js viewer open to see the layout live.

## Geometry demo (scripted motion, no dynamics)

To drive the viewer with scripted body motion + servo/thrust commands computed in Python (no client-side overrides):

```bash
python -m sim.demo_geometry
```

Edit `sim/demo_geometry.py` to change body trajectory, servo angles, or thrust commands. Uses `lib_geometry` for all transforms and streams snapshots over `ws://localhost:8765`.

## Kinematic demo (moving without dynamics)

To see live motion without running dynamics, stream a scripted trajectory:

```bash
python -m sim.kinematic_demo
```

This publishes a slow circular path with yaw about Z while keeping rotors aligned to their baselink poses and zero servo tilt. Open the viewer to see continuous updates.

## Running the viewer (Three.js)

```bash
cd viewer
npm install   # no extra deps beyond three
npm run dev -- --host
```

Open the printed URL; it connects to `ws://localhost:8765`. Use the small buttons (top corners) to toggle rotor text, thrust arrows (motor/body), net thrust, and torques.

## Next steps (dynamics/control)

- Add actuator allocation: desired wrench → per-rotor thrust/tilt with limits.
- Implement a simple attitude/position controller and plug it into `sim.server` (physics loop).
- Keep `lib_geometry` for viz/config preview; rely on physics for actual thrust/torque values in snapshots.
