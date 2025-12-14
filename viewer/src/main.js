import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x060606);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.05, 200);
camera.position.set(3, 3, 3);
camera.up.set(0, 0, 1); // Z-up

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.update();

const legend = document.createElement("div");
legend.style.position = "absolute";
legend.style.top = "10px";
legend.style.left = "10px";
legend.style.padding = "8px 10px";
legend.style.background = "rgba(0,0,0,0.6)";
legend.style.color = "#e0e0e0";
legend.style.fontFamily = "Arial, sans-serif";
legend.style.fontSize = "12px";
legend.style.lineHeight = "16px";
legend.style.borderRadius = "4px";
legend.innerHTML =
  "World: Z-up<br/>Axes colors: <span style='color:#ff0000'>X=red</span>, <span style='color:#00ff00'>Y=green</span>, <span style='color:#0000ff'>Z=blue</span><br/>Thrust arrows: <span style='color:#00ffff'>cyan</span>";
document.body.appendChild(legend);

const labelToggle = document.createElement("div");
labelToggle.style.position = "absolute";
labelToggle.style.top = "10px";
labelToggle.style.right = "10px";
labelToggle.style.padding = "6px 8px";
labelToggle.style.background = "rgba(0,0,0,0.6)";
labelToggle.style.color = "#e0e0e0";
labelToggle.style.fontFamily = "Arial, sans-serif";
labelToggle.style.fontSize = "12px";
labelToggle.style.borderRadius = "4px";
labelToggle.style.cursor = "pointer";
labelToggle.textContent = "Toggle text";
document.body.appendChild(labelToggle);

const thrustToggle = document.createElement("div");
thrustToggle.style.position = "absolute";
thrustToggle.style.top = "40px";
thrustToggle.style.right = "10px";
thrustToggle.style.padding = "6px 8px";
thrustToggle.style.background = "rgba(0,0,0,0.6)";
thrustToggle.style.color = "#e0e0e0";
thrustToggle.style.fontFamily = "Arial, sans-serif";
thrustToggle.style.fontSize = "12px";
thrustToggle.style.borderRadius = "4px";
thrustToggle.style.cursor = "pointer";
thrustToggle.textContent = "Toggle thrust arrows";
document.body.appendChild(thrustToggle);

const netToggle = document.createElement("div");
netToggle.style.position = "absolute";
netToggle.style.top = "70px";
netToggle.style.right = "10px";
netToggle.style.padding = "6px 8px";
netToggle.style.background = "rgba(0,0,0,0.6)";
netToggle.style.color = "#e0e0e0";
netToggle.style.fontFamily = "Arial, sans-serif";
netToggle.style.fontSize = "12px";
netToggle.style.borderRadius = "4px";
netToggle.style.cursor = "pointer";
netToggle.textContent = "Toggle net thrust";
document.body.appendChild(netToggle);

const torqueToggle = document.createElement("div");
torqueToggle.style.position = "absolute";
torqueToggle.style.top = "100px";
torqueToggle.style.right = "10px";
torqueToggle.style.padding = "6px 8px";
torqueToggle.style.background = "rgba(0,0,0,0.6)";
torqueToggle.style.color = "#e0e0e0";
torqueToggle.style.fontFamily = "Arial, sans-serif";
torqueToggle.style.fontSize = "12px";
torqueToggle.style.borderRadius = "4px";
torqueToggle.style.cursor = "pointer";
torqueToggle.textContent = "Toggle torques";
document.body.appendChild(torqueToggle);

const grid = new THREE.GridHelper(10, 20);
grid.rotation.x = Math.PI / 2;
scene.add(grid);
scene.add(new THREE.AxesHelper(1));

const light = new THREE.DirectionalLight(0xffffff, 1.2);
light.position.set(5, 5, 5);
scene.add(light);
scene.add(new THREE.AmbientLight(0x404040));

const bodyGroup = new THREE.Group();
bodyGroup.add(new THREE.AxesHelper(0.4));
scene.add(bodyGroup);

const rotorObjects = [];
let latestSnapshot = null;
let showLabels = false;
let showThrustMotor = true;
let showThrustBody = false;
let showNetThrust = true;
let showTorque = false;
let showNetTorque = false;

const netArrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 0), 0.0, 0xff4444);
const netTorqueArrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 0), 0.0, 0xff66ff);
scene.add(netArrow);
scene.add(netTorqueArrow);

labelToggle.onclick = () => {
  showLabels = !showLabels;
};

thrustToggle.onclick = () => {
  if (showThrustMotor && !showThrustBody) {
    showThrustMotor = false;
    showThrustBody = true;
  } else if (!showThrustMotor && showThrustBody) {
    showThrustMotor = true;
    showThrustBody = false;
  } else {
    showThrustMotor = true;
    showThrustBody = false;
  }
};

netToggle.onclick = () => {
  showNetThrust = !showNetThrust;
};

torqueToggle.onclick = () => {
  if (showTorque && !showNetTorque) {
    showNetTorque = true;
  } else if (showTorque && showNetTorque) {
    showTorque = false;
    showNetTorque = false;
  } else {
    showTorque = true;
  }
};

function createRotorObject(id) {
  const group = new THREE.Group();
  const sphereGeom = new THREE.SphereGeometry(0.03, 12, 12);
  const sphereMat = new THREE.MeshStandardMaterial({ color: 0xffffff });
  group.add(new THREE.Mesh(sphereGeom, sphereMat));
  group.add(new THREE.AxesHelper(0.2));
  const arrowMotor = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 0), 0.0, 0x00ffff);
  const arrowBody = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 0), 0.0, 0xffaa00);
  const arrowTorque = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 0), 0.0, 0xff00ff);
  group.add(arrowMotor);
  group.add(arrowBody);
  group.add(arrowTorque);
  const labelDiv = document.createElement("div");
  labelDiv.textContent = `R${id}`;
  labelDiv.style.position = "absolute";
  labelDiv.style.color = "#ffffff";
  labelDiv.style.fontSize = "12px";
  labelDiv.style.pointerEvents = "none";
  document.body.appendChild(labelDiv);

  group.userData.arrowMotor = arrowMotor;
  group.userData.arrowBody = arrowBody;
  group.userData.arrowTorque = arrowTorque;
  group.userData.label = labelDiv;
  rotorObjects[id] = group;
  scene.add(group);
}

function updateArrow(arrow, vec) {
  const mag = Math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
  if (mag < 1e-6) {
    arrow.setLength(0.0);
    return;
  }
  const dir = new THREE.Vector3(0, 0, 1);
  const scaled = Math.min(mag * 0.2, 1.0);
  arrow.setDirection(dir);
  arrow.setLength(0.05 + scaled);
}

function updateFromSnapshot(snapshot) {
  const p = snapshot.body.position;
  const q = snapshot.body.quaternion;
  bodyGroup.position.set(p[0], p[1], p[2]);
  bodyGroup.quaternion.set(q[1], q[2], q[3], q[0]);
  const bodyQuat = new THREE.Quaternion(q[1], q[2], q[3], q[0]);
  const netThrust = new THREE.Vector3(0, 0, 0);
  const netTorque = new THREE.Vector3(0, 0, 0);

  snapshot.rotors.forEach((r) => {
    if (!rotorObjects[r.id]) createRotorObject(r.id);
    const obj = rotorObjects[r.id];
    obj.position.set(r.position_world[0], r.position_world[1], r.position_world[2]);
    obj.quaternion.set(r.quaternion_world[1], r.quaternion_world[2], r.quaternion_world[3], r.quaternion_world[0]);
    obj.userData.arrowMotor.visible = showThrustMotor;
    if (showThrustMotor) {
      updateArrow(obj.userData.arrowMotor, r.thrust_world);
    }

    obj.userData.arrowBody.visible = showThrustBody;
    if (showThrustBody) {
      let thrustWorld = new THREE.Vector3(r.thrust_world[0], r.thrust_world[1], r.thrust_world[2]);
      if (r.thrust_body) {
        const thrustBody = new THREE.Vector3(r.thrust_body[0], r.thrust_body[1], r.thrust_body[2]);
        thrustWorld = thrustBody.clone().applyQuaternion(bodyQuat);
      }
      updateArrow(obj.userData.arrowBody, thrustWorld.toArray());
    }
    netThrust.x += r.thrust_world[0];
    netThrust.y += r.thrust_world[1];
    netThrust.z += r.thrust_world[2];

    obj.userData.arrowTorque.visible = showTorque && r.torque_world;
    if (showTorque && r.torque_world) {
      updateArrow(obj.userData.arrowTorque, r.torque_world);
    }
    if (r.torque_world) {
      netTorque.x += r.torque_world[0];
      netTorque.y += r.torque_world[1];
      netTorque.z += r.torque_world[2];
    }

    if (obj.userData.label) {
      const label = obj.userData.label;
      if (showLabels) {
        const magF = Math.sqrt(
          r.thrust_world[0] * r.thrust_world[0] + r.thrust_world[1] * r.thrust_world[1] + r.thrust_world[2] * r.thrust_world[2]
        );
        const magTau = r.torque_world
          ? Math.sqrt(
              r.torque_world[0] * r.torque_world[0] + r.torque_world[1] * r.torque_world[1] + r.torque_world[2] * r.torque_world[2]
            )
          : 0;
        label.textContent = `R${r.id} | F=${magF.toFixed(2)} | tau=${magTau.toFixed(2)}`;
        const proj = obj.position.clone().project(camera);
        const x = (proj.x * 0.5 + 0.5) * window.innerWidth;
        const y = (-proj.y * 0.5 + 0.5) * window.innerHeight;
        label.style.left = `${x}px`;
        label.style.top = `${y}px`;
        label.style.display = "block";
      } else {
        label.style.display = "none";
      }
    }
  });

  netArrow.visible = showNetThrust;
  if (showNetThrust) {
    const mag = netThrust.length();
    const dir = mag > 1e-6 ? netThrust.clone().normalize() : new THREE.Vector3(0, 0, 1);
    const scaled = Math.min(mag * 0.2, 1.0);
    netArrow.position.set(p[0], p[1], p[2]);
    netArrow.setDirection(dir);
    netArrow.setLength(0.05 + scaled);
  }
  netTorqueArrow.visible = showNetTorque;
  if (showNetTorque) {
    const magT = netTorque.length();
    const dirT = magT > 1e-6 ? netTorque.clone().normalize() : new THREE.Vector3(0, 0, 1);
    const scaledT = Math.min(magT * 0.2, 1.0);
    netTorqueArrow.position.set(p[0], p[1], p[2]);
    netTorqueArrow.setDirection(dirT);
    netTorqueArrow.setLength(0.05 + scaledT);
  }
}

function connectWebSocket() {
  const ws = new WebSocket("ws://localhost:8766");
  ws.onopen = () => console.log("Connected to simulation");
  ws.onmessage = (event) => {
    try {
      latestSnapshot = JSON.parse(event.data);
    } catch (err) {
      console.error("Failed to parse snapshot", err);
    }
  };
  ws.onclose = () => {
    console.warn("WebSocket closed, retrying in 1s");
    setTimeout(connectWebSocket, 1000);
  };
}
connectWebSocket();

function animate() {
  requestAnimationFrame(animate);
  if (latestSnapshot) {
    updateFromSnapshot(latestSnapshot);
    latestSnapshot = null;
  }
  renderer.render(scene, camera);
}
animate();

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
