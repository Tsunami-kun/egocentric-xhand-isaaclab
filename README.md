# Egocentric XHand Isaac Lab

Standalone Isaac Lab environment for dual-arm UR5E + XHand manipulation with a robot-head egocentric RGB-D camera.

## Contents

- `xhand_isaaclab_env/`: environment package and smoke test script
- `assets/`: URDF, USD, mesh, and object assets required by the environment

## Requirements

- Isaac Lab / Isaac Sim installed locally
- Python packages available in the Isaac Lab runtime:
  - `numpy`
  - `scipy`
  - `Pillow`

## Run

```bash
cd /path/to/IsaacLab
./isaaclab.sh -p /path/to/egocentric-xhand-isaaclab/xhand_isaaclab_env/env.py
```

Headless smoke test:

```bash
cd /path/to/IsaacLab
./isaaclab.sh -p /path/to/egocentric-xhand-isaaclab/xhand_isaaclab_env/test_env.py --headless
```

## Notes

- Assets are referenced relative to the repository root via `assets/`.
- The environment uses a single egocentric main camera intended as the model's primary observation.
- Generated screenshots from the smoke test are written to `xhand_isaaclab_env/test_output/`.
