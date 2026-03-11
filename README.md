Logic-Agent
=================

Minimal README for the project. Contains installation and quickstart notes.

Installation
------------

- Create a virtual environment and install runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

- Install developer tools (optional):

```bash
pip install -r requirements-dev.txt
```

Quickstart (example)
---------------------

- Prepare model weights and set environment variables (example):

```bash
export LOGIC_DATA_ROOT=/path/to/data
export DINO_MODEL=/path/to/dinov3
export SAM2_CKPT=/path/to/sam_checkpoint.pt
```

- Run the small demo scripts (CPU/GPU as available):

```bash
python src/logic_agent/ape.py
python src/logic_agent/model/segment.py
```

Training with DataParallel (multi-GPU)
--------------------------------------

- The training script supports `torch.nn.DataParallel` with `--data_parallel`.
- Example (use physical GPUs 2 and 3):

```bash
CUDA_VISIBLE_DEVICES=2,3 python -m logic_agent.run \
	--category juice_bottle \
	--data_root /path/to/mvtec_loco_ad \
	--data_parallel
```

- If multiple GPUs are visible, startup logs print `Using DataParallel on N GPUs`.

Notes
-----
- This project expects large pretrained weights (DINOv3 / SAM2). Replace absolute paths in code with environment variables or pass paths via a config.
- See `src/logic_agent` for modules and utilities.
