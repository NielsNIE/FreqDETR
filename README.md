## FreqDETR — Quick start

This repository contains an object detection training implementation. Below are the minimal, repeatable steps to run training locally or on a GPU server.

1) Create / activate an environment (conda or venv) and install deps:

```bash
# Conda (recommended):
# scripts/setup_env.sh --conda --name frepdetr
conda activate frepdetr
pip install -r requirements.txt

# Or venv:
# scripts/setup_env.sh --venv --venv-path .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Single-GPU / debug (run from project root):

```bash
python -m src.train.trainer \
	--config configs/model/freqdetr_base.yaml \
	--data configs/dataset/data.yaml
```

3) Multi-GPU (server) with torchrun:

```bash
torchrun --nproc_per_node=NUM python run_trainer.py \
	--config configs/model/freqdetr_base.yaml \
	--data /absolute/path/to/Maize/data.yaml
```

4) Resume training:

```bash
python -m src.train.trainer --config configs/model/freqdetr_base.yaml --data configs/dataset/data.yaml --resume ./checkpoints/freqdetr_best.pt
```

See `notes.md` for a slightly longer explanation and troubleshooting tips.
