"""Top-level launcher to run trainer in a way that is compatible with torchrun.

Use:
  torchrun --nproc_per_node=4 python run_trainer.py --config ... --data ...

This avoids relative-import issues when executing the trainer as a script.
"""
from src.train.trainer import main


if __name__ == "__main__":
    main()
