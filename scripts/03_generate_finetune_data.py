import _bootstrap  # noqa: F401

from src.config import ensure_dirs
from src.finetuning.prepare_dataset import prepare_finetune_dataset

if __name__ == "__main__":
    ensure_dirs()
    prepare_finetune_dataset()
    print("Fine-tuning dataset generated.")
