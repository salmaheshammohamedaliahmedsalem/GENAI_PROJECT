import _bootstrap  # noqa: F401

from src.config import ensure_dirs
from src.evaluation.generate_report import generate_report

if __name__ == "__main__":
    ensure_dirs()
    generate_report()
    print("Evaluation report generated in outputs/evaluation/.")
