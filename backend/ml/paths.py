"""Central filesystem locations for datasets, trained models and reports."""
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
DATA_CACHE_DIR = BACKEND_DIR / "data" / "hf_cache"

for _d in (MODELS_DIR, REPORTS_DIR, DATA_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def model_dir(name: str) -> Path:
    d = MODELS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def report_dir(task: str) -> Path:
    d = REPORTS_DIR / task
    d.mkdir(parents=True, exist_ok=True)
    return d
