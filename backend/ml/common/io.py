"""Saving model metadata and metric reports in a single consistent format."""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from ml.paths import report_dir


def write_meta(model_path: Path, meta: dict) -> None:
    meta = {**meta, "trained_at": datetime.now(timezone.utc).isoformat(),
            "environment": {"python": platform.python_version(), "platform": platform.platform()}}
    (model_path / "meta.json").write_text(json.dumps(meta, indent=2))
    task = meta["task"]
    (report_dir(task) / f"{meta['name']}_metrics.json").write_text(json.dumps(meta, indent=2))


def to_jsonable(o):
    import numpy as np
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o
