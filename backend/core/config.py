from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    data_dir: Path
    model_dir: Path


BASE_DIR = Path(__file__).resolve().parents[1]
SETTINGS = Settings(
    base_dir=BASE_DIR,
    data_dir=BASE_DIR / "data",
    model_dir=BASE_DIR / "models",
)
