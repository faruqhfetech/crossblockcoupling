"""Settings from dlmidterm/.env; shell environment variables take precedence."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def read_env_file(path):
    settings = {}
    if not path.exists():
        return settings
    for line in path.read_text(encoding="utf-8").splitlines():
        is_comment = line.lstrip().startswith("#")
        if "=" not in line or is_comment:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        settings[key] = value
    return settings


_ENV = read_env_file(ENV_FILE)


def get(key, default=None):
    if key in os.environ:
        return os.environ[key]
    return _ENV.get(key, default)


def data_root():
    if not get("DATA_ROOT"):
        raise SystemExit("DATA_ROOT is not set in .env")
    return Path(get("DATA_ROOT"))


def results_root():
    path = Path(get("RESULTS_ROOT", "results"))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def device():
    import torch
    wanted = get("DEVICE", "cuda")
    if wanted == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return wanted
