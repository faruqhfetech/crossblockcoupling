import csv
import json
import os
from pathlib import Path

import pandas as pd


def write_rows(path, rows):
    column_names = []
    for row in rows:
        for key in row:
            if key not in column_names:
                column_names.append(key)
    tmp_path = Path(str(path) + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def load_runs(*dirs):
    """Every run CSV under dirs (recursive), with its JSON config as columns."""
    frames = []
    for directory in dirs:
        for csv_path in sorted(Path(directory).rglob("*.csv")):
            config_path = csv_path.with_suffix(".json")
            if not config_path.exists():
                continue
            df = pd.read_csv(csv_path)
            config = json.loads(config_path.read_text())
            for key, value in config.items():
                if key in df:
                    continue
                if isinstance(value, list):
                    value = str(value)
                df[key] = value
            df["complete"] = csv_path.with_suffix(".done").exists()
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
