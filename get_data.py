"""Download MNIST, CIFAR-10 and CIFAR-100 from the Hugging Face CDN into DATA_ROOT/<name>.npz.

  python get_data.py                 # all three
  python get_data.py cifar10         # one
"""
import io
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

from cbfc import env

BASE = "https://huggingface.co/datasets/{}/resolve/refs%2Fconvert%2Fparquet/{}/{}/0000.parquet"
SOURCES = {"mnist": ("ylecun/mnist", "mnist", "image", "label"),
           "cifar10": ("uoft-cs/cifar10", "plain_text", "img", "label"),
           "cifar100": ("uoft-cs/cifar100", "cifar100", "img", "fine_label")}


def fetch(name, split):
    repo, config, image_column, label_column = SOURCES[name]
    url = BASE.format(repo, config, split)
    with urllib.request.urlopen(url, timeout=120) as response:
        df = pd.read_parquet(io.BytesIO(response.read()))
    images = []
    for cell in df[image_column]:
        image = Image.open(io.BytesIO(cell["bytes"]))
        images.append(np.asarray(image))
    x = np.stack(images).astype(np.uint8)
    y = df[label_column].to_numpy(np.int64)
    return x, y


def fetch_job(job):
    name, split = job
    return fetch(name, split)


def main():
    names = sys.argv[1:] or list(SOURCES)
    root = env.data_root()
    root.mkdir(parents=True, exist_ok=True)
    jobs = [(name, split) for name in names for split in ("train", "test")]
    with ThreadPoolExecutor(len(jobs)) as pool:
        results = dict(zip(jobs, pool.map(fetch_job, jobs)))
    for name in names:
        x_train, y_train = results[(name, "train")]
        x_test, y_test = results[(name, "test")]
        path = root / f"{name}.npz"
        np.savez(path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        print(f"{name}: train {x_train.shape}, test {x_test.shape} -> {path}")


if __name__ == "__main__":
    main()
