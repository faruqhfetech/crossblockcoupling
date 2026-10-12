"""Download WikiText-103 to DATA_ROOT/wikitext103, train a SentencePiece BPE, encode to .npy."""
import argparse
import hashlib
import json
import sys
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from cbfc import env

MIRRORS = ["https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip",
           "https://data.deepai.org/wikitext-103-raw.zip"]
CHUNK_BYTES = 1 << 20


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def download(url, dst):
    request = urllib.request.Request(url, headers={"User-Agent": "cbfc-lm-data/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, open(dst, "wb") as f:
        while True:
            block = response.read(CHUNK_BYTES)
            if not block:
                break
            f.write(block)


def check_archive_paths(archive, root):
    """Refuse archives whose entries would extract outside root."""
    for member in archive.infolist():
        target = (root / member.filename).resolve()
        inside = root in target.parents or target == root
        if not inside:
            raise RuntimeError(f"unsafe path in archive: {member.filename}")


def extract(archive_path, out):
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(archive_path) as archive:
        root = Path(tmp).resolve()
        check_archive_paths(archive, root)
        archive.extractall(tmp)
        for split in ("train", "valid", "test"):
            hits = list(Path(tmp).rglob(f"wiki.{split}.raw"))
            if not hits:
                hits = list(Path(tmp).rglob(f"wiki.{split}.tokens"))
            if not hits:
                raise FileNotFoundError(f"wiki.{split} not in archive")
            text = hits[0].read_text(encoding="utf-8").replace("\r\n", "\n")
            (out / f"wiki.{split}.txt").write_text(text, encoding="utf-8", newline="\n")


def encode(out, chunk_lines=100_000):
    import sentencepiece as spm
    tokenizer = spm.SentencePieceProcessor(model_file=str(out / "wiki.model"))
    for split in ("train", "valid"):
        lines = (out / f"wiki.{split}.txt").read_text(encoding="utf-8").splitlines()
        pieces = []
        for start in range(0, len(lines), chunk_lines):
            encoded = tokenizer.encode(lines[start:start + chunk_lines], out_type=int)
            line_arrays = []
            for ids in encoded:
                if ids:
                    line_arrays.append(np.array(ids + [tokenizer.eos_id()], dtype=np.uint16))
            pieces.append(np.concatenate(line_arrays))
        np.save(out / f"wiki.{split}.npy", np.concatenate(pieces))


def download_first_working(urls, archive):
    errors = []
    for url in urls:
        try:
            print("downloading", url)
            download(url, archive)
            return
        except Exception as e:
            errors.append(f"{url}: {e}")
            archive.unlink(missing_ok=True)
    raise SystemExit("all downloads failed:\n" + "\n".join(errors))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None)
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--tokenizer", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--encode", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = env.data_root() / "wikitext103"
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "wikitext-103-raw.zip"
    urls = [a.url] if a.url else MIRRORS

    if a.force or not (out / "wiki.train.txt").exists():
        if a.force or not archive.exists():
            download_first_working(urls, archive)
        extract(archive, out)
    if a.tokenizer and (a.force or not (out / "wiki.model").exists()):
        import sentencepiece as spm
        spm.SentencePieceTrainer.train(input=str(out / "wiki.train.txt"), model_prefix=str(out / "wiki"),
                                       vocab_size=a.vocab_size, model_type="bpe", character_coverage=1.0,
                                       input_sentence_size=2_000_000, shuffle_input_sentence=True,
                                       pad_id=0, bos_id=1, eos_id=2, unk_id=3)
    if a.encode and a.tokenizer:
        encode(out)

    hashes = {}
    for path in sorted(out.iterdir()):
        if path.suffix in (".txt", ".model", ".npy"):
            hashes[path.name] = sha256(path)
    meta = dict(created=datetime.now(timezone.utc).isoformat(), sources=urls, files=hashes)
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("ready:", out)


if __name__ == "__main__":
    main()
