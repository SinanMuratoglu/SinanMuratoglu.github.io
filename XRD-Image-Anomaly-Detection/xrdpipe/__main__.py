"""Command-line entry point.

  python -m xrdpipe inspect  IMAGE.png [--out fig.png]
  python -m xrdpipe extract  --images DIR [--labels labels.csv] --out features.csv [--limit N] [--jobs N]
  python -m xrdpipe train    --features features.csv --labels labels.csv \
                             --train labels_train.csv --val labels_val.csv --test labels_test.csv --out results/
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from .features import extract_features
from .io import find_images, load_image, load_labels, load_split_ids


def _one(path):
    try:
        f, _ = extract_features(load_image(path))
        return Path(path).stem, f, None
    except Exception as e:  # keep going on a bad file, report it
        return Path(path).stem, None, repr(e)


def cmd_inspect(a):
    from .plots import diagnostic_figure
    img = load_image(a.image)
    f, d = extract_features(img)
    out = a.out or Path(a.image).with_suffix(".diag.png").name
    diagnostic_figure(img, d, f, title=Path(a.image).name, path=out)
    for k, v in f.items():
        print(f"{k:24s} {v:.4g}")
    print("figure ->", out)


def cmd_extract(a):
    imgs = find_images(a.images)
    if a.labels:
        keep = set(load_labels(a.labels).index)
        imgs = {k: v for k, v in imgs.items() if k in keep}
    paths = sorted(imgs.values())[: a.limit or None]
    out = Path(a.out)
    done = set(pd.read_csv(out, index_col=0).index.astype(str)) if out.exists() else set()
    todo = [p for p in paths if p.stem not in done]
    print(f"{len(paths)} images, {len(done)} already done, {len(todo)} to process")
    rows, errs, t0 = [], [], time.time()

    def flush():
        if rows:
            df = pd.DataFrame.from_dict(dict(rows), orient="index")
            df.index.name = "image"
            df.to_csv(out, mode="a", header=not out.exists())
            rows.clear()

    with ProcessPoolExecutor(a.jobs) as ex:
        for i, (stem, f, err) in enumerate(ex.map(_one, todo, chunksize=4), 1):
            if err:
                errs.append((stem, err))
            else:
                rows.append((stem, f))
            if i % 25 == 0 or i == len(todo):
                flush()
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(todo)}  {rate:.1f} img/s  eta {(len(todo) - i) / rate / 60:.1f} min", flush=True)
    flush()
    for s, e in errs:
        print("FAILED", s, e, file=sys.stderr)


def cmd_train(a):
    from .model import train_and_evaluate
    feats = pd.read_csv(a.features, index_col=0)
    feats.index = feats.index.astype(str)
    labels = load_labels(a.labels)
    splits = {k: load_split_ids(getattr(a, k)) for k in ("train", "val", "test")}
    report, imp = train_and_evaluate(feats, labels, splits, a.out)
    pd.set_option("display.width", 120)
    print(report.round(3).to_string(index=False))
    print("\nTop features per class:")
    for c in imp.columns:
        print(f"  {c:22s}", ", ".join(imp[c].nlargest(3).index))


def main(argv=None):
    p = argparse.ArgumentParser(prog="xrdpipe")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("inspect"); s.add_argument("image"); s.add_argument("--out")
    s.set_defaults(fn=cmd_inspect)
    s = sub.add_parser("extract"); s.add_argument("--images", required=True); s.add_argument("--labels")
    s.add_argument("--out", required=True); s.add_argument("--limit", type=int, default=0)
    s.add_argument("--jobs", type=int, default=4); s.set_defaults(fn=cmd_extract)
    s = sub.add_parser("train")
    for k in ("features", "labels", "train", "val", "test", "out"):
        s.add_argument(f"--{k}", required=True)
    s.set_defaults(fn=cmd_train)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
