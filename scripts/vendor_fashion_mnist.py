"""Warm the offline cache for the Fashion-MNIST (and MNIST) teaching image datasets.

The course fetches Fashion-MNIST on first use and caches it under the package as an ``.npz``
(git-ignored); see :mod:`ml_course.datasets`. Run this once to populate that cache ahead of time
(e.g., before going offline or in CI setup), instead of letting the module-12 capstone trigger the
download:

    uv run python scripts/vendor_fashion_mnist.py

Pass ``--mnist`` to also warm the MNIST digits cache. Nothing is committed to the repo — the cached
arrays are git-ignored.

Data: Xiao H, Rasul K, Vollgraf R (2017), Fashion-MNIST (arXiv:1708.07747); LeCun Y et al. (1998),
MNIST (Proc. IEEE 86(11):2278-2324). Source: OpenML via scikit-learn's ``fetch_openml``.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from ml_course import datasets


def _report(name: str, X: np.ndarray, y: np.ndarray) -> None:
    """Print the shape and per-class balance of a freshly cached image dataset."""
    counts = np.bincount(y)
    print(f"\n{name}: {X.shape[0]} images x {X.shape[1]} pixels, {len(counts)} classes")
    print(f"pixel range [{X.min():.3f}, {X.max():.3f}] (dtype {X.dtype})")
    print("class counts:", counts.tolist())


def main() -> None:
    """Download (if needed) and report the cached image dataset(s)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mnist", action="store_true", help="also warm the MNIST digits cache"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    X, y = datasets.load_fashion_mnist()
    _report("Fashion-MNIST", X, y)
    print("classes:", datasets.FASHION_MNIST_CLASSES)

    Xtr, Xte, ytr, yte = datasets.fashion_mnist_subset(10000, 5000, seed=0)
    print(
        f"\nmodule-12 capstone subset (stratified, seed 0): "
        f"train {Xtr.shape[0]} / test {Xte.shape[0]}, "
        f"balanced ({np.bincount(ytr).min()}-{np.bincount(ytr).max()} per class in train)."
    )

    if args.mnist:
        Xm, ym = datasets.load_mnist()
        _report("MNIST", Xm, ym)


if __name__ == "__main__":
    main()
