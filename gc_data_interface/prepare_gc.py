"""Audit the raw data and export one example plus a JSON contract."""
import argparse
import json
from pathlib import Path
import numpy as np
from gc_dataset import GCStore, GCDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Raw dataset folder containing Annotation and Frame")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "preview")
    args = parser.parse_args()
    store = GCStore(args.root)
    args.output.mkdir(parents=True, exist_ok=True)
    report = store.report()
    report["example_fields"] = {}
    for split in ("train", "val", "test"):
        dataset = GCDataset(store, split)
        if not len(dataset):
            raise ValueError(f"No valid samples in {split}; review the split/window policy")
        sample = dataset[0]
        np.savez_compressed(args.output / f"{split}_sample.npz", **sample)
        report["example_fields"][split] = {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            if isinstance(value, (np.ndarray, np.generic)) else {"type": type(value).__name__}
            for key, value in sample.items()
        }
    path = args.output / "manifest.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "example_fields"}, ensure_ascii=False, indent=2))
    print(f"Saved manifest and three sample NPZ files to {args.output}")


if __name__ == "__main__":
    main()
