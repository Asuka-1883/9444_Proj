"""Reproducible, bounded CPU training study for the actual GC interface.

python train_audit.py --root RAW_DATA --work CACHE_DIR --output RESULTS_DIR
No test model score is used for selecting hyperparameters or checkpoints.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from gc_dataset import GCStore, GCDataset


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf8")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ForecastGRU(nn.Module):
    def __init__(self, hidden=64, neighbors=0, scale=1920., horizon=6, residual=False):
        super().__init__()
        self.scale, self.neighbor_count, self.horizon, self.residual = scale, neighbors, horizon, residual
        self.gru = nn.GRU(4, hidden, batch_first=True)
        if neighbors:
            # Lightweight masked history encoder; explicitly not a full social SOTA model.
            self.neighbor_encoder = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 32))
            self.query = nn.Linear(hidden, 32)
        self.head = nn.Sequential(nn.Linear(hidden + (32 if neighbors else 0), 128),
                                  nn.ReLU(), nn.Linear(128, horizon*2))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, obs, neighbors=None, mask=None):
        # The exported interface stays at scale=1920. Test internal rescaling explicitly.
        factor = 1920. / self.scale
        x = obs * factor
        _, h = self.gru(x)
        feature = h[-1]
        if self.neighbor_count:
            n = neighbors[:, :self.neighbor_count] * factor
            valid = mask[:, :self.neighbor_count]
            encoded = self.neighbor_encoder(n)
            summed = (encoded * valid[..., None]).sum(dim=2)
            neighbor_h = summed / valid.sum(dim=2, keepdim=True).clamp_min(1)
            slot_valid = valid.any(dim=2)
            score = (neighbor_h * self.query(feature)[:, None]).sum(dim=-1) / math.sqrt(32)
            score = score.masked_fill(~slot_valid, -1e9)
            weight = score.softmax(dim=-1) * slot_valid
            weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            context = (neighbor_h * weight[..., None]).sum(dim=1)
            feature = torch.cat((feature, context), dim=-1)
        pred = self.head(feature).reshape(-1, self.horizon, 2)
        if self.residual:
            steps = torch.arange(1, self.horizon+1, device=obs.device, dtype=obs.dtype)
            pred = pred + x[:, -1, 2:, None].transpose(1, 2) * steps[None, :, None]
        return pred / factor  # Same normalized coordinate units as batch['future'].


def subset_indices(length, limit, seed):
    rng = np.random.default_rng(seed)
    # Random order gives nested training subsets covering the full chronological train range.
    return rng.permutation(length)[:min(length, limit)].astype(np.int64)


def cache_split(store, split, limit, work):
    indices = subset_indices(len(store.samples[split]), limit, 310)
    cache_path = work / f"{split}_{len(indices)}_v1.npz"
    if cache_path.exists():
        a = np.load(cache_path, allow_pickle=False)
        return {k: a[k] for k in a.files}
    arrays = {
        "obs": np.empty((len(indices), 5, 4), np.float32),
        "future": np.empty((len(indices), 6, 2), np.float32),
        "neighbors": np.empty((len(indices), 16, 5, 4), np.float32),
        "neighbor_mask": np.empty((len(indices), 16, 5), bool),
        "pedestrian_id": np.empty(len(indices), np.int64),
        "anchor_frame": np.empty(len(indices), np.int64),
        "origin": np.empty((len(indices), 2), np.float32),
        "index": indices,
    }
    loader = DataLoader(Subset(GCDataset(store, split, include_image=False), indices.tolist()),
                        batch_size=256, num_workers=0, shuffle=False)
    offset, start = 0, time.perf_counter()
    for batch in loader:
        size = len(batch["obs"])
        for key in arrays:
            if key != "index":
                arrays[key][offset:offset+size] = batch[key].numpy()
        offset += size
        if offset % 5120 == 0 or offset == len(indices):
            print(f"CACHE {split}: {offset}/{len(indices)} ({time.perf_counter()-start:.1f}s)", flush=True)
    np.savez(cache_path, **arrays)
    return arrays


def to_tensors(arrays):
    return {k: torch.from_numpy(v) for k, v in arrays.items()}


@torch.no_grad()
def evaluate(model, data, count=None, batch_size=1024, return_errors=False):
    model.eval()
    total = min(len(data["obs"]), count or len(data["obs"]))
    errors = []
    for start in range(0, total, batch_size):
        end = min(total, start+batch_size)
        pred = model(data["obs"][start:end], data["neighbors"][start:end], data["neighbor_mask"][start:end])
        errors.append(((pred-data["future"][start:end])*1920).norm(dim=-1).numpy())
    distance = np.concatenate(errors)
    metric = {"ade_px": float(distance.mean()), "fde_px": float(distance[:, -1].mean())}
    if return_errors:
        return metric, distance
    return metric


def baselines(data):
    obs, y = data["obs"].numpy(), data["future"].numpy()
    steps = np.arange(1, 7, dtype=np.float32)[None, :, None]
    history = obs[:, :, :2]
    centered_time = np.arange(5, dtype=np.float32)-2
    ols_velocity = (history*centered_time[None, :, None]).sum(axis=1)/10
    estimates = {
        "stationary": np.zeros_like(y),
        "last_step_velocity": obs[:, -1, 2:][:, None]*steps,
        "mean_history_velocity": ((history[:, -1]-history[:, 0])/4)[:, None]*steps,
        "ols_history_velocity": ols_velocity[:, None]*steps,
    }
    scores, errors = {}, {}
    for name, pred in estimates.items():
        d = np.linalg.norm((pred-y)*1920, axis=-1)
        scores[name] = {"ade_px": float(d.mean()), "fde_px": float(d[:, -1].mean())}
        errors[name] = d
    return scores, errors


def run_experiment(config, train, val, output, steps, eval_every=150):
    seed_all(config["seed"])
    model = ForecastGRU(hidden=config["hidden"], neighbors=config["neighbors"],
                        scale=config["scale"], residual=config.get("residual", False))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(config["seed"]+1)
    n, batch_size = min(config["samples"], len(train["obs"])), config["batch_size"]
    trace, best, best_state = [], float("inf"), None
    started, last = time.perf_counter(), time.perf_counter()
    rolling = 0.
    for step in range(1, steps+1):
        model.train()
        idx = torch.randint(n, (batch_size,), generator=gen)
        prediction = model(train["obs"][idx], train["neighbors"][idx], train["neighbor_mask"][idx])
        # Common physical-error units across input/target scaling experiments.
        loss = F.smooth_l1_loss(prediction*19.2, train["future"][idx]*19.2, beta=.2)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss in {config['name']} step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        rolling += float(loss.detach())
        if step % eval_every == 0 or step == steps:
            val_metric = evaluate(model, val, count=min(8192, len(val["obs"])))
            train_metric = evaluate(model, train, count=min(4096, n))
            row = {"step": step, "loss": rolling/(step % eval_every or eval_every),
                   "train_ade_px": train_metric["ade_px"],
                   "val_ade_px": val_metric["ade_px"], "val_fde_px": val_metric["fde_px"],
                   "elapsed_seconds": time.perf_counter()-started}
            trace.append(row)
            if val_metric["ade_px"] < best:
                best = val_metric["ade_px"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_step = step
            print(f"TRAIN {config['name']} step {step}/{steps}: trainADE={train_metric['ade_px']:.2f} valADE={val_metric['ade_px']:.2f} ({time.perf_counter()-last:.1f}s)", flush=True)
            last, rolling = time.perf_counter(), 0.
    model.load_state_dict(best_state)
    metric = evaluate(model, val)
    result = {**config, "samples": n, "steps": steps, "examples_seen": steps*batch_size,
              "equivalent_epochs": steps*batch_size/n, "best_step": best_step,
              "parameters": sum(p.numel() for p in model.parameters()),
              "seconds": time.perf_counter()-started, "validation": metric, "trace": trace}
    torch.save({"config": config, "state_dict": model.state_dict()}, output / f"{config['name']}.pt")
    write_json(output / f"{config['name']}.json", result)
    return model, result


def audit_data(store, data):
    report = store.report()
    nonoverlap = {}
    for split, windows in store.samples.items():
        last_anchor, count = {}, 0
        for aid, anchor in windows:
            frame = int(store.tracks[aid][0][anchor])
            if aid not in last_anchor or frame-last_anchor[aid] >= (store.config.obs_len+store.config.pred_len)*20:
                last_anchor[aid] = frame
                count += 1
        nonoverlap[split] = count
    report["per_agent_nonoverlapping_window_counts"] = nonoverlap
    report["cached_counts"] = {k: len(v["obs"]) for k, v in data.items()}
    report["cached_target_agents"] = {k: len(np.unique(v["pedestrian_id"].numpy())) for k, v in data.items()}
    train = data["train"]
    report["train_future_endpoint_displacement_px_quantiles_10_50_90_99"] = np.quantile(
        np.linalg.norm(train["future"].numpy()[:, -1]*1920, axis=-1), [.1, .5, .9, .99]).tolist()
    mask = train["neighbor_mask"].numpy()
    report["train_neighbor_occupied_slots_mean"] = float(mask.any(axis=-1).sum(axis=-1).mean())
    report["train_neighbor_valid_history_fraction"] = float(mask.mean())
    report["train_position_abs_normalized_quantiles_50_90_99"] = np.quantile(
        np.abs(train["obs"].numpy()[:, :, :2]), [.5,.9,.99]).tolist()
    # Frames are disjoint across split ranges; agent identity is deliberately not held out.
    ids = {k: set(v["pedestrian_id"].numpy().tolist()) for k, v in data.items()}
    report["target_agent_overlap_train_val"] = len(ids["train"] & ids["val"])
    report["target_agent_overlap_train_test"] = len(ids["train"] & ids["test"])
    timings = {}
    for include_image in [False, True]:
        dataset = GCDataset(store, "train", include_image=include_image)
        subset = Subset(dataset, np.linspace(0,len(dataset)-1,128,dtype=int).tolist())
        start = time.perf_counter()
        for batch in DataLoader(subset, batch_size=64, num_workers=0):
            if include_image:
                assert batch["image"].shape[1:] == (3,128,128)
        timings[str(include_image)] = {"samples": 128, "seconds": time.perf_counter()-start}
    report["direct_dataloader_timings"] = timings
    return report


def paired_agent_bootstrap(errors_a, errors_b, ids, repetitions=2000):
    unique, inverse = np.unique(ids, return_inverse=True)
    count = np.bincount(inverse)
    summed = np.bincount(inverse, weights=errors_a.mean(axis=1)-errors_b.mean(axis=1))
    rng = np.random.default_rng(843)
    estimates = []
    for _ in range(repetitions):
        chosen = rng.integers(len(unique), size=len(unique))
        estimates.append(summed[chosen].sum()/count[chosen].sum())
    return {"difference_ade_px": float((errors_a-errors_b).mean()),
            "cluster_bootstrap_95pct_ci": np.quantile(estimates,[.025,.975]).tolist(),
            "cluster": "target pedestrian ID", "repetitions": repetitions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1500)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    store = GCStore(args.root)
    data = {split: to_tensors(cache_split(store, split, limit, args.work))
            for split, limit in [("train",60000),("val",10**9),("test",10**9)]}
    audit = audit_data(store, data)
    audit["torch_version"] = torch.__version__
    audit["device"] = "cpu"
    write_json(args.output / "data_audit.json", audit)
    print("DATA_AUDIT", json.dumps(audit, ensure_ascii=False), flush=True)
    val_baselines, _ = baselines(data["val"])
    write_json(args.output / "validation_baselines.json", val_baselines)
    print("VALIDATION_BASELINES", json.dumps(val_baselines), flush=True)
    base = {"samples":40000,"hidden":64,"neighbors":0,"scale":100.,"batch_size":256,"seed":42}
    experiments = [
        {**base,"name":"gru64_scale1920","scale":1920.},
        {**base,"name":"gru64_scale100"},
        {**base,"name":"gru32_scale100","hidden":32},
        {**base,"name":"gru64_10k","samples":10000},
        {**base,"name":"gru64_60k","samples":60000},
        {**base,"name":"gru64_batch64","batch_size":64},
        {**base,"name":"social8","neighbors":8},
        {**base,"name":"social16","neighbors":16},
    ]
    results, models = [], {}
    for config in experiments:
        # Batch study sees the same number of sampled examples as other runs.
        steps = args.steps * (256//config["batch_size"])
        model, result = run_experiment(config, data["train"], data["val"], args.output,
                                       steps=steps, eval_every=max(100,steps//10))
        results.append(result)
        models[config["name"]] = model
        write_json(args.output / "validation_results.json", results)

    winner = min(results, key=lambda x:x["validation"]["ade_px"])
    print("SELECTED_BY_VALIDATION",winner["name"],winner["validation"],flush=True)
    selected_models = [models[winner["name"]]]
    for seed in [17,23]:
        config = {k:winner[k] for k in ["samples","hidden","neighbors","scale","batch_size"]}
        config.update(name=f"selected_seed{seed}",seed=seed)
        model, result = run_experiment(config,data["train"],data["val"],args.output,
                                       steps=winner["steps"],eval_every=max(100,winner["steps"]//10))
        results.append(result)
        selected_models.append(model)
    write_json(args.output / "validation_results.json", results)

    # Only now evaluate the untouched test set, after model selection is frozen.
    test_baselines, baseline_errors = baselines(data["test"])
    selected_test, selected_errors = [], []
    for model in selected_models:
        metric, errors = evaluate(model,data["test"],return_errors=True)
        selected_test.append(metric)
        selected_errors.append(errors)
    baseline_name = min(val_baselines,key=lambda k:val_baselines[k]["ade_px"])
    summary = {
        "selection_rule":"Lowest full-validation ADE across eight fixed-budget configurations; checkpoints chosen on a fixed validation probe.",
        "selected_config": {k:v for k,v in winner.items() if k != "trace"},
        "validation_baselines":val_baselines,"test_baselines":test_baselines,
        "baseline_selected_by_validation":baseline_name,
        "test_gru_seeds":[42,17,23],"test_gru":selected_test,
        "test_gru_ade_mean":float(np.mean([m["ade_px"] for m in selected_test])),
        "test_gru_ade_std":float(np.std([m["ade_px"] for m in selected_test])),
        "test_gru_fde_mean":float(np.mean([m["fde_px"] for m in selected_test])),
        "test_gru_fde_std":float(np.std([m["fde_px"] for m in selected_test])),
        "first_seed_vs_baseline":paired_agent_bootstrap(selected_errors[0],baseline_errors[baseline_name],data["test"]["pedestrian_id"].numpy()),
        "runtime_total_seconds":time.perf_counter()-started,
    }
    write_json(args.output / "summary.json", summary)
    np.savez_compressed(args.output / "test_example_predictions.npz",
        index=data["test"]["index"][:200].numpy(),
        obs=data["test"]["obs"][:200].numpy(),future=data["test"]["future"][:200].numpy(),
        origin=data["test"]["origin"][:200].numpy(),
        prediction=selected_models[0](data["test"]["obs"][:200],data["test"]["neighbors"][:200],data["test"]["neighbor_mask"][:200]).detach().numpy())
    print("FINAL_SUMMARY",json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
