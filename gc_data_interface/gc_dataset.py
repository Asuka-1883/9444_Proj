"""GC trajectory data interface. Core dependencies: NumPy and Pillow.

GCDataset follows PyTorch's map-style dataset protocol without importing torch.
It can be passed directly to torch.utils.data.DataLoader.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import re

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class GCConfig:
    fps: float = 25.0
    frame_step: int = 20
    history_seconds: float = 3.2
    future_seconds: float = 4.8
    max_neighbors: int = 16
    scale_pixels: float = 1920.0
    crop_pixels: int = 256
    image_size: int = 128
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    split_strategy: str = "annotation_points"

    def __post_init__(self):
        if self.fps <= 0 or self.frame_step <= 0 or self.scale_pixels <= 0:
            raise ValueError("fps, frame_step and scale_pixels must be positive")
        if self.history_seconds <= 0 or self.future_seconds <= 0:
            raise ValueError("history_seconds and future_seconds must be positive")
        if self.max_neighbors < 1 or self.crop_pixels < 1 or self.image_size < 1:
            raise ValueError("neighbor count and image dimensions must be positive")
        if not (0 < self.train_fraction < 1 and 0 < self.val_fraction < 1
                and self.train_fraction + self.val_fraction < 1):
            raise ValueError("Invalid train/validation fractions")
        if self.split_strategy not in ("annotation_points", "time"):
            raise ValueError("split_strategy must be annotation_points or time")
        for duration in (self.history_seconds, self.future_seconds):
            intervals = duration / self.dt
            if not np.isclose(intervals, round(intervals)):
                raise ValueError("Window durations must be multiples of frame_step / fps")

    @property
    def dt(self):
        return self.frame_step / self.fps

    @property
    def obs_len(self):
        # Both history endpoints are included: [-3.2, -2.4, -1.6, -0.8, 0].
        return round(self.history_seconds / self.dt) + 1

    @property
    def pred_len(self):
        # Future starts after the anchor: [0.8, 1.6, ..., 4.8].
        return round(self.future_seconds / self.dt)


class GCStore:
    """Read raw annotations once, index agents by frame, and build split windows."""

    def __init__(self, root, config=None):
        self.root = Path(root).expanduser().resolve()
        self.config = config or GCConfig()
        c = self.config
        self.frame_paths = {
            int(p.stem): p for p in (self.root / "Frame").glob("*.jpg")
            if p.stem.isdigit()
        }
        self.frame_ids = np.array(sorted(self.frame_paths), dtype=np.int64)
        if len(self.frame_ids) < 3:
            raise ValueError("Expected Frame/*.jpg with numeric frame names")
        if not np.all(np.diff(self.frame_ids) == c.frame_step):
            raise ValueError("Image frame spacing differs from config.frame_step")
        with Image.open(self.frame_paths[int(self.frame_ids[0])]) as image:
            self.image_width, self.image_height = image.size

        self.tracks = {}
        self.renamed_files = []
        frame_rows = {}
        self.gap_count = 0
        self.samples = {k: [] for k in ("train", "val", "test")}

        for path in sorted((self.root / "Annotation").glob("*.txt")):
            match = re.match(r"^(\d+)", path.stem)
            if not match:
                raise ValueError(f"Cannot extract pedestrian ID: {path.name}")
            agent_id = int(match.group(1))
            if agent_id in self.tracks:
                raise ValueError(f"Duplicate pedestrian ID: {agent_id}")
            if path.stem != f"{agent_id:06d}":
                self.renamed_files.append(path.name)
            values = np.loadtxt(path, dtype=np.float64).reshape(-1)
            if len(values) == 0 or len(values) % 3:
                raise ValueError(f"Expected repeated x,y,frame triples: {path.name}")
            rows = values.reshape(-1, 3)
            if not np.isfinite(rows).all():
                raise ValueError(f"Non-finite values: {path.name}")
            xy = rows[:, :2].astype(np.float32)
            frames = rows[:, 2].astype(np.int64)
            if not np.array_equal(frames, rows[:, 2]):
                raise ValueError(f"Non-integer frame IDs: {path.name}")
            if np.any(np.diff(frames) <= 0):
                raise ValueError(f"Non-increasing frame IDs: {path.name}")
            if not np.isin(frames, self.frame_ids).all():
                raise ValueError(f"Annotations reference unavailable images: {path.name}")
            if np.any(xy < 0) or np.any(xy[:, 0] >= self.image_width) or np.any(xy[:, 1] >= self.image_height):
                raise ValueError(f"Out-of-image position: {path.name}")
            self.tracks[agent_id] = (frames, xy)
            for frame, point in zip(frames, xy):
                frame_rows.setdefault(int(frame), []).append((agent_id, float(point[0]), float(point[1])))

        if not self.tracks:
            raise ValueError("No annotation TXT files found")
        self.frame_agents = {}
        for frame, rows in frame_rows.items():
            a = np.asarray(rows, dtype=np.float64)
            self.frame_agents[frame] = (a[:, 0].astype(np.int64), a[:, 1:].astype(np.float32))

        # Retain chronological isolation. Annotation-weighted cuts avoid assigning
        # almost all test time to the sparsely annotated tail of this raw release.
        n = len(self.frame_ids)
        if c.split_strategy == "time":
            cut1 = int(n * c.train_fraction)
            cut2 = cut1 + int(n * c.val_fraction)
        else:
            counts = np.array([len(frame_rows.get(int(f), [])) for f in self.frame_ids])
            cumulative = np.cumsum(counts)
            cut1 = int(np.searchsorted(cumulative, cumulative[-1]*c.train_fraction)) + 1
            cut2 = int(np.searchsorted(cumulative, cumulative[-1]*(c.train_fraction+c.val_fraction))) + 1
        if not (0 < cut1 < cut2 < n):
            raise ValueError("Not enough distinct annotated time blocks for this split")
        slices = {"train": self.frame_ids[:cut1], "val": self.frame_ids[cut1:cut2],
                  "test": self.frame_ids[cut2:]}
        self.split_ranges = {k: (int(v[0]), int(v[-1])) for k, v in slices.items()}
        self.split_point_counts = {
            key: sum(len(frame_rows.get(int(f), [])) for f in fs)
            for key, fs in slices.items()
        }
        for agent_id, (frames, _) in self.tracks.items():
            # Split at every missing timestamp. Never silently bridge a gap.
            breaks = np.flatnonzero(np.diff(frames) != c.frame_step) + 1
            self.gap_count += len(breaks)
            edges = np.r_[0, breaks, len(frames)]
            for start, stop in zip(edges[:-1], edges[1:]):
                for anchor in range(int(start) + c.obs_len - 1, int(stop) - c.pred_len):
                    first = int(frames[anchor-c.obs_len+1])
                    last = int(frames[anchor+c.pred_len])
                    for split, (lo, hi) in self.split_ranges.items():
                        if lo <= first and last <= hi:
                            self.samples[split].append((agent_id, anchor))
                            break

    def report(self):
        return {
            "root": str(self.root), "config": asdict(self.config),
            "dt_seconds": self.config.dt, "obs_len": self.config.obs_len,
            "pred_len": self.config.pred_len,
            "image_size_original": [self.image_width, self.image_height],
            "image_count": len(self.frame_ids), "pedestrian_count": len(self.tracks),
            "annotation_points": sum(len(f) for f, _ in self.tracks.values()),
            "trajectory_gap_count": self.gap_count,
            "nonstandard_filenames": self.renamed_files,
            "split_frame_ranges_inclusive": self.split_ranges,
            "split_annotation_point_counts": self.split_point_counts,
            "sample_counts": {k: len(v) for k, v in self.samples.items()},
            "sample_pedestrian_counts": {k: len({aid for aid, _ in v}) for k, v in self.samples.items()},
        }


def encode_history(xy, valid, origin, scale):
    """[relative x, relative y, delta x, delta y]; absent values stay zero."""
    features = np.zeros((len(xy), 4), dtype=np.float32)
    features[valid, :2] = (xy[valid] - origin) / scale
    delta_valid = np.zeros(len(xy), dtype=np.bool_)
    delta_valid[1:] = valid[1:] & valid[:-1]
    indices = np.flatnonzero(delta_valid)
    features[indices, 2:] = (xy[indices] - xy[indices-1]) / scale
    return features, delta_valid


def restore_pixels(relative_xy, origin, scale):
    """NumPy helper. For batched torch predictions use the equation in README."""
    return np.asarray(relative_xy) * np.asarray(scale) + np.asarray(origin)


class GCDataset:
    def __init__(self, store, split="train", include_image=True):
        if split not in store.samples:
            raise ValueError("split must be train, val or test")
        self.store = store
        self.split = split
        self.include_image = include_image

    def __len__(self):
        return len(self.store.samples[self.split])

    def __getitem__(self, index):
        s, c = self.store, self.store.config
        agent_id, anchor = s.samples[self.split][index]
        frames, xy = s.tracks[agent_id]
        begin = anchor-c.obs_len+1
        obs_frames = frames[begin:anchor+1]
        future_frames = frames[anchor+1:anchor+1+c.pred_len]
        anchor_frame = int(frames[anchor])
        origin = xy[anchor].copy()
        scale = np.full(2, c.scale_pixels, dtype=np.float32)
        obs_valid = np.ones(c.obs_len, dtype=np.bool_)
        obs, obs_delta_valid = encode_history(xy[begin:anchor+1], obs_valid, origin, scale)

        # Neighbor selection uses the last OBSERVED frame only.
        ids, positions = s.frame_agents[anchor_frame]
        keep = ids != agent_id
        ids, positions = ids[keep], positions[keep]
        distances = np.sum((positions-origin)**2, axis=1)
        order = np.lexsort((ids, distances))[:c.max_neighbors]
        selected = ids[order]
        neighbor_ids = np.full(c.max_neighbors, -1, dtype=np.int64)
        neighbors = np.zeros((c.max_neighbors, c.obs_len, 4), dtype=np.float32)
        neighbor_mask = np.zeros((c.max_neighbors, c.obs_len), dtype=np.bool_)
        neighbor_delta_mask = np.zeros_like(neighbor_mask)
        for slot, neighbor_id in enumerate(selected):
            neighbor_ids[slot] = neighbor_id
            nf, nxy = s.tracks[int(neighbor_id)]
            indices = np.searchsorted(nf, obs_frames)
            valid = indices < len(nf)
            valid[valid] &= nf[indices[valid]] == obs_frames[valid]
            aligned = np.zeros((c.obs_len, 2), dtype=np.float32)
            aligned[valid] = nxy[indices[valid]]
            neighbor_mask[slot] = valid
            neighbors[slot], neighbor_delta_mask[slot] = encode_history(aligned, valid, origin, scale)

        image_path = s.frame_paths[anchor_frame]
        left = int(np.floor(float(origin[0]) - c.crop_pixels/2))
        top = int(np.floor(float(origin[1]) - c.crop_pixels/2))
        crop_box = (left, top, left+c.crop_pixels, top+c.crop_pixels)
        sample = {
            "obs": obs, "obs_mask": obs_valid, "obs_delta_mask": obs_delta_valid,
            "neighbors": neighbors, "neighbor_mask": neighbor_mask,
            "neighbor_slot_mask": neighbor_mask.any(axis=1),
            "neighbor_delta_mask": neighbor_delta_mask,
            "future": ((xy[anchor+1:anchor+1+c.pred_len]-origin)/scale).astype(np.float32),
            "origin": origin, "scale": scale,
            "pedestrian_id": np.int64(agent_id), "neighbor_ids": neighbor_ids,
            "anchor_frame": np.int64(anchor_frame),
            "obs_frame_ids": obs_frames.copy(), "future_frame_ids": future_frames.copy(),
            "image_path": str(image_path), "crop_box": np.array(crop_box, dtype=np.int64),
        }
        if self.include_image:
            with Image.open(image_path) as image:
                crop = image.convert("RGB").crop(crop_box).resize(
                    (c.image_size, c.image_size), Image.Resampling.BILINEAR)
            sample["image"] = np.asarray(crop, dtype=np.float32).transpose(2, 0, 1).copy() / 255.0
            # Mark valid source pixels; areas outside the original image are black padded.
            mask = Image.new("L", (s.image_width, s.image_height), 255).crop(crop_box)
            mask = mask.resize((c.image_size, c.image_size), Image.Resampling.NEAREST)
            sample["image_mask"] = (np.asarray(mask) > 0)[None].copy()
        return sample
