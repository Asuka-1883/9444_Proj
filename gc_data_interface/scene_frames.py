"""Downsampled frame cache + heading-aligned image crops for the scene Transformer.

Why: the training set has ~200k windows but only 6001 distinct frames. Decoding a
1920x1080 JPEG per sample is far too slow, so every frame is decoded ONCE, shrunk
by FACTOR (default 4 -> 480x270) and stored as a uint8 memory-mapped .npy file
under work/ (ignored by git, ~2.2 GB).

Crops use only the anchor frame (the last observed time step), so no future
information reaches the model.

Build once:
    python scene_frames.py --root DATASET_ROOT --out work/frames
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image


def build_frame_cache(root, out_dir, factor=4):
    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted((p for p in (root / 'Frame').glob('*.jpg') if p.stem.isdigit()),
                   key=lambda p: int(p.stem))
    if not paths:
        raise ValueError('No Frame/*.jpg found under --root')
    with Image.open(paths[0]) as image:
        width, height = image.size
    w, h = width // factor, height // factor
    array = np.lib.format.open_memmap(out_dir / 'frames.npy', mode='w+', dtype=np.uint8,
                                      shape=(len(paths), h, w, 3))
    for i, path in enumerate(paths):
        with Image.open(path) as image:
            image.draft('RGB', (w, h))  # fast JPEG DCT-domain downscale
            array[i] = np.asarray(image.convert('RGB').resize((w, h), Image.Resampling.BILINEAR))
        if (i + 1) % 500 == 0 or i + 1 == len(paths):
            print(f'FRAMES {i + 1}/{len(paths)}', flush=True)
    array.flush()
    meta = {'frame_ids': [int(p.stem) for p in paths], 'factor': factor,
            'original_size': [width, height], 'cached_size': [w, h]}
    (out_dir / 'meta.json').write_text(json.dumps(meta) + '\n')
    return out_dir


class FrameCache:
    """Crops a heading-aligned window around each target from its anchor frame.

    Output image channels: RGB in [0,1] plus one validity channel (0 outside the
    original picture). The crop's +x axis points along the walking direction,
    matching the team's heading-aligned coordinates (prepare_context).
    """

    def __init__(self, out_dir, crop_pixels=256, image_size=64, forward_shift=64):
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / 'meta.json').read_text())
        self.frames = np.load(out_dir / 'frames.npy', mmap_mode='r')
        ids = np.asarray(meta['frame_ids'], dtype=np.int64)
        self.first, self.step = int(ids[0]), int(ids[1] - ids[0])
        if not np.array_equal(ids, self.first + self.step * np.arange(len(ids))):
            raise ValueError('Frame cache ids are not evenly spaced')
        self.factor = meta['factor']
        self.crop_pixels, self.image_size, self.forward_shift = crop_pixels, image_size, forward_shift
        # Axis-aligned source window (in cached pixels) large enough to hold the
        # rotated, shifted crop; odd size so its centre is a whole pixel.
        reach = (crop_pixels / 2 * 2 ** .5 + abs(forward_shift)) / self.factor
        self.window = int(np.ceil(reach)) * 2 + 3

    def _windows(self, anchor_frames, centers):
        """NumPy: [B] frame ids, [B,2] integer centres -> [B,4,W,W] float32."""
        n, (fh, fw) = len(anchor_frames), self.frames.shape[1:3]
        half, size = self.window // 2, self.window
        out = np.zeros((n, size, size, 4), dtype=np.float32)
        indices = (anchor_frames - self.first) // self.step
        for i in range(n):
            cx, cy = int(centers[i, 0]), int(centers[i, 1])
            x0, y0 = cx - half, cy - half
            sx0, sy0 = max(x0, 0), max(y0, 0)
            sx1, sy1 = min(x0 + size, fw), min(y0 + size, fh)
            if sx0 >= sx1 or sy0 >= sy1:
                continue
            patch = self.frames[int(indices[i]), sy0:sy1, sx0:sx1]
            out[i, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0, :3] = patch / 255.
            out[i, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0, 3] = 1.
        return out.transpose(0, 3, 1, 2)

    def crops(self, anchor_frames, origin, heading, device='cpu'):
        """anchor_frames [B], origin [B,2] original pixels, heading [B,2] unit vector."""
        anchor_frames = anchor_frames.cpu().numpy().astype(np.int64)
        origin = origin.detach().cpu().float()
        heading = heading.detach().cpu().float()
        center = origin / self.factor
        rounded = center.round()
        source = torch.from_numpy(self._windows(anchor_frames, rounded.numpy())).to(device)
        # affine_grid maps output (u,v) in [-1,1] to source coords in [-1,1].
        perpendicular = torch.stack((-heading[:, 1], heading[:, 0]), dim=-1)
        per_pixel = 2. / self.window  # normalized units per cached pixel
        half = self.crop_pixels / 2 / self.factor * per_pixel
        rotation = torch.stack((heading, perpendicular), dim=-1) * half  # columns: u-axis, v-axis
        offset = (heading * self.forward_shift / self.factor + (center - rounded)) * per_pixel
        theta = torch.cat((rotation, offset[:, :, None]), dim=-1).to(device)
        grid = F.affine_grid(theta, (len(theta), 4, self.image_size, self.image_size), align_corners=False)
        return F.grid_sample(source, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', required=True, help='folder containing Annotation and Frame')
    parser.add_argument('--out', default='work/frames')
    parser.add_argument('--factor', type=int, default=4)
    args = parser.parse_args()
    build_frame_cache(args.root, args.out, args.factor)


if __name__ == '__main__':
    main()
