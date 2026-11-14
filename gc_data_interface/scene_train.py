"""Train / validate the scene + social Transformer with video-frame input.

Mirrors team_train.py (same data split, cache, loss, schedule, early stopping and
validation-only model selection) so results are comparable with runs/transformer.
Only the model and the extra image input differ. team_train.py is untouched.

Build the frame cache first (once):
    python scene_frames.py --root DATASET_ROOT --out work/frames

Smoke test:
    python scene_train.py --root DATASET_ROOT --output runs/scene_smoke --epochs 2 --train-limit 2048 --val-limit 512
Full run:
    python scene_train.py --root DATASET_ROOT --output runs/scene_transformer --device mps
Ablations: add --no-image or --no-social.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from improved_forecast import forecast_loss, turn_mask, prepare_context
from team_data import TeamData, move_batch
from team_model import load_team_config
from scene_frames import FrameCache
from scene_transformer import SceneSocialTransformer


def load_split(data, split, limit):
    raw = data.raw(split, limit)
    context = prepare_context(raw)
    context['anchor_frame'] = raw['anchor_frame']  # which frame to crop: the last OBSERVED one
    return context


def with_images(batch, frames, device, use_image):
    if use_image:
        batch['image'] = frames.crops(batch['anchor_frame'], batch['origin'], batch['heading'], device)
    return batch


@torch.no_grad()
def evaluate(model, data, frames, device, use_image, batch_size=512):
    model.eval()
    errors = []
    for start in range(0, len(data['obs']), batch_size):
        batch = with_images(move_batch(data, slice(start, start + batch_size), device), frames, device, use_image)
        paths, _scores = model(batch)
        errors.append(((paths[:, 0] - batch['future']) * 100).norm(dim=-1).cpu())  # 100-px units -> px
    distance = torch.cat(errors)
    turning = turn_mask(data).cpu()
    return {'samples': len(distance), 'ade_px': float(distance.mean()), 'fde_px': float(distance[:, -1].mean()),
            'turn_samples': int(turning.sum()),
            'turn_ade_px': float(distance[turning].mean()) if turning.any() else None,
            'turn_fde_px': float(distance[turning, -1].mean()) if turning.any() else None}


def choose_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    return torch.device(name)


def run(args):
    config = load_team_config()
    for key in ('epochs', 'batch_size', 'seed'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    use_image, use_social = not args.no_image, not args.no_social
    device = choose_device(args.device)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.set_num_threads(args.threads)

    output = Path(args.output)
    if (output / 'result.json').exists() or (output / 'best.pt').exists():
        raise FileExistsError('Output already contains a run; choose a new --output directory')
    data = TeamData(args.root, args.cache_dir)
    train, val = load_split(data, 'train', args.train_limit), load_split(data, 'val', args.val_limit)
    frames = FrameCache(args.frames, crop_pixels=args.crop_pixels, image_size=64, forward_shift=args.forward_shift)
    output.mkdir(parents=True, exist_ok=True)

    model = SceneSocialTransformer(width=config['hidden_size'], heads=config['transformer_heads'],
                                   ffn=config['transformer_ffn'], dropout=config['encoder_dropout'],
                                   use_image=use_image, use_social=use_social).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_learning_rate'])
    generator = torch.Generator().manual_seed(config['seed'] + 130)
    turning = turn_mask(train)
    trace, best, best_state, best_epoch, stale = [], float('inf'), None, 0, 0
    began = time.perf_counter()
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        order = torch.randperm(len(train['obs']), generator=generator)
        loss_sum, seen = 0., 0
        for step, indices in enumerate(order.split(config['batch_size'])):
            batch = with_images(move_batch(train, indices, device), frames, device, use_image)
            paths, scores = model(batch)
            loss = forecast_loss(paths, scores, batch['future'], turning[indices].to(device), config['turn_weight'])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices); seen += len(indices)
            if args.log_every and step % args.log_every == 0:
                print(f'epoch {epoch} step {step} seen {seen}/{len(order)} loss {loss_sum / seen:.4f} '
                      f'{time.perf_counter() - began:.0f}s', flush=True)
        scheduler.step()
        metric = evaluate(model, val, frames, device, use_image)
        objective = metric['ade_px'] + .3 * (metric['turn_ade_px'] or 0.)  # same selection rule as team_train
        row = {'epoch': epoch, 'loss': loss_sum / seen, 'validation': metric,
               'selection_objective': objective, 'seconds': time.perf_counter() - began}
        trace.append(row); print(json.dumps(row), flush=True)
        if objective < best:
            best, best_epoch, stale = objective, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({'state_dict': best_state, 'args': vars(args), 'config': config}, output / 'best.pt')
        else:
            stale += 1
        if stale >= config['patience'] and epoch >= config['min_epochs']:
            break
    model.load_state_dict(best_state)
    result = {'model': 'scene_social_transformer', 'use_image': use_image, 'use_social': use_social,
              'config': config, 'args': vars(args), 'device': str(device), 'torch_version': torch.__version__,
              'data_fingerprint': data.fingerprint, 'train_samples': len(train['obs']), 'val_samples': len(val['obs']),
              'parameters': sum(p.numel() for p in model.parameters()),
              'epochs_completed': len(trace), 'best_epoch': best_epoch, 'seconds': time.perf_counter() - began,
              'validation': evaluate(model, val, frames, device, use_image), 'trace': trace, 'test_evaluated': False}
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('SAVED', output / 'best.pt', 'parameters', result['parameters'], flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--cache-dir', default='work/team_cache')
    p.add_argument('--frames', default='work/frames')
    p.add_argument('--epochs', type=int); p.add_argument('--batch-size', type=int); p.add_argument('--seed', type=int)
    p.add_argument('--train-limit', type=int); p.add_argument('--val-limit', type=int)
    p.add_argument('--no-image', action='store_true'); p.add_argument('--no-social', action='store_true')
    p.add_argument('--crop-pixels', type=int, default=256, help='crop side length in original pixels')
    p.add_argument('--forward-shift', type=int, default=64, help='move the crop centre ahead along the heading (px)')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--log-every', type=int, default=100)
    run(p.parse_args())


if __name__ == '__main__':
    main()
