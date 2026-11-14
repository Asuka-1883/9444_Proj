"""Collect validation results from runs/*/result.json into pandas DataFrames.

Runs that differ only by a seed suffix (_s42, _s43, ...) are grouped into one
configuration; a run without a suffix counts as one more seed of that config.
Smoke tests are skipped. Only reads files, never modifies anything.

    python seed_summary.py                    # print both tables
    python seed_summary.py --csv summary.csv  # also save the mean ± std table

In a notebook / other script:
    from seed_summary import load_runs, summarize
    runs = load_runs()          # one row per run
    table = summarize(runs)     # one row per configuration (mean / std)
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

METRICS = ['ade_px', 'fde_px', 'turn_ade_px', 'turn_fde_px']


def load_runs(runs_dir='runs'):
    """One row per finished run: config, seed, metrics, best epoch, parameters, seconds."""
    rows = []
    for path in sorted(Path(runs_dir).glob('*/result.json')):
        name = path.parent.name
        if 'smoke' in name:
            continue
        result = json.loads(path.read_text())
        validation = result.get('validation') or {}
        if 'ade_px' not in validation:
            print('skip (no validation metrics):', name)
            continue
        match = re.search(r'_s(\d+)$', name)
        seed = int(match.group(1)) if match else (result.get('config') or {}).get('seed')
        rows.append({'config': re.sub(r'_s\d+$', '', name), 'run': name, 'seed': seed,
                     **{m: validation.get(m) for m in METRICS},
                     'best_epoch': result.get('best_epoch'),
                     'parameters': result.get('parameters'),
                     'seconds': result.get('seconds')})
    return pd.DataFrame(rows)


def summarize(runs):
    """One row per config: number of seeds, parameters, mean and std of each metric."""
    grouped = runs.groupby('config')
    table = grouped[METRICS].agg(['mean', 'std'])
    table.columns = [f'{metric}_{stat}' for metric, stat in table.columns]
    table.insert(0, 'seeds', grouped.size())
    table.insert(1, 'parameters', grouped['parameters'].first())
    table['minutes_mean'] = grouped['seconds'].mean() / 60
    return table.sort_values('ade_px_mean')


def pretty(table):
    """'mean ± std' strings for reading or pasting into the report."""
    out = table[['seeds', 'parameters']].copy()
    for m in METRICS:
        out[m] = table[f'{m}_mean'].map('{:.2f}'.format) + ' ± ' + table[f'{m}_std'].map('{:.2f}'.format)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', default='runs')
    parser.add_argument('--csv', help='save the numeric mean/std table to this CSV file')
    args = parser.parse_args()
    runs = load_runs(args.runs)
    if runs.empty:
        raise SystemExit('No finished runs found under ' + args.runs)
    pd.set_option('display.width', 200, 'display.max_columns', 20)
    print(runs.sort_values(['config', 'seed']).round(2).to_string(index=False), end='\n\n')
    table = summarize(runs)
    print(pretty(table).to_string())
    if args.csv:
        table.round(4).to_csv(args.csv)
        print('saved', args.csv)


if __name__ == '__main__':
    main()
