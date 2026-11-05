"""Compare validation results only when all shared run settings match."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    results=[json.loads((run/'result.json').read_text()) for run in args.runs]
    signatures={r['protocol_signature'] for r in results}
    if len(signatures)!=1:raise ValueError('Different data/shared training settings: use the same config, seed and subset limits')
    rows=[{k:r[k] for k in ['encoder','device','parameters','encoder_parameters','train_samples','val_samples',
                           'epochs_completed','best_epoch','seconds','validation']} for r in results]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'protocol_signature':next(iter(signatures)),'comparison':rows},indent=2)+'\n')
    print('Encoder | Params | Val ADE px | Val FDE px | Seconds')
    for r in rows:print(f"{r['encoder']} | {r['parameters']} | {r['validation']['ade_px']:.3f} | {r['validation']['fde_px']:.3f} | {r['seconds']:.2f}")


if __name__=='__main__':main()
