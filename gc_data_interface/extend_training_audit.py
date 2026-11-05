"""Validation-only checks of full training data and less-overlapping windows.

This extension does not read or evaluate the test set. The original test results
and selected checkpoint remain frozen.
"""
import argparse
from pathlib import Path
import torch
import numpy as np
from gc_dataset import GCStore
from train_audit import cache_split,to_tensors,run_experiment,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    store=GCStore(args.root)
    full=to_tensors(cache_split(store,'train',10**9,args.work))
    val=to_tensors(cache_split(store,'val',10**9,args.work))
    chosen=[];last={}
    for i,(aid,anchor) in enumerate(store.samples['train']):
        frame=int(store.tracks[aid][0][anchor])
        if aid not in last or frame-last[aid]>=220:
            chosen.append(i);last[aid]=frame
    keep=np.flatnonzero(np.isin(full['index'].numpy(),chosen))
    independent={k:v[keep] for k,v in full.items()}
    base={'hidden':64,'neighbors':0,'scale':100.,'batch_size':256,'seed':42}
    configurations=[
        ({**base,'name':'long_gru_40k','samples':40000},full),
        ({**base,'name':'long_gru_full','samples':len(full['obs'])},full),
        ({**base,'name':'long_gru_nonoverlap','samples':len(independent['obs'])},independent),
    ]
    result=[]
    for config,train in configurations:
        _,r=run_experiment(config,train,val,args.output,steps=8000,eval_every=800)
        result.append(r)
        write_json(args.output/'validation_extensions.json',result)


if __name__=='__main__':main()
