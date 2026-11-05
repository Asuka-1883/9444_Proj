"""Repeat the validation-selected top-1 configuration, without further tuning."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from train_audit import to_tensors,write_json
from improved_forecast import prepare_context
from train_improved import train_one,assess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=16)
    args=p.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    selection=json.loads((args.output/'selection.json').read_text())
    name=selection['single_best']
    ck=torch.load(args.output/f'{name}.pt',map_location='cpu',weights_only=True)
    def read(split,count):
        a=np.load(args.cache/f'{split}_{count}_v1.npz',allow_pickle=False)
        return prepare_context(to_tensors({k:a[k] for k in a.files}))
    train=read('train',203475);val=read('val',43814)
    models=[];records=[]
    for seed in [17,23]:
        config={**ck['config'],'seed':seed,'name':f'{name}_seed{seed}'}
        model,record=train_one(config,train,val,ck['anchors'],ck['background'],args.output,args.epochs)
        models.append(model);records.append(record)
    # Freeze both replicas before reading test labels for evaluation.
    test=read('test',45454)
    original=json.loads((args.output/'summary.json').read_text())['test'][name]
    tests=[original]+[assess(model,test) for model in models]
    result={'selected_config':name,'seeds':[42,17,23],'test':tests,
            'top1_ade_mean':float(np.mean([r['top1_ade_px'] for r in tests])),
            'top1_ade_std':float(np.std([r['top1_ade_px'] for r in tests])),
            'top1_fde_mean':float(np.mean([r['top1_fde_px'] for r in tests])),
            'turn_top1_ade_mean':float(np.mean([r['turn_top1_ade_px'] for r in tests])),
            'turn_top1_ade_std':float(np.std([r['turn_top1_ade_px'] for r in tests])),
            'turn_top1_fde_mean':float(np.mean([r['turn_top1_fde_px'] for r in tests]))}
    write_json(args.output/'repeatability.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
