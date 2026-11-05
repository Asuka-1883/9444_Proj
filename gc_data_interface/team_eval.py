"""Explicit evaluation entry point; training never automatically scores test samples."""
import argparse,json
from pathlib import Path
import torch
from train_improved import load_model
from team_data import TeamData
from team_train import choose_device,evaluate_team,save_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--cache-dir',required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--split',choices=['val','test'],required=True)
    p.add_argument('--limit',type=int);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['auto','cpu','cuda','mps'],default='cpu')
    args=p.parse_args();device=choose_device(args.device);torch.set_num_threads(4)
    data=TeamData(args.root,args.cache_dir)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    run_path=args.checkpoint.parent/'result.json'
    if run_path.exists():
        run=json.loads(run_path.read_text())
        if run['protocol']['data']!=data.protocol:
            # JSON represents tuples as lists: compare normalized JSON text.
            if json.dumps(run['protocol']['data'],sort_keys=True)!=json.dumps(data.protocol,sort_keys=True):
                raise ValueError('Evaluation data differs from the training data protocol')
    context=data.context(args.split,args.limit)
    model=load_model(args.checkpoint).to(device)
    result={'encoder':ck['config'].get('encoder_type','gru'),'split':args.split,
            'limit':args.limit,'protocol_signature':ck.get('protocol_signature'),
            'data_fingerprint':data.fingerprint,'metrics':evaluate_team(model,context,device)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    save_json(args.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
