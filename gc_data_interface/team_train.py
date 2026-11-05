"""One common training/evaluation entry point for GRU, LSTM and Transformer.

Examples:
python team_train.py --root RAW_DATA --encoder gru --cache-dir CACHE --output runs/gru
python team_train.py --root RAW_DATA --encoder lstm --cache-dir CACHE --output runs/lstm
python team_train.py --root RAW_DATA --encoder transformer --cache-dir CACHE --output runs/transformer
"""
import argparse
import copy
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from improved_forecast import forecast_loss,turn_mask
from team_model import build_team_model,load_team_config,checkpoint_config
from team_data import TeamData,stable_hash,move_batch


def save_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


@torch.no_grad()
def evaluate_team(model,data,device='cpu',batch_size=512):
    model.eval();device=torch.device(device)
    errors=[]
    for start in range(0,len(data['obs']),batch_size):
        batch=move_batch(data,slice(start,start+batch_size),device)
        paths,scores=model(batch)
        chosen=paths[torch.arange(len(paths),device=device),scores.argmax(dim=-1)]
        errors.append(((chosen-batch['future'])*100).norm(dim=-1).cpu())
    distance=torch.cat(errors);turning=turn_mask(data).cpu()
    return {'samples':len(distance),'ade_px':float(distance.mean()),'fde_px':float(distance[:,-1].mean()),
            'turn_samples':int(turning.sum()),
            'turn_ade_px':float(distance[turning].mean()) if turning.any() else None,
            'turn_fde_px':float(distance[turning,-1].mean()) if turning.any() else None}


def choose_device(name):
    if name=='auto':return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA is not available')
    if name=='mps' and not torch.backends.mps.is_available():raise ValueError('MPS is not available')
    return torch.device(name)


def run(args):
    config=load_team_config(args.config)
    if args.epochs is not None:config['epochs']=args.epochs
    if args.batch_size is not None:config['batch_size']=args.batch_size
    if args.seed is not None:config['seed']=args.seed
    if config['epochs']<1 or config['batch_size']<1:raise ValueError('epochs/batch size must be positive')
    for limit in (args.train_limit,args.val_limit):
        if limit is not None and limit<1:raise ValueError('Subset limits must be positive')
    device=choose_device(args.device)
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    torch.set_num_threads(args.threads)
    data=TeamData(args.root,args.cache_dir)
    train=data.context('train',args.train_limit);val=data.context('val',args.val_limit)
    training_protocol={'data':data.protocol,'shared_config':config,'train_limit':args.train_limit,
                       'val_limit':args.val_limit,'decoder':'shared_waypoint_single_v2',
                       'loss':'position+0.5endpoint+0.3displacement;SmoothL1_beta0.1',
                       'selection':'validation_ade+0.3*validation_turn_ade',
                       'shared_initialization':'same non-motion state for the same seed',
                       'train_subset_indices':stable_hash(train['index'].tolist()),
                       'val_subset_indices':stable_hash(val['index'].tolist())}
    signature=stable_hash(training_protocol)
    output=Path(args.output)
    if (output/'result.json').exists() or (output/'best.pt').exists():
        raise FileExistsError('Output already contains a run; choose a new --output directory')
    output.mkdir(parents=True,exist_ok=True)
    model=build_team_model(args.encoder,config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['epochs'],eta_min=config['min_learning_rate'])
    generator=torch.Generator().manual_seed(config['seed']+130)
    torch.manual_seed(config['seed']+1000)
    turning=turn_mask(train)
    trace=[];best=float('inf');best_state=None;stale=0;began=time.perf_counter()
    for epoch in range(1,config['epochs']+1):
        model.train();order=torch.randperm(len(train['obs']),generator=generator)
        loss_sum=0.;seen=0
        for indices in order.split(config['batch_size']):
            batch=move_batch(train,indices,device)
            paths,scores=model(batch)
            loss=forecast_loss(paths,scores,batch['future'],turning[indices].to(device),config['turn_weight'])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'])
            optimizer.step();loss_sum+=float(loss.detach())*len(indices);seen+=len(indices)
        assert seen==len(order)
        scheduler.step()
        metric=evaluate_team(model,val,device)
        objective=metric['ade_px']+.3*(metric['turn_ade_px'] or 0.)
        row={'epoch':epoch,'seen':seen,'loss':loss_sum/seen,'validation':metric,
             'selection_objective':objective,'seconds':time.perf_counter()-began}
        trace.append(row);print(json.dumps(row),flush=True)
        if objective<best:
            best=objective;best_epoch=epoch;stale=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        if stale>=config['patience'] and epoch>=config['min_epochs']:break
    model.load_state_dict(best_state)
    result={'package_version':'2.0.0','encoder':args.encoder,'config':config,'device':str(device),
            'torch_version':torch.__version__,'protocol_signature':signature,'protocol':training_protocol,
            'train_samples':len(train['obs']),'val_samples':len(val['obs']),
            'parameters':sum(p.numel() for p in model.parameters()),
            'encoder_parameters':sum(p.numel() for p in model.motion.parameters()),
            'epochs_completed':len(trace),'best_epoch':best_epoch,'seconds':time.perf_counter()-began,
            'validation':evaluate_team(model,val,device),'trace':trace,
            'test_evaluated':False}
    torch.save({'config':checkpoint_config(args.encoder,config),'anchors':model.anchors.cpu(),
                'background':None,'state_dict':best_state,'protocol_signature':signature},output/'best.pt')
    save_json(output/'result.json',result)
    print('SAVED',str(output/'best.pt'),'parameters',result['parameters'],flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--encoder',choices=['gru','lstm','transformer'],required=True)
    p.add_argument('--cache-dir',default='work/team_cache')
    p.add_argument('--output',required=True)
    p.add_argument('--config',default=None)
    p.add_argument('--epochs',type=int);p.add_argument('--batch-size',type=int);p.add_argument('--seed',type=int)
    p.add_argument('--train-limit',type=int);p.add_argument('--val-limit',type=int)
    p.add_argument('--device',choices=['auto','cpu','cuda','mps'],default='cpu')
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    if args.threads<1:p.error('--threads must be positive')
    run(args)


if __name__=='__main__':main()
