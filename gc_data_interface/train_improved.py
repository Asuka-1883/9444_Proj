"""Train context/goal ablations through complete shuffled training epochs."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from gc_dataset import GCStore
from train_audit import ForecastGRU, to_tensors, write_json, seed_all
from improved_forecast import (prepare_context, local_to_pixels, turn_mask,
    make_background, cluster_anchors, SceneGoalForecast, forecast_loss)


def batch_at(data, indices):
    return {k:v[indices] for k,v in data.items()}


@torch.no_grad()
def assess(model, data, count=None, return_paths=False):
    model.eval()
    n=min(len(data['obs']),count or len(data['obs']))
    distances=[];predictions=[];scores_all=[]
    for start in range(0,n,512):
        batch=batch_at(data,slice(start,min(start+512,n)))
        paths,scores=model(batch)
        distances.append(((paths-batch['future'][:,None])*100).norm(dim=-1).numpy())
        if return_paths:
            predictions.append(paths.numpy());scores_all.append(scores.numpy())
    distance=np.concatenate(distances)
    if not return_paths:
        # Re-evaluate only score logits in chunks, keeping scoring explicitly label-free.
        for start in range(0,n,512):
            _,scores=model(batch_at(data,slice(start,min(start+512,n))))
            scores_all.append(scores.numpy())
    scores=np.concatenate(scores_all)
    top=scores.argmax(axis=-1)
    top_distance=distance[np.arange(n),top]
    turn=turn_mask(batch_at(data,slice(0,n))).numpy()
    metric={
        'count':n,'k':model.k,
        'top1_ade_px':float(top_distance.mean()),'top1_fde_px':float(top_distance[:,-1].mean()),
        'turn_count':int(turn.sum()),'turn_top1_ade_px':float(top_distance[turn].mean()),
        'turn_top1_fde_px':float(top_distance[turn,-1].mean()),
        'minade_px':float(distance.mean(axis=-1).min(axis=-1).mean()),
        'minfde_px':float(distance[:,:,-1].min(axis=-1).mean()),
        'turn_minade_px':float(distance[turn].mean(axis=-1).min(axis=-1).mean()),
        'turn_minfde_px':float(distance[turn,:,-1].min(axis=-1).mean()),
    }
    if return_paths:
        return metric,np.concatenate(predictions),scores
    return metric


def train_one(config,train,val,anchors,background,out,epochs):
    seed_all(config['seed'])
    model=SceneGoalForecast(anchors if config['modes']>1 else torch.zeros(1,6,2),
                            background if config['scene'] else None,
                            spatial=config['spatial'],social=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=2e-4)
    gen=torch.Generator().manual_seed(config['seed']+130)
    turning=turn_mask(train)
    best=float('inf');best_state=None;trace=[];start=time.perf_counter();stale=0
    n=len(train['obs']);batch_size=256
    for epoch in range(1,epochs+1):
        model.train();permutation=torch.randperm(n,generator=gen)
        sum_loss=0.;seen=0
        for indices in permutation.split(batch_size):
            batch=batch_at(train,indices)
            paths,scores=model(batch)
            loss=forecast_loss(paths,scores,batch['future'],turning[indices],config.get('turn_weight',1.))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step();sum_loss+=float(loss.detach())*len(indices);seen+=len(indices)
        assert seen==n
        scheduler.step()
        metric=assess(model,val,count=6144)
        # A declared turn-sensitive selection criterion using VALIDATION only.
        objective=metric['top1_ade_px']+.3*metric['turn_top1_ade_px']
        trace.append({'epoch':epoch,'loss':sum_loss/seen,'seen':seen,'validation_probe':metric,
                      'selection_objective':objective,'seconds':time.perf_counter()-start})
        if objective<best:
            best=objective;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
        else:stale+=1
        print(f"{config['name']} epoch {epoch}/{epochs}: val top1 ADE {metric['top1_ade_px']:.2f}, turn {metric['turn_top1_ade_px']:.2f}, min@{model.k} {metric['minade_px']:.2f}, {time.perf_counter()-start:.1f}s",flush=True)
        if stale>=5 and epoch>=10:break
    model.load_state_dict(best_state)
    full_val=assess(model,val)
    result={'config':config,'samples_per_epoch':n,'epochs_completed':len(trace),
            'best_epoch':best_epoch,'parameters':sum(p.numel() for p in model.parameters()),
            'validation':full_val,'trace':trace,'seconds':time.perf_counter()-start}
    checkpoint={'config':config,'anchors':model.anchors,'background':background if config['scene'] else None,
                'state_dict':model.state_dict()}
    torch.save(checkpoint,out/f"{config['name']}.pt")
    write_json(out/f"{config['name']}.json",result)
    return model,result


def load_model(path):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    c=ck['config']
    model=SceneGoalForecast(ck['anchors'],ck['background'],spatial=c['spatial'],social=c.get('social',True),
          encoder_type=c.get('encoder_type','gru'),hidden_size=c.get('hidden_size',64),
          encoder_layers=c.get('encoder_layers'),transformer_heads=c.get('transformer_heads',4),
          transformer_ffn=c.get('transformer_ffn',128),encoder_dropout=c.get('encoder_dropout',.1))
    model.load_state_dict(ck['state_dict']);model.eval()
    return model


def old_scores(path,raw):
    ck=torch.load(path,map_location='cpu',weights_only=True);c=ck['config']
    model=ForecastGRU(hidden=c['hidden'],neighbors=c['neighbors'],scale=c['scale'])
    model.load_state_dict(ck['state_dict']);model.eval()
    with torch.no_grad():
        pred=torch.cat([model(raw['obs'][i:i+1024],raw['neighbors'][i:i+1024],raw['neighbor_mask'][i:i+1024])
                        for i in range(0,len(raw['obs']),1024)])
    target=prepare_context(raw);turn=turn_mask(target)
    d=((pred-raw['future'])*1920).norm(dim=-1)
    return {'top1_ade_px':float(d.mean()),'top1_fde_px':float(d[:,-1].mean()),
            'turn_top1_ade_px':float(d[turn].mean()),'turn_top1_fde_px':float(d[turn,-1].mean())},pred.numpy()*1920


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--old-results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=16)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    store=GCStore(args.root)
    def read(split,count):
        a=np.load(args.cache/f'{split}_{count}_v1.npz',allow_pickle=False)
        return to_tensors({k:a[k] for k in a.files})
    train_raw=read('train',203475);val_raw=read('val',43814)
    train=prepare_context(train_raw);val=prepare_context(val_raw)
    print('Loaded',len(train['obs']),'training windows; every epoch traverses all once.',flush=True)
    background=make_background(args.root,store.split_ranges['train'][1],args.output/'training_background.png')
    anchors=cluster_anchors(train,count=20)
    torch.save(anchors,args.output/'training_anchors.pt')
    print('Training-only background and 20 trajectory-mode anchors ready.',flush=True)
    seed_all(77)
    # Check causality/shape before starting expensive training.
    example=batch_at(val,slice(0,3))
    check=SceneGoalForecast(anchors,background).eval()
    with torch.no_grad():
        paths,scores=check(example)
        changed={**example,'future':torch.randn_like(example['future'])*50}
        again,_=check(changed)
        assert paths.shape==(3,20,6,2) and scores.shape==(3,20)
        assert torch.isfinite(paths).all() and torch.equal(paths,again)
        world=local_to_pixels(example['future'],example['basis'])
        torch.testing.assert_close(world,val_raw['future'][:3]*1920,atol=1e-3,rtol=1e-5)
    base={'seed':42,'spatial':True,'scene':False,'modes':1,'turn_weight':1.}
    configs=[
        {**base,'name':'position_waypoint'},
        {**base,'name':'scene_waypoint','scene':True},
        {**base,'name':'intent20_position','modes':20},
        {**base,'name':'intent20_scene','modes':20,'scene':True},
        {**base,'name':'intent20_scene_turn2','modes':20,'scene':True,'turn_weight':2.},
    ]
    records=[];models={}
    for config in configs:
        model,result=train_one(config,train,val,anchors,background,args.output,args.epochs)
        models[config['name']]=model;records.append(result)
        write_json(args.output/'validation_results.json',records)
    def criterion(row):return row['validation']['top1_ade_px']+.3*row['validation']['turn_top1_ade_px']
    selected=min(records,key=criterion)
    selected_multimodal=min([r for r in records if r['config']['modes']>1],key=criterion)
    selection={'single_best':selected['config']['name'],'multimodal_best':selected_multimodal['config']['name'],
               'criterion':'Validation top1 ADE + 0.3 * validation turning top1 ADE',
               'caveat':'Old test split has already been inspected during development. These are iterative development comparisons, not pristine independent test claims.'}
    write_json(args.output/'selection.json',selection)
    print('SELECTION_FROZEN',selection,flush=True)
    test_raw=read('test',45454);test=prepare_context(test_raw)
    result={}
    result['old_gru'],old_pred=old_scores(args.old_results/'social16.pt',test_raw)
    result['old_long_full_gru'],_=old_scores(args.old_results/'long_gru_full.pt',test_raw)
    stored={}
    for name in dict.fromkeys([selection['single_best'],selection['multimodal_best']]):
        metric,paths,scores=assess(models[name],test,return_paths=True)
        result[name]=metric;stored[name]=(paths,scores)
    summary={'selection':selection,'test':result,'train_windows':len(train['obs']),
             'validation_windows':len(val['obs']),'test_windows':len(test['obs'])}
    write_json(args.output/'summary.json',summary)
    for name,(paths,scores) in stored.items():
        paths_t=torch.from_numpy(paths)
        world=local_to_pixels(paths_t,test['basis']).numpy()
        np.savez_compressed(args.output/f'{name}_test_predictions.npz',paths_pixels=world,scores=scores,
            future_pixels=test_raw['future'].numpy()*1920,obs_pixels=test_raw['obs'].numpy()[:,:,:2]*1920,
            old_pixels=old_pred,origin=test_raw['origin'].numpy(),index=test_raw['index'].numpy(),
            turning=turn_mask(test).numpy())
    print('FINAL',json.dumps(summary),flush=True)


if __name__=='__main__':main()
