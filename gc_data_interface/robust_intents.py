"""Robust training-only mode anchors; all raw train/evaluation windows retained."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from train_audit import to_tensors,write_json
from improved_forecast import prepare_context,cluster_anchors,local_to_pixels
from train_improved import train_one,assess,load_model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=16)
    args=p.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    def read(split,count):
        a=np.load(args.cache/f'{split}_{count}_v1.npz',allow_pickle=False)
        raw=to_tensors({k:a[k] for k in a.files})
        return raw,prepare_context(raw)
    _,train=read('train',203475);_,val=read('val',43814)
    anchors=cluster_anchors(train,count=20,trim_quantile=.995)
    torch.save(anchors,args.output/'robust_training_anchors.pt')
    scene_checkpoint=torch.load(args.output/'intent20_scene.pt',weights_only=True,map_location='cpu')
    background=scene_checkpoint['background']
    records=[];models={}
    for scene in [False,True]:
        config={'name':'robust_intent20_scene' if scene else 'robust_intent20_position',
                'seed':42,'modes':20,'spatial':True,'scene':scene,'turn_weight':1.,'anchor_trim_quantile':.995}
        model,record=train_one(config,train,val,anchors,background,args.output,args.epochs)
        models[config['name']]=model;records.append(record)
        write_json(args.output/'robust_validation_results.json',records)
    chosen=min(records,key=lambda r:r['validation']['top1_ade_px']+.3*r['validation']['turn_top1_ade_px'])
    name=chosen['config']['name']
    write_json(args.output/'robust_selection.json',{'name':name,'criterion':'validation top1 ADE + .3*turn ADE','anchor_trim_quantile':.995})
    raw,test=read('test',45454)
    metric,paths,scores=assess(models[name],test,return_paths=True)
    world=local_to_pixels(torch.from_numpy(paths),test['basis']).numpy()
    old=np.load(args.output/'position_waypoint_test_predictions.npz')['old_pixels']
    np.savez_compressed(args.output/f'{name}_test_predictions.npz',paths_pixels=world,scores=scores,
        future_pixels=raw['future'].numpy()*1920,obs_pixels=raw['obs'].numpy()[:,:,:2]*1920,
        old_pixels=old,origin=raw['origin'].numpy(),index=raw['index'].numpy(),
        turning=np.load(args.output/'position_waypoint_test_predictions.npz')['turning'])
    summary=json.loads((args.output/'summary.json').read_text())
    summary['selection']['multimodal_best']=name
    summary['selection']['robust_anchor_update']='After the initial development pass, anchors refit using training-only 99.5% residual-norm quantile; no train, validation or test windows dropped.'
    summary['test'][name]=metric
    write_json(args.output/'summary.json',summary)
    print('ROBUST_FINAL',name,json.dumps(metric),flush=True)


if __name__=='__main__':main()
