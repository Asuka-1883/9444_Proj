"""Validation-constrained turn-focused single-trajectory alternatives."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from train_audit import to_tensors,write_json
from improved_forecast import prepare_context,local_to_pixels
from train_improved import train_one,assess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    def read(split,count):
        a=np.load(args.cache/f'{split}_{count}_v1.npz',allow_pickle=False)
        raw=to_tensors({k:a[k] for k in a.files})
        return raw,prepare_context(raw)
    _,train=read('train',203475);_,val=read('val',43814)
    records=[];models={}
    for weight in [2.,3.]:
        config={'name':f'position_turn{int(weight)}','seed':42,'modes':1,'spatial':True,'scene':False,'turn_weight':weight}
        model,record=train_one(config,train,val,torch.zeros(1,6,2),None,args.output,16)
        records.append(record);models[config['name']]=model
        write_json(args.output/'turn_focused_validation.json',records)
    base=json.loads((args.output/'position_waypoint.json').read_text())
    ceiling=1.10*base['validation']['top1_ade_px']
    acceptable=[base]+[r for r in records if r['validation']['top1_ade_px']<=ceiling]
    chosen=min(acceptable,key=lambda r:r['validation']['turn_top1_ade_px'])
    name=chosen['config']['name']
    selection={'name':name,'rule':'Lowest validation turning ADE subject to overall validation ADE <= 1.10 times position_waypoint; base is eligible.',
               'validation_overall_ceiling':ceiling,'validation':chosen['validation']}
    write_json(args.output/'turn_focused_selection.json',selection)
    if name=='position_waypoint':return
    raw,test=read('test',45454)
    metric,paths,scores=assess(models[name],test,return_paths=True)
    world=local_to_pixels(torch.from_numpy(paths),test['basis']).numpy()
    original=np.load(args.output/'position_waypoint_test_predictions.npz')
    np.savez_compressed(args.output/f'{name}_test_predictions.npz',paths_pixels=world,scores=scores,
        future_pixels=raw['future'].numpy()*1920,obs_pixels=raw['obs'].numpy()[:,:,:2]*1920,
        old_pixels=original['old_pixels'],origin=raw['origin'].numpy(),index=raw['index'].numpy(),turning=original['turning'])
    write_json(args.output/'turn_focused_test.json',{'name':name,'test':metric})
    print('TURN_FOCUSED',name,json.dumps(metric),flush=True)


if __name__=='__main__':main()
