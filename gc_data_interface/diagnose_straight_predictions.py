"""Post-hoc diagnostics of a frozen checkpoint, with direct CV plot comparison.

True future labels define the diagnostic groups only; no model is retrained.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_audit import ForecastGRU, evaluate, to_tensors


def angle(a,b):
    denominator=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
    cosine=np.sum(a*b,axis=-1)/np.maximum(denominator,1e-8)
    return np.degrees(np.arccos(np.clip(cosine,-1,1)))


def chord_deviation(path):
    end=path[:,-1]
    length=np.linalg.norm(end,axis=-1)
    cross=path[:,:,0]*end[:,None,1]-path[:,:,1]*end[:,None,0]
    return np.max(np.abs(cross)/np.maximum(length[:,None],1e-8),axis=1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--results',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    arrays=np.load(args.cache,allow_pickle=False)
    data=to_tensors({k:arrays[k] for k in arrays.files})
    ck=torch.load(args.results/'social16.pt',map_location='cpu',weights_only=True)
    c=ck['config']
    model=ForecastGRU(hidden=c['hidden'],neighbors=c['neighbors'],scale=c['scale'])
    model.load_state_dict(ck['state_dict']);model.eval()
    assert model.residual is False
    with torch.no_grad():
        pred=np.concatenate([model(data['obs'][i:i+1024],data['neighbors'][i:i+1024],data['neighbor_mask'][i:i+1024]).numpy()
                             for i in range(0,len(data['obs']),1024)])*1920
    history=arrays['obs'][:,:,:2]*1920
    true=arrays['future']*1920
    last_velocity=arrays['obs'][:,-1,2:]*1920
    cv=last_velocity[:,None]*np.arange(1,7,dtype=np.float32)[None,:,None]
    # Smooth observed velocity by using all five history points; future terminal
    # direction uses its last three intervals (2.4 seconds), reducing point jitter.
    past_velocity=(history*np.arange(-2,3,dtype=np.float32)[None,:,None]).sum(axis=1)/10
    terminal_velocity=(true[:,-1]-true[:,-4])/3
    turning=angle(past_velocity,terminal_velocity)
    moving=(np.linalg.norm(past_velocity,axis=-1)>=5)&(np.linalg.norm(terminal_velocity,axis=-1)>=5)
    groups={'straight_le15deg':moving&(turning<=15),
            'gentle_15to30deg':moving&(turning>15)&(turning<30),
            'turn_ge30deg':moving&(turning>=30),
            'slow_or_stop':~moving}
    cv_errors=np.linalg.norm(cv-true,axis=-1)
    model_errors=np.linalg.norm(pred-true,axis=-1)
    stats={
        'checkpoint':'social16.pt; fixed seed 42; no retraining',
        'definition':'Turn angle between OLS history velocity and last 2.4s true-future velocity; moving requires both speeds >=5 pixels per 0.8s. Groups use true future only for post-hoc diagnosis.',
        'groups':{},
        'median_max_chord_deviation_px':{
            'true_future':float(np.median(chord_deviation(true))),
            'gru_future':float(np.median(chord_deviation(pred))),
            'constant_velocity':float(np.median(chord_deviation(cv))),
        },
    }
    for name,mask in groups.items():
        stats['groups'][name]={
            'count':int(mask.sum()),'fraction':float(mask.mean()),
            'cv_ade_px':float(cv_errors[mask].mean()),'gru_ade_px':float(model_errors[mask].mean()),
            'cv_fde_px':float(cv_errors[mask,-1].mean()),'gru_fde_px':float(model_errors[mask,-1].mean()),
            'true_chord_deviation_median_px':float(np.median(chord_deviation(true)[mask])),
            'gru_chord_deviation_median_px':float(np.median(chord_deviation(pred)[mask])),
        }
    active=np.linalg.norm(last_velocity,axis=-1)>=5
    stats['gru_vs_cv_endpoint_direction_difference_median_deg']=float(np.median(angle(pred[active,-1],cv[active,-1])))
    stats['gru_vs_cv_endpoint_distance_ratio_median']=float(np.median(
        np.linalg.norm(pred[active,-1],axis=-1)/np.linalg.norm(cv[active,-1],axis=-1)))
    u=last_velocity[active]/np.linalg.norm(last_velocity[active],axis=-1,keepdims=True)
    stats['mean_absolute_error_components_px']={}
    for name,guess in [('cv',cv),('gru',pred)]:
        e=guess[active]-true[active]
        parallel=np.sum(e*u[:,None],axis=-1)
        perpendicular=e[:,:,0]*u[:,None,1]-e[:,:,1]*u[:,None,0]
        stats['mean_absolute_error_components_px'][name]={
            'along_last_motion':float(np.abs(parallel).mean()),
            'across_last_motion':float(np.abs(perpendicular).mean()),
        }
    (args.results/'turn_diagnostics.json').write_text(json.dumps(stats,indent=2)+'\n')
    first_errors=model_errors[:200].mean(axis=1)
    chosen=np.argsort(first_errors)[np.linspace(0,199,6,dtype=int)]
    fig,axes=plt.subplots(2,3,figsize=(13,8.5),layout='constrained')
    for ax,i in zip(axes.flat,chosen):
        ax.plot(history[i,:,0],history[i,:,1],'o-',color='#64748b',label='Observed',markersize=4)
        for points,color,label,style in [(true[i],'#16a34a','Ground truth','o-'),
                                         (cv[i],'#2563eb','Constant velocity','--'),
                                         (pred[i],'#ea580c','GRU + social','o-')]:
            points=np.vstack(([0,0],points))
            ax.plot(points[:,0],points[:,1],style,color=color,label=label,markersize=3)
        ax.set_title(f'Example {i}: ADE GRU {model_errors[i].mean():.1f} / CV {cv_errors[i].mean():.1f} px',fontsize=10)
        ax.set_xlabel('Relative x (pixels)');ax.set_ylabel('Relative y (pixels)')
        ax.invert_yaxis();ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Same frozen test examples: learned predictor versus constant-velocity extrapolation',fontsize=14)
    fig.savefig(args.results/'trajectory_vs_constant_velocity.png',dpi=170)
    plt.close(fig)
    print(json.dumps(stats,indent=2),flush=True)


if __name__=='__main__':main()
