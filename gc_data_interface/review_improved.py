"""Create honest top-1 versus best-of-K comparison figures and diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def chord_deviation(path):
    end=path[:,-1]
    cross=path[:,:,0]*end[:,None,1]-path[:,:,1]*end[:,None,0]
    return np.max(np.abs(cross)/np.linalg.norm(end,axis=-1)[:,None].clip(1e-8),axis=1)


def direction_angle(history,path):
    past=(history*np.arange(-2,3,dtype=np.float32)[None,:,None]).sum(axis=1)/10
    terminal=(path[:,-1]-path[:,-4])/3
    denominator=np.linalg.norm(past,axis=-1)*np.linalg.norm(terminal,axis=-1)
    return np.degrees(np.arccos(np.clip((past*terminal).sum(axis=-1)/denominator.clip(1e-8),-1,1)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--model',help='Optional named top-1 model, e.g. position_turn3')
    args=parser.parse_args();root=args.results
    summary=json.loads((root/'summary.json').read_text())
    focused_path=root/'turn_focused_test.json'
    if focused_path.exists():
        focused=json.loads(focused_path.read_text())
        summary['test'][focused['name']]=focused['test']
    records=json.loads((root/'validation_results.json').read_text())
    best=args.model or summary['selection']['single_best'];multi=summary['selection']['multimodal_best']
    prefix=f'{best}_' if args.model else ''
    a=np.load(root/f'{best}_test_predictions.npz')
    paths=a['paths_pixels'];scores=a['scores']
    top1=paths[np.arange(len(paths)),scores.argmax(axis=-1)]
    true=a['future_pixels'];old=a['old_pixels'];history=a['obs_pixels'];turn=a['turning']
    old_ade=np.linalg.norm(old-true,axis=-1).mean(axis=-1)
    new_ade=np.linalg.norm(top1-true,axis=-1).mean(axis=-1)
    chosen=np.argsort(old_ade[:200])[np.linspace(0,199,6,dtype=int)]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(13,8.5),layout='constrained')
    for ax,i in zip(axes.flat,chosen):
        ax.plot(history[i,:,0],history[i,:,1],'o-',color='#64748b',label='Observed',markersize=4)
        for points,color,label,style in [(true[i],'#16a34a','Ground truth','o-'),
                                         (old[i],'#f59e0b','Old GRU','--'),
                                         (top1[i],'#7c3aed','Improved top-1','o-')]:
            points=np.vstack(([0,0],points))
            ax.plot(points[:,0],points[:,1],style,color=color,label=label,markersize=3)
        ax.set_title(f'Same example {i}\nADE old {old_ade[i]:.1f} / new {new_ade[i]:.1f} px',fontsize=11)
        ax.set_xlabel('Relative x (pixels)');ax.set_ylabel('Relative y (pixels)')
        ax.invert_yaxis();ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Same six previously shown examples: highest-scored trajectory, no oracle selection',fontsize=14)
    fig.savefig(root/f'{prefix}same_examples_top1.png',dpi=170);plt.close(fig)

    m=np.load(root/f'{multi}_test_predictions.npz')
    mp=m['paths_pixels'];ms=m['scores'];mtop=mp[np.arange(len(mp)),ms.argmax(axis=-1)]
    distance=np.linalg.norm(mp-m['future_pixels'][:,None],axis=-1).mean(axis=-1)
    choices=np.flatnonzero(m['turning'][:200])[:6]
    fig,axes=plt.subplots(2,3,figsize=(13,8.5),layout='constrained')
    for ax,i in zip(axes.flat,choices):
        for path in mp[i]:
            path=np.vstack(([0,0],path));ax.plot(path[:,0],path[:,1],color='#93c5fd',alpha=.45,lw=.8)
        ax.plot(m['obs_pixels'][i,:,0],m['obs_pixels'][i,:,1],'o-',color='#64748b',label='Observed',markersize=3)
        for points,color,label in [(m['future_pixels'][i],'#16a34a','Ground truth'),(mtop[i],'#7c3aed','Model top-1')]:
            points=np.vstack(([0,0],points));ax.plot(points[:,0],points[:,1],'o-',color=color,label=label,markersize=3,lw=2)
        ax.set_title(f'Turn example {i}\nTop-1 {distance[i,ms[i].argmax()]:.1f} / oracle min@20 {distance[i].min():.1f} px',fontsize=10)
        ax.set_xlabel('Relative x (pixels)');ax.set_ylabel('Relative y (pixels)')
        ax.invert_yaxis();ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('First six turning examples in a fixed random subset: all 20 candidates and model-selected top-1',fontsize=13)
    fig.savefig(root/'candidate_examples.png',dpi=170);plt.close(fig)

    labels=['Old GRU','Old long/full GRU',best]
    metrics=[summary['test']['old_gru'],summary['test']['old_long_full_gru'],summary['test'][best]]
    if multi!=best:labels.append(multi);metrics.append(summary['test'][multi])
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,key,title in [(axes[0],'top1_ade_px','All windows'),(axes[1],'turn_top1_ade_px','Turning windows (>=30 degrees)')]:
        values=[x[key] for x in metrics]
        ax.barh(np.arange(len(labels)),values,color=['#cbd5e1','#94a3b8']+['#7c3aed']*(len(labels)-2))
        ax.set_yticks(np.arange(len(labels)),labels=labels);ax.invert_yaxis()
        for i,v in enumerate(values):ax.text(v+.25,i,f'{v:.2f}',va='center')
        ax.set_xlim(0,max(values)*1.15);ax.set_xlabel('Top-1 ADE (pixels)');ax.set_title(title);ax.grid(axis='x',alpha=.2)
    fig.suptitle('Development-set comparison: one model-selected trajectory per window',fontsize=14)
    fig.savefig(root/f'{prefix}top1_comparison.png',dpi=170);plt.close(fig)

    diagnostics={
        'chosen_model':best,'median_chord_deviation_px':{
            'ground_truth':float(np.median(chord_deviation(true))),
            'old':float(np.median(chord_deviation(old))),
            'new_top1':float(np.median(chord_deviation(top1)))},
        'turn_direction_change_recall':{
            'old':float((direction_angle(history,old)[turn]>=30).mean()),
            'new_top1':float((direction_angle(history,top1)[turn]>=30).mean())},
        'same_example_indices':chosen.tolist(),
        'same_example_old_ade':old_ade[chosen].tolist(),
        'same_example_new_top1_ade':new_ade[chosen].tolist(),
        'note':'Post-hoc diagnostic grouping uses ground truth; top-1 selection always uses model scores only.'}
    (root/f'{prefix}diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    print(json.dumps(diagnostics,indent=2))


if __name__=='__main__':main()
