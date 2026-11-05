"""Render standalone charts after train_audit.py completes."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    root = args.results
    summary = json.loads((root/"summary.json").read_text())
    results = json.loads((root/"validation_results.json").read_text())
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig, axes = plt.subplots(4,2,figsize=(12,13),layout="constrained")
    for ax, run in zip(axes.flat,results[:8]):
        trace = run["trace"]
        ax.plot([t["step"] for t in trace],[t["train_ade_px"] for t in trace],label="Train probe")
        ax.plot([t["step"] for t in trace],[t["val_ade_px"] for t in trace],label="Validation probe")
        ax.set_title(run["name"])
        ax.set_xlabel("Optimizer steps")
        ax.set_ylabel("ADE (pixels)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("GC training audit: held-out validation curves",fontsize=16)
    fig.savefig(root/"learning_curves.png",dpi=160)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
    labels=list(summary["test_baselines"])+["GRU + social (3 seeds)"]
    for ax,key,mean,std in [(axes[0],"ade_px","test_gru_ade_mean","test_gru_ade_std"),
                            (axes[1],"fde_px","test_gru_fde_mean","test_gru_fde_std")]:
        values=[v[key] for v in summary["test_baselines"].values()]+[summary[mean]]
        ax.barh(np.arange(len(labels)),values,color=["#cbd5e1"]*4+["#2563eb"])
        ax.errorbar(values[-1],len(labels)-1,xerr=summary[std],fmt="none",ecolor="#111827",capsize=4)
        ax.set_yticks(np.arange(len(labels)),labels=[s.replace("_"," ") for s in labels])
        ax.set_xlabel("ADE (pixels)" if key=="ade_px" else "FDE (pixels)")
        ax.invert_yaxis()
        ax.grid(axis="x",alpha=.2)
        for i,v in enumerate(values):
            ax.text(v+.3,i,f"{v:.2f}",va="center",fontsize=9)
        ax.set_xlim(0,max(values)*1.22)
    fig.suptitle("Final test evaluation: configurations selected using validation only",fontsize=14)
    fig.savefig(root/"test_comparison.png",dpi=170)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,5),layout="constrained")
    labels=[r["name"] for r in results[:8]]
    values=[r["validation"]["ade_px"] for r in results[:8]]
    ax.barh(np.arange(8),values,color=["#2563eb" if s==summary["selected_config"]["name"] else "#93c5fd" for s in labels])
    ax.set_yticks(np.arange(8),labels=labels)
    ax.invert_yaxis();ax.set_xlabel("Full validation ADE (pixels; lower is better)")
    ax.set_title("Parameter comparisons at a matched number of sampled training examples")
    for i,v in enumerate(values):ax.text(v+.15,i,f"{v:.2f}",va="center")
    ax.set_xlim(0,max(values)*1.15);ax.grid(axis="x",alpha=.2)
    fig.savefig(root/"validation_comparison.png",dpi=170)
    plt.close(fig)

    a=np.load(root/"test_example_predictions.npz")
    obs=a["obs"][:,:,:2]*1920;future=a["future"]*1920;pred=a["prediction"]*1920
    ade=np.linalg.norm(pred-future,axis=-1).mean(axis=-1)
    ordering=np.argsort(ade)
    chosen=ordering[np.linspace(0,len(ordering)-1,6,dtype=int)]
    fig,axes=plt.subplots(2,3,figsize=(12,8),layout="constrained")
    for ax,i in zip(axes.flat,chosen):
        ax.plot(obs[i,:,0],obs[i,:,1],"o-",color="#64748b",label="Observed")
        for points,color,label in [(future[i],"#16a34a","Ground truth"),(pred[i],"#ea580c","GRU + social")]:
            points=np.vstack(([0,0],points))
            ax.plot(points[:,0],points[:,1],"o-",color=color,label=label,markersize=3)
        ax.set_title(f"Test example {int(i)} | ADE {ade[i]:.1f} px")
        ax.set_xlabel("Relative x (pixels)");ax.set_ylabel("Relative y (pixels)")
        ax.invert_yaxis();ax.axis("equal");ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Examples across error quantiles of 200 fixed random test windows",fontsize=14)
    fig.savefig(root/"trajectory_examples.png",dpi=160)
    plt.close(fig)
    print("Saved four chart PNGs to",root)


if __name__=="__main__":main()
