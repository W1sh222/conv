"""Replot saved results only: no model loading, inference or invented observations."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

COLORS={'initial':'#547FA5','positive_1x1':'#87919A',
        'fixed_vertical_diagonal':'#B4946D','learned_5000':'#DB985D'}
LABELS={'initial':'Initial scores','positive_1x1':'1x1 control',
        'fixed_vertical_diagonal':'Fixed vertical + diagonal','learned_5000':'ConvPrefill'}


def style():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':9,'axes.labelsize':10,'axes.titlesize':10,
                         'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})


def save(fig,path):
    fig.tight_layout(pad=1.2)
    fig.savefig(path.with_suffix('.png'),dpi=600,facecolor='white')
    fig.savefig(path.with_suffix('.pdf'),facecolor='white')
    fig.savefig(path.with_suffix('.svg'),facecolor='white')
    plt.close(fig)


def figure():
    return plt.subplots(figsize=(3.5,3.1))


def plot(root,kind):
    root=Path(root);style()
    cases=[]
    for p in sorted(root.glob('*/experiment.json')):
        d=json.loads(p.read_text(encoding='utf-8'))
        if d['status']=='complete': cases.append((p.parent,d))
    if not cases: raise ValueError('No complete experiment.json cases under '+str(root))
    out=root/'plots';out.mkdir(exist_ok=True)
    if kind==3:
        # Preserve candidate-level data in each case. These dots are NOT independent inputs.
        for directory,d in cases:
            trials=[json.loads(x) for x in (directory/'trials.jsonl').read_text().splitlines()]
            fig,ax=figure()
            for key,color,marker in [('initial_score',COLORS['initial'],'o'),('learned_score',COLORS['learned_5000'],'s')]:
                ax.scatter([r[key] for r in trials],[r['utility'] for r in trials],s=34,
                           color=color,marker=marker,edgecolors='white',linewidths=.5,
                           label='Initial' if key=='initial_score' else 'ConvPrefill',alpha=.8)
            ax.axhline(0,color='#777777',ls='--',lw=.8)
            ax.set(xlabel='Candidate block score',ylabel='Replacement utility (NLL decrease)')
            ax.legend(frameon=True);save(fig,out/(directory.name+'_score_utility'))
        fig,ax=figure();valid=0
        for directory,d in cases:
            m=d['metrics'];x=m['initial_spearman'];y=m['learned_spearman']
            if x is None or y is None: continue
            valid+=1;ax.plot([0,1],[x,y],color='#AAAAAA',lw=.7,alpha=.6)
            ax.scatter([0,1],[x,y],s=38,c=[COLORS['initial'],COLORS['learned_5000']],zorder=3)
        ax.set(xticks=[0,1],xticklabels=['Initial','ConvPrefill'],ylabel='Spearman correlation with utility',ylim=(-1.05,1.05))
        ax.axhline(0,color='#777777',ls='--',lw=.8)
        ax.set_title(f'{valid} paired cases; see input-level summary.json')
        save(fig,out/'rank_correlation')
        fig,ax=figure()
        for _,d in cases:
            entries=d['metrics']['top_candidates'];x=[v['actual_pool_top_n'] for v in entries]
            for method,color,marker in [('initial',COLORS['initial'],'o'),('learned',COLORS['learned_5000'],'s')]:
                ax.plot(x,[v[method+'_mean_utility'] for v in entries],c=color,marker=marker,lw=.8,alpha=.6)
        ax.axhline(0,color='#777777',ls='--',lw=.8)
        ax.set(xlabel='Number of top-ranked candidates',ylabel='Mean candidate utility')
        ax.legend(handles=[Patch(facecolor=COLORS['initial'],label='Initial'),Patch(facecolor=COLORS['learned_5000'],label='ConvPrefill')],frameon=True)
        save(fig,out/'top_candidates_utility')
    elif kind==4:
        for directory,d in cases:
            z=np.load(directory/'block_maps.npz');a=z['initial_mask'];b=z['learned_mask'];qb=int(z['query_block'])
            category=np.zeros(a.size,dtype=int)
            category[a&b]=1;category[a&~b]=2;category[b&~a]=3;category[qb+1:]=4
            fig,ax=plt.subplots(figsize=(7.2,1.9))
            colors=['#FFFFFF','#A8CCE3','#EAB076','#769FAA','#EEEEEE']
            ax.imshow(category[None,:],aspect='auto',interpolation='nearest',
                      cmap=ListedColormap(colors),norm=BoundaryNorm(np.arange(-.5,5),5))
            ax.set(xlabel='Key block',yticks=[])
            ax.legend(handles=[Patch(facecolor=colors[i],edgecolor='#777777',label=name)
                               for i,name in enumerate(['Unselected','Common','Removed','Added','Causally masked'])],
                      ncol=3,loc='upper center',bbox_to_anchor=(.5,-.45),frameon=False)
            save(fig,out/(directory.name+'_selection'))
            rows=[json.loads(x) for x in (directory/'trials.jsonl').read_text().splitlines()]
            single=[r['utility'] for r in rows if r['trial_type']=='single_swap']
            fig,ax=figure()
            if single: ax.hist(single,bins=min(16,max(3,int(np.sqrt(len(single))))),color=COLORS['learned_5000'],edgecolor='white')
            ax.axvline(0,color='#777777',ls='--',lw=.9)
            ax.set(xlabel='Single-swap utility (NLL decrease)',ylabel='Number of swaps')
            save(fig,out/(directory.name+'_swap_utility'))
        fig,ax=figure()
        values=[d['metrics']['whole_row_utility'] for _,d in cases]
        ax.scatter(range(1,len(values)+1),values,s=42,color=COLORS['learned_5000'],marker='s')
        ax.axhline(0,color='#777777',ls='--',lw=.8)
        ax.set(xlabel='Case',ylabel='Whole-row utility (NLL decrease)')
        save(fig,out/'whole_row_utility')
    else:
        policies=list(COLORS)
        fig,ax=plt.subplots(figsize=(7.2,3.4))
        for _,d in cases:
            ys=[d['metrics']['losses'][p] for p in policies]
            ax.plot(range(4),ys,c='#BBBBBB',lw=.65,alpha=.65)
            for i,p in enumerate(policies): ax.scatter(i,ys[i],s=38,c=COLORS[p],marker=['o','D','^','s'][i],zorder=3)
        ax.set(xticks=range(4),xticklabels=['Initial','1x1 control','Fixed V + D','ConvPrefill'],ylabel='Ground-truth answer NLL')
        ax.set_title(f'{len(cases)} inputs; each line connects the same input')
        save(fig,out/'neighborhood_ablation_nll')
    (out/'plot_notes.json').write_text(json.dumps({
        'experiment':kind,'complete_cases':len(cases),'backend':'matplotlib',
        'alignment':'Not applicable: each export is a single panel',
        'uncertainty':'Paired case observations shown; input-cluster bootstrap intervals are in summary.json. No significance claims.',
        'source':'../*/trials.csv and ../*/experiment.json; ../*/block_maps.npz',
        'qa':'Inspect exported PDFs/PNGs after inference at final paper size; plotting alone does not verify scientific outcomes.'
    },indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--experiment',type=int,choices=[3,4,5],required=True)
    a=p.parse_args();plot(a.input,a.experiment)
