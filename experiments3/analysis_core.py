"""CPU analysis for block utility, equal-budget replacements and ablations."""
import math
import numpy as np


def ranks(values):
    values=np.asarray(values,dtype=float)
    if not np.isfinite(values).all(): raise ValueError('Non-finite ranking values')
    order=np.argsort(values,kind='stable'); result=np.empty(len(values),dtype=float)
    start=0
    while start<len(values):
        end=start+1
        while end<len(values) and values[order[end]]==values[order[start]]: end+=1
        result[order[start:end]]=(start+end-1)/2+1
        start=end
    return result


def spearman(x,y):
    if len(x)<3: return None
    a,b=ranks(x),ranks(y)
    if np.std(a)==0 or np.std(b)==0: return None
    return float(np.corrcoef(a,b)[0,1])


def stable_top(scores, candidates, n):
    return sorted(candidates,key=lambda k:(-float(scores[k]),int(k)))[:n]


def fixed_vertical_diagonal_kernel(size=7):
    if size<1 or size%2==0: raise ValueError('Kernel size must be positive and odd')
    kernel=np.eye(size,dtype=np.float32)
    kernel[:,size//2]=1
    return kernel


def row_swap(mask,q,removed,added):
    mask=np.asarray(mask,dtype=bool)
    if not 0<=removed<=q or not 0<=added<=q or not mask[removed] or mask[added]:
        raise ValueError('Swap requires a selected block and an unselected causal block')
    out=mask.copy();out[removed]=False;out[added]=True
    assert out.sum()==mask.sum() and np.count_nonzero(out!=mask)==2
    return out


def compare_sets(a,b,q,initial,learned,all_pairs=False):
    a,b=np.asarray(a,dtype=bool),np.asarray(b,dtype=bool)
    if a.shape!=b.shape or a.sum()!=b.sum() or a[q+1:].any() or b[q+1:].any():
        raise ValueError('Selections must have identical causal block budgets')
    removed=stable_top(initial,np.flatnonzero(a & ~b).tolist(),len(a))
    added=stable_top(learned,np.flatnonzero(b & ~a).tolist(),len(b))
    assert len(removed)==len(added)
    pairs=[(r,k) for r in removed for k in added] if all_pairs else list(zip(removed,added))
    return dict(common=np.flatnonzero(a & b).tolist(),removed=removed,added=added,pairs=pairs)


def ranking_summary(trials,top_sizes,tolerance):
    if not trials: raise ValueError('No candidate trials')
    initial=[r['initial_score'] for r in trials]; learned=[r['learned_score'] for r in trials]
    utility=[r['utility'] for r in trials]
    out={'candidate_count':len(trials),'initial_spearman':spearman(initial,utility),
         'learned_spearman':spearman(learned,utility),'top_candidates':[]}
    for requested in top_sizes:
        n=min(requested,len(trials))
        entry={'requested_pool_top_n':requested,'actual_pool_top_n':n}
        for method,key in [('initial','initial_score'),('learned','learned_score')]:
            chosen=sorted(trials,key=lambda r:(-r[key],r['key_block']))[:n]
            entry[method+'_mean_utility']=float(np.mean([r['utility'] for r in chosen]))
            entry[method+'_improvement_fraction']=float(np.mean([r['utility']>tolerance for r in chosen]))
        out['top_candidates'].append(entry)
    return out


def input_cluster_summary(records,key,seed=42,draws=2000):
    """Average within independent inputs before bootstrap; do not count blocks as replicates."""
    grouped={}
    for r in records:
        value=r.get(key)
        if value is not None and math.isfinite(value):
            grouped.setdefault(r['input_id'],[]).append(value)
    values=np.array([np.mean(v) for v in grouped.values()])
    if not len(values): return {'mean':None,'n_inputs':0,'ci95':None}
    ci=None
    if len(values)>1:
        rng=np.random.default_rng(seed)
        means=np.mean(rng.choice(values,size=(draws,len(values)),replace=True),axis=1)
        ci=np.quantile(means,[.025,.975]).tolist()
    return {'mean':float(values.mean()),'n_inputs':len(values),'ci95':ci,
            'unit':'independent input; layers/heads/query rows averaged within input'}
