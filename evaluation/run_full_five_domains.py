"""CIRRUS++ synthetic 5-domain exhaustive benchmark.

Reuses EXACT synthetic generators and nested corruption from phase 1 (cirrus_eval.py).
Implements all upstream CIRRUS++ 336 cleaning configurations using the locally
transcribed stage functions, plus MICE/none and reverse order. Scaling is disabled.
Does not import original cirruspp_core.py (not locally available): upstream parity remains unverified.
"""
from __future__ import annotations
import argparse
from itertools import product
from pathlib import Path
import os, time, warnings, json, platform
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import sklearn, scipy
from cirrus_eval import (generate_domain_series, corrupt_nested, aggregate_profile,
                         quality_components, apply_imputation as core_impute,
                         apply_outlier_handling, apply_smoothing, DOMAIN_WEIGHTS)

HERE=Path(__file__).resolve().parent
DOMAINS=['industrial','energy','physiological','finance','network']
IMP=['none','mean','median','locf','linear','knn','mice']
OUT=['none','zscore','iqr','mad','winsorize','isolation_forest']
SMOOTH=['none','moving_median','savgol','butterworth']
ORDERS=['imputation_outlier_smoothing_scaling','outlier_imputation_smoothing_scaling']
COMP=['completeness','continuity','plausibility','distribution','stability','correlation']


def original_equivalent_impute(x,method,seed=42):
    if method=='none': return x.copy()
    if method=='mice':
        obj=IterativeImputer(max_iter=8,random_state=seed,sample_posterior=False,
                             initial_strategy='median',skip_complete=True)
        return pd.DataFrame(obj.fit_transform(x),index=x.index,columns=x.columns)
    return core_impute(x,method)


def q_profile(frame,fs,domain):
    p=aggregate_profile(frame,fs)
    c=quality_components(p)
    w=np.asarray(DOMAIN_WEIGHTS[domain],float)
    return float(c@w / w.sum()),dict(zip(COMP,map(float,c)))


def evaluate_rmse(clean,proc):
    sig=[c for c in clean.columns if c!='timestamp']
    y=clean[sig].to_numpy(float); z=proc[sig].to_numpy(float)
    nanfraction=float((~np.isfinite(z)).mean())
    if not np.isfinite(z).all():
        return {'nrmse':np.nan, 'mean_raw_rmse':np.nan, 'remaining_missing_fraction':nanfraction,'valid_rmse':False}
    dif=z-y
    perraw=np.sqrt(np.mean(dif**2,axis=0))
    nrmse=perraw/(np.std(y,axis=0)+1e-12)
    return {'nrmse':float(nrmse.mean()),'mean_raw_rmse':float(perraw.mean()),'remaining_missing_fraction':nanfraction,'valid_rmse':True,
            **{f'nrmse_{c}':float(v) for c,v in zip(sig,nrmse)},
            **{f'rmse_{c}':float(v) for c,v in zip(sig,perraw)}}


def stage_cache(x,fs):
    imputed={im:original_equivalent_impute(x,im,42) for im in IMP}
    out_default={(im,o):apply_outlier_handling(imputed[im],o,seed=42) for im in IMP for o in OUT}
    out_raw={o:apply_outlier_handling(x,o,seed=42) for o in OUT}
    out_reverse={(im,o):original_equivalent_impute(out_raw[o],im,42) for im in IMP for o in OUT}
    return dict(zip(ORDERS,(out_default,out_reverse)))


def scenario(domain,seed,level,n):
    t0=time.monotonic()
    clean,fs=generate_domain_series(domain,n,seed)
    dirty=corrupt_nested(clean,level/100,seed+1000) if level else clean.copy()
    qc,c_c=q_profile(clean,fs,domain)
    qd,c_d=q_profile(dirty,fs,domain)
    baseline={'domain':domain,'seed':seed,'severity_pct':level,'n':n,'fs':fs,'clean_q':qc,'corrupted_q':qd,
              **{f'clean_{k}':v for k,v in c_c.items()},**{f'corrupted_{k}':v for k,v in c_d.items()}}
    cols=[c for c in clean.columns if c!='timestamp']
    cache=stage_cache(dirty[cols],fs)
    rows=[]
    for order in ORDERS:
        for im,o in product(IMP,OUT):
            x=cache[order][(im,o)]
            for sm in SMOOTH:
                z=apply_smoothing(x,sm,fs)
                proc=dirty.copy();proc[cols]=z
                q,comps=q_profile(proc,fs,domain)
                err=evaluate_rmse(clean,proc)
                rows.append({'domain':domain,'seed':seed,'severity_pct':level,'order':order,
                             'imputation':im,'outlier':o,'smoothing':sm,'scaling':'none',
                             'q_score':q,'delta_q':q-qd,**comps,**err})
    assert len(rows)==336
    df=pd.DataFrame(rows)
    valid=df[df['valid_rmse']].copy()
    rho=spearmanr(valid.q_score,valid.nrmse).statistic if valid.q_score.nunique()>1 else np.nan
    maxq=valid.q_score.max(); minrm=valid.nrmse.min()
    topq=valid[valid.q_score.ge(maxq-1e-9)]
    # Defined deterministic selection for score ties, but evaluate tie-aware hits separately.
    qw=valid.sort_values(['q_score','nrmse'],ascending=[False,True],kind='stable').iloc[0]
    rw=valid.sort_values(['nrmse','q_score'],ascending=[True,False],kind='stable').iloc[0]
    def id_of(row): return f"{row['order']} / {row['imputation']} / {row['outlier']} / {row['smoothing']}"
    summ={'domain':domain,'seed':seed,'severity_pct':level,'n_candidates':len(df),'n_valid_rmse':len(valid),
          'n_invalid_rmse':len(df)-len(valid),'spearman_q_nrmse':float(rho),
          'quality_max':float(maxq),'qmax_tie_count':len(topq),
          'qmax_ties_include_rmse_min':bool((topq.nrmse <= minrm+1e-12).any()),
          'best_q_pipeline':id_of(qw),'nrmse_best_q':float(qw.nrmse),
          'best_nrmse_pipeline':id_of(rw),'best_nrmse':float(minrm),'q_of_best_nrmse':float(rw.q_score),
          'nrmse_regret_abs':float(qw.nrmse-minrm),
          'nrmse_regret_ratio':float(qw.nrmse/minrm) if minrm>1e-10 else np.nan,
          'top5_contains_optimum':bool((valid.nlargest(5,'q_score').nrmse<=minrm+1e-12).any()),
          'top10_contains_optimum':bool((valid.nlargest(10,'q_score').nrmse<=minrm+1e-12).any()),
          'elapsed_seconds':round(time.monotonic()-t0,3)}
    crows=[]
    for comp in COMP:
        g=valid[[comp,'nrmse']].dropna()
        r=float(spearmanr(g[comp],g.nrmse).statistic) if g[comp].nunique()>1 else np.nan
        crows.append({'domain':domain,'seed':seed,'severity_pct':level,'component':comp,'spearman_component_nrmse':r,
                      'component_weight':float(DOMAIN_WEIGHTS[domain][COMP.index(comp)])})
    return df,pd.DataFrame([baseline]),pd.DataFrame([summ]),pd.DataFrame(crows)


def main(domain,seed,levels,n,output):
    output.mkdir(exist_ok=True,parents=True)
    rs=[];bs=[];ss=[];cs=[]
    for lev in levels:
        a,b,s,c=scenario(domain,seed,lev,n)
        rs.append(a);bs.append(b);ss.append(s);cs.append(c)
        print(json.dumps({'domain':domain,'seed':seed,'severity':lev,
                          'rho':round(s.spearman_q_nrmse.iloc[0],4),
                          'hit':bool(s.qmax_ties_include_rmse_min.iloc[0]),
                          'elapsed_sec':s.elapsed_seconds.iloc[0]}),flush=True)
    pd.concat(rs,ignore_index=True).to_csv(output/f'results_{domain}_{seed}.csv',index=False)
    pd.concat(bs,ignore_index=True).to_csv(output/f'baselines_{domain}_{seed}.csv',index=False)
    pd.concat(ss,ignore_index=True).to_csv(output/f'summary_{domain}_{seed}.csv',index=False)
    pd.concat(cs,ignore_index=True).to_csv(output/f'components_{domain}_{seed}.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--domain',choices=DOMAINS,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--levels',type=int,nargs='+',default=[0,5,10,15]);p.add_argument('--n',type=int,default=480)
    p.add_argument('--output',type=Path,default=HERE/'runs')
    args=p.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        warnings.simplefilter('ignore',RuntimeWarning)
        main(args.domain,args.seed,args.levels,args.n,args.output)
