"""Exploratory CIRRUS++ quality-index vs synthetic reconstruction experiment.
Quality profiling and pipeline methods transcribed from github.com/eyetracking-data/CIRRUS_Plus_Plus/blob/main/cirruspp_core.py as read 2026-11-03.
Synthetic eye_tracking/industrial/finance match repository generator structure; others extended.
"""
import numpy as np, pandas as pd, itertools
from scipy.stats import median_abs_deviation, skew, kurtosis, spearmanr
from scipy.signal import welch, medfilt, savgol_filter, butter, filtfilt
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest

DOMAIN_WEIGHTS={
'eye_tracking': [0.23,0.24,0.17,0.12,0.16,0.08],
'industrial': [0.17,0.16,0.23,0.08,0.25,0.11],
'energy': [0.20,0.16,0.12,0.10,0.30,0.12],
'physiological': [0.17,0.22,0.12,0.09,0.30,0.10],
'finance': [0.14,0.10,0.08,0.30,0.30,0.08],
'network': [0.18,0.15,0.20,0.07,0.27,0.13],
}

def _runs(mask):
    if mask.size==0 or not mask.any():return np.array([],dtype=int)
    padded=np.r_[False,mask,False].astype(int)
    edges=np.flatnonzero(np.diff(padded))
    return edges[1::2]-edges[::2]

def _robust_outlier_rate(x,threshold=3.5):
    med=np.nanmedian(x); mad=median_abs_deviation(x,nan_policy='omit',scale=1.0)
    if not np.isfinite(mad) or mad<=0:return 0.0
    rz=.67448975*(x-med)/mad
    return float(np.nanmean(np.abs(rz)>threshold))

def _iqr_outlier_rate(x,factor=1.5):
    q1,q3=np.nanpercentile(x,[25,75]); iqr=q3-q1
    if not np.isfinite(iqr) or iqr<=0:return 0.0
    return float(np.nanmean((x<q1-factor*iqr)|(x>q3+factor*iqr)))

def _spectral_entropy(x,fs):
    x=pd.Series(x).interpolate(limit_direction='both').fillna(0).to_numpy(float)
    if len(x)<16 or np.nanstd(x)==0:return 0.0,0.0
    _,p=welch(x-np.mean(x),fs=max(fs,1e-9),nperseg=min(256,len(x)))
    p=np.maximum(p,0)
    if p.sum()<=0:return 0.0,0.0
    p=p/p.sum(); ent=-np.sum(p*np.log(p+1e-12))/np.log(len(p))
    return float(ent),float(p.max())

def _drift_score(x):
    s=pd.Series(x).interpolate(limit_direction='both')
    if len(s)<30 or s.std()==0:return 0.0
    w=max(10,len(s)//10); r=s.rolling(w,min_periods=max(5,w//3)).mean().dropna()
    if r.empty:return 0.0
    return float(np.clip((r.max()-r.min())/(4*s.std()),0,1))

def _feature_profile(s,fs):
    x=pd.to_numeric(s,errors='coerce').to_numpy(float)
    missing=~np.isfinite(x); runs=_runs(missing); valid=x[~missing]; n=len(x)
    if len(valid)<4:
        return dict(n=n,missing_rate=float(missing.mean()),longest_missing_run_seconds=float(runs.max()/fs) if len(runs) else 0.,missing_run_p95_seconds=float(np.quantile(runs,.95)/fs) if len(runs) else 0.,missing_burstiness=0.,outlier_iqr_rate=0.,outlier_mad_rate=0.,skewness=0.,kurtosis_excess=0.,lag1_autocorr=0.,drift_score=0.,local_volatility=0.,spectral_entropy=0.,dominant_frequency_ratio=0.,zero_variance=True)
    run_concentration=float(runs.max()/max(1,runs.sum())) if len(runs) else 0.
    mean_run_ratio=float(runs.mean()/max(1,n)) if len(runs) else 0.
    burst=float(np.clip(.65*run_concentration+8*.35*mean_run_ratio,0,1))
    interp=pd.Series(x).interpolate(limit_direction='both')
    lag1=float(interp.autocorr(lag=1)) if len(interp)>2 else 0.
    if not np.isfinite(lag1):lag1=0.
    local_vol=float(np.nanstd(np.diff(interp))/(np.nanstd(interp)+1e-12))
    ent,dom=_spectral_entropy(x,fs)
    return dict(n=n,missing_rate=float(missing.mean()),longest_missing_run_seconds=float(runs.max()/fs) if len(runs) else 0.,missing_run_p95_seconds=float(np.quantile(runs,.95)/fs) if len(runs) else 0.,missing_burstiness=burst,outlier_iqr_rate=_iqr_outlier_rate(valid),outlier_mad_rate=_robust_outlier_rate(valid),skewness=float(skew(valid,bias=False,nan_policy='omit')),kurtosis_excess=float(kurtosis(valid,fisher=True,bias=False,nan_policy='omit')),lag1_autocorr=lag1,drift_score=_drift_score(x),local_volatility=local_vol,spectral_entropy=ent,dominant_frequency_ratio=dom,zero_variance=bool(np.nanstd(valid)<=1e-12))

def aggregate_profile(df,fs):
    cols=[c for c in df.columns if c!='timestamp']; corr=df[cols].corr().abs() if len(cols)>1 else None
    rows=[]
    for c in cols:
        row=_feature_profile(df[c],fs)
        if corr is not None:
            vals=corr.loc[c].drop(c,errors='ignore')
            row['corr_mean_abs']=float(vals.mean()) if not vals.empty else 0.
        else:row['corr_mean_abs']=0.
        rows.append(row)
    out=pd.DataFrame(rows); agg={'n_features':len(out)}
    for c in out.columns:
        if c=='zero_variance':continue
        if pd.api.types.is_numeric_dtype(out[c]):
            agg[c]=float(out[c].median()); agg['max_'+c]=float(out[c].max())
    agg['zero_variance_fraction']=float(out['zero_variance'].mean())
    return agg

def quality_components(p):
    missing=p.get('missing_rate',0.); long_gap=p.get('missing_run_p95_seconds',0.); burst=p.get('missing_burstiness',0.)
    outlier=max(p.get('outlier_iqr_rate',0.),p.get('outlier_mad_rate',0.))
    sk=abs(p.get('skewness',0.)); ku=max(0.,p.get('kurtosis_excess',0.));drift=p.get('drift_score',0.);vol=p.get('local_volatility',0.);corr=p.get('corr_mean_abs',0.)
    return np.array([
        100*(1-np.clip(missing/.45,0,1)),
        100*(1-np.clip(.55*burst+.45*long_gap/(1+long_gap),0,1)),
        100*(1-np.clip(outlier/.25,0,1)),
        100*(1-np.clip(.6*sk/3+.4*ku/10,0,1)),
        100*(1-np.clip(.65*drift+.35*vol/(1+vol),0,1)),
        100*np.clip(.25+.75*corr,0,1)
    ])

def quality_index(frame,fs,domain):
    v=quality_components(aggregate_profile(frame,fs));w=np.array(DOMAIN_WEIGHTS[domain])
    return float(np.dot(v,w)/np.sum(w))

def generate_domain_series(domain,n=480,seed=42):
    rng=np.random.default_rng(seed)
    fs=120. if domain=='eye_tracking' else 10. if domain=='industrial' else 50. if domain=='physiological' else 1. if domain=='finance' else 1/60 if domain=='network' else 1/900
    t=np.arange(n)/fs
    if domain=='eye_tracking':
        gx=960+350*np.sin(.8*t)+rng.normal(0,8,n);gy=540+240*np.cos(.6*t)+rng.normal(0,7,n)
        pupil=3.5+.15*np.sin(.2*t)+rng.normal(0,.03,n)
        df=pd.DataFrame({'timestamp':t,'gaze_x':gx,'gaze_y':gy,'pupil_left':pupil,'pupil_right':pupil+rng.normal(0,.02,n)})
    elif domain=='industrial':
        df=pd.DataFrame({'timestamp':t,'temperature':65+2*np.sin(.1*t)+rng.normal(0,.2,n),'pressure':4.5+.15*np.sin(.08*t)+rng.normal(0,.02,n),'vibration':.9+.2*np.abs(np.sin(1.8*t))+rng.normal(0,.03,n),'flow':100+6*np.sin(.1*t)+rng.normal(0,.7,n)})
    elif domain=='finance':
        ret=rng.standard_t(5,n)*.01;price=100*np.exp(np.cumsum(ret))
        df=pd.DataFrame({'timestamp':t,'price':price,'log_return':ret,'volume':np.exp(10+5*np.abs(ret)+rng.normal(0,.3,n))})
    elif domain=='energy':
        # Additional purpose-built 3-channel energy profiles (repository's energy demo is generic sinusoids).
        phase=2*np.pi*np.arange(n)/48.
        base=110+22*np.sin(phase-1.0)+10*np.sin(2*phase)+np.linspace(0,14,n)
        load=base+rng.normal(0,2,n)
        df=pd.DataFrame({'timestamp':t,'power_load':load,'demand':.83*base+12+rng.normal(0,1.8,n),'solar_power':np.clip(35*np.sin(phase-.5),0,None)+rng.normal(0,1,n)})
    elif domain=='physiological':
        # Additional purpose-built waveforms (repo's physiological demo uses generic sinusoids).
        base=np.sin(2*np.pi*1.2*t)+.35*np.sin(2*np.pi*2.4*t)
        df=pd.DataFrame({'timestamp':t,'ecg':base+rng.normal(0,.08,n),'ppg':.7*np.sin(2*np.pi*1.2*t-.4)+rng.normal(0,.06,n),'resp':np.sin(2*np.pi*.23*t)+rng.normal(0,.035,n)})
    else:
        # Additional purpose-built bursty network profile.
        pulse=np.zeros(n)
        for k in [50,125,205,315,405]:
            if k<n:pulse[k:min(n,k+7)]+=np.array([.2,1,2.4,3.4,2.1,1.1,.3])[:min(n-k,7)]
        df=pd.DataFrame({'timestamp':t,'cpu':40+8*np.sin(np.arange(n)/30)+11*pulse+rng.normal(0,1.2,n),'latency':15+4*np.sin(np.arange(n)/21)+20*pulse+rng.normal(0,1.0,n),'traffic':200+25*np.sin(np.arange(n)/24)+35*pulse+rng.normal(0,3.2,n)})
    return df,fs

def corrupt_nested(clean,severity,seed):
    """Exactly approx severity% of each channel: half MCAR, half spikes, nested ranks across severities."""
    r=np.random.default_rng(seed); out=clean.copy();n=len(clean); p=int(round(n*severity));nm=p//2;ns=p-nm
    for c in clean.columns:
        if c=='timestamp':continue
        order=r.permutation(n)
        # Fixed non-overlapping missing and spike ranks across all severity levels.
        # Select missing and spike samples from fixed halves of permutation so nested across levels.
        miss=order[:nm];spike=order[n//2:n//2+ns]
        scale=float(np.std(clean[c])) or 1.
        signs=r.choice([-1.,1.],size=n);amps=r.uniform(4.,9.,size=n)
        out.loc[miss,c]=np.nan
        out.loc[spike,c]=clean.loc[spike,c].to_numpy()+signs[spike]*amps[spike]*scale
    return out

def apply_imputation(x,method):
    if method=='mean':return pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(x),index=x.index,columns=x.columns)
    if method=='median':return pd.DataFrame(SimpleImputer(strategy='median').fit_transform(x),index=x.index,columns=x.columns)
    if method=='locf':return x.ffill().bfill()
    if method=='linear':return x.interpolate(method='linear',limit_direction='both')
    if method=='knn':
        k=max(2,min(7,int(np.sqrt(max(4,len(x))))//4))
        return pd.DataFrame(KNNImputer(n_neighbors=k,weights='distance').fit_transform(x),index=x.index,columns=x.columns)
    raise ValueError(method)

def apply_outlier_handling(x,method,seed=42):
    if method=='none':return x.copy()
    if method=='winsorize':
        out=x.copy()
        for c in x.columns:
            lo,hi=x[c].quantile([.01,.99]);out[c]=x[c].clip(lo,hi)
        return out
    mask=pd.DataFrame(False,index=x.index,columns=x.columns)
    if method=='isolation_forest':
        filled=x.interpolate(limit_direction='both').fillna(x.median()).fillna(0)
        pred=IsolationForest(n_estimators=150,contamination='auto',random_state=seed,n_jobs=1).fit_predict(filled)
        mask.loc[pred==-1,:]=True
    else:
        for c in x.columns:
            s=x[c];v=s.dropna()
            if len(v)<8:continue
            if method=='zscore':
                sd=v.std(ddof=0)
                if sd>0:mask[c]=((s-v.mean()).abs()/sd)>3
            elif method=='iqr':
                q1,q3=v.quantile([.25,.75]);iqr=q3-q1
                if iqr>0:mask[c]=(s<q1-1.5*iqr)|(s>q3+1.5*iqr)
            elif method=='mad':
                med=v.median();mad=median_abs_deviation(v,scale=1.0)
                if mad>0:mask[c]=(0.67448975*(s-med)/mad).abs()>3.5
            else:raise ValueError(method)
    out=x.mask(mask)
    return out.interpolate(method='linear',limit_direction='both').fillna(out.median())

def apply_smoothing(x,method,fs):
    if method=='none':return x.copy()
    out=x.copy()
    for c in x.columns:
        s=x[c].interpolate(limit_direction='both').to_numpy(float)
        if len(s)<9:continue
        if method=='moving_median':
            k=min(9,len(s) if len(s)%2 else len(s)-1);k=max(3,k);out[c]=medfilt(s,kernel_size=k)
        elif method=='savgol':
            w=min(21,len(s) if len(s)%2 else len(s)-1);w=max(5,w)
            out[c]=savgol_filter(s,w,polyorder=min(3,w-2),mode='interp')
        elif method=='butterworth':
            nyq=max(fs/2,1e-9);cutoff=min(.20*fs,.45*fs);wn=cutoff/nyq
            b,a=butter(3,wn,btype='low');padlen=3*(max(len(a),len(b))-1)
            out[c]=filtfilt(b,a,s) if len(s)>padlen else s
        else:raise ValueError(method)
    return out

def execute_pipeline(frame,spec,fs):
    # Same stage order as CIRRUS++ default, scaling=none to retain physical units for RMSE.
    cols=[c for c in frame.columns if c!='timestamp'];x=frame[cols].copy()
    x=apply_imputation(x,spec[0]);x=apply_outlier_handling(x,spec[1]);x=apply_smoothing(x,spec[2],fs)
    out=frame.copy();out[cols]=x;return out

def nrmse(clean,processed):
    """Avg per-column RMSE / clean standard deviation. All time samples, not just corrupted."""
    cols=[c for c in clean.columns if c!='timestamp'];scores=[]
    for c in cols:
        y=clean[c].to_numpy(float);z=processed[c].to_numpy(float)
        scores.append(float(np.sqrt(np.mean((y-z)**2))/(np.std(y)+1e-12)))
    return float(np.mean(scores))

def run(seed=42,domain_names=None,only_levels=None):
    rng=np.random.default_rng(20261103)
    candidates=list(itertools.product(['mean','median','locf','linear','knn'],['none','zscore','iqr','mad','winsorize','isolation_forest'],['none','moving_median','savgol','butterworth']))
    ix=rng.choice(len(candidates),size=24,replace=False)
    specs=[('linear','none','none')]+[candidates[i] for i in ix if candidates[i]!=('linear','none','none')]
    if len(specs)<25:specs.append(next(x for x in candidates if x not in specs))
    assert len(specs)==25 and len(set(specs))==25
    rows=[];baseline=[]
    for domain in (domain_names or list(DOMAIN_WEIGHTS)):
        clean,fs=generate_domain_series(domain,480,seed)
        q_clean=quality_index(clean,fs,domain)
        for level in (only_levels or [.05,.1,.15]):
            dirty=corrupt_nested(clean,level,seed+1000)
            q_dirty=quality_index(dirty,fs,domain)
            for i,spec in enumerate(specs):
                processed=execute_pipeline(dirty,spec,fs)
                q=quality_index(processed,fs,domain)
                rm=nrmse(clean,processed)
                rows.append(dict(seed=seed,domain=domain,corruption_pct=int(level*100),pipeline_id=i+1,imp=spec[0],out=spec[1],smooth=spec[2],Q=q,delta_Q=q-q_dirty,NRMSE=rm,clean_Q=q_clean,corrupt_Q=q_dirty))
            baseline.append(dict(seed=seed,domain=domain,corruption_pct=int(level*100),clean_Q=q_clean,corrupt_Q=q_dirty))
    return pd.DataFrame(rows),pd.DataFrame(baseline),specs

def summarize(res):
    groups=[]
    for (dom,lev),g in res.groupby(['domain','corruption_pct']):
        idx_q=g.Q.idxmax();idx_rm=g.NRMSE.idxmin()
        rho,p=spearmanr(g.Q,g.NRMSE)
        groups.append(dict(domain=dom,level=lev,rho=rho,p_value=p,best_Q_pipeline=int(g.loc[idx_q,'pipeline_id']),best_Q=float(g.loc[idx_q,'Q']),error_of_best_Q=float(g.loc[idx_q,'NRMSE']),best_RMSE_pipeline=int(g.loc[idx_rm,'pipeline_id']),best_RMSE=float(g.loc[idx_rm,'NRMSE']),regret=float(g.loc[idx_q,'NRMSE']/g.loc[idx_rm,'NRMSE']-1),worst_RMSE=float(g.NRMSE.max())))
    return pd.DataFrame(groups)

if __name__=='__main__':
    import time
    t=__import__('time').time(); r,b,s=run()
    r.to_csv('/mnt/data/cirrus_experiment_results.csv',index=False);b.to_csv('/mnt/data/cirrus_experiment_baselines.csv',index=False)
    sm=summarize(r);sm.to_csv('/mnt/data/cirrus_experiment_summary.csv',index=False)
    pd.DataFrame([dict(id=i+1,imputation=x[0],outlier=x[1],smoothing=x[2],scaling='none') for i,x in enumerate(s)]).to_csv('/mnt/data/cirrus_25_pipelines.csv',index=False)
    print('SECONDS',time.time()-t)
    print('PIPELINES:',list(enumerate(s,1)))
    print('BASELINE\n'+b.round(3).to_string(index=False))
    print('SUMMARY\n'+sm.round(3).to_string(index=False))
    print('OVERALL median rho',sm.rho.median(),'mean',sm.rho.mean(),'Q_win',sum(sm.best_Q_pipeline==sm.best_RMSE_pipeline),'of',len(sm))
