"""Aggregate reproducible 5-domain exhaustive synthetic benchmark results."""
from pathlib import Path
import pandas as pd, numpy as np, json, hashlib, shutil, zipfile, sys, time
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent
OUT=BASE
OUT.mkdir(parents=True,exist_ok=True)
FILES={
 'results':sorted((BASE/'runs').glob('results_*.csv')),
 'summary':sorted((BASE/'runs').glob('summary_*.csv')),
 'baselines':sorted((BASE/'runs').glob('baselines_*.csv')),
 'components':sorted((BASE/'runs').glob('components_*.csv'))
}
for key,files in FILES.items():
 assert len(files)==25,(key,len(files))
 df=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
 df.to_csv(OUT/f'{key}_all.csv',index=False)
 print(key,df.shape)

res=pd.read_csv(OUT/'results_all.csv');sm=pd.read_csv(OUT/'summary_all.csv');bs=pd.read_csv(OUT/'baselines_all.csv');comps=pd.read_csv(OUT/'components_all.csv')
assert len(res)==33600 and len(sm)==100 and len(bs)==100 and len(comps)==600
assert res.groupby(['domain','seed','severity_pct']).size().eq(336).all()
assert set(res.severity_pct.unique())=={0,5,10,15} and set(res.seed.unique())=={42,43,44,45,46}

# Cross-experiment identity audit: previous phase-1 values must be identical on overlapping 25 specs.
old=pd.read_csv(BASE/'phase1_pilot_results.csv')
merge=res[(res.order=='imputation_outlier_smoothing_scaling') & res.severity_pct.gt(0)].merge(
 old,left_on=['domain','seed','severity_pct','imputation','outlier','smoothing'],
 right_on=['domain','seed','corruption_pct','imp','out','smooth'])
assert len(merge)==1875,(len(merge),'old 25 match')
max_q_diff=float((merge.q_score-merge.Q).abs().max())
max_rmse_diff=float((merge.nrmse-merge.NRMSE).abs().max())
assert max_q_diff<1e-10 and max_rmse_diff<1e-10,(max_q_diff,max_rmse_diff)
print('Identity audit:',len(merge),'pipeline records, differences',max_q_diff,max_rmse_diff)

# Summaries, excluding zero-percent cases from success-rate interpretation.
rows=[]
for (dom,sev),g in sm.groupby(['domain','severity_pct'],sort=True):
 b=bs[(bs.domain==dom)&(bs.severity_pct==sev)]
 rows.append(dict(domain=dom,severity_pct=int(sev),seeds=len(g),
                  median_spearman=float(g.spearman_q_nrmse.median()),
                  min_spearman=float(g.spearman_q_nrmse.min()),
                  max_spearman=float(g.spearman_q_nrmse.max()),
                  median_clean_q=float(b.clean_q.median()),
                  median_corrupted_q=float(b.corrupted_q.median()),
                  median_quality_max=float(g.quality_max.median()),
                  median_best_nrmse=float(g.best_nrmse.median()),
                  median_q_winner_nrmse=float(g.nrmse_best_q.median()),
                  median_regret_ratio=float(g.nrmse_regret_ratio.median()) if sev>0 else np.nan,
                  qmax_hits=int(g.qmax_ties_include_rmse_min.sum()),
                  top5_hits=int(g.top5_contains_optimum.sum()),
                  top10_hits=int(g.top10_contains_optimum.sum()),
                  median_invalid_pipelines=float(g.n_invalid_rmse.median())))
agg=pd.DataFrame(rows)
agg.to_csv(OUT/'domain_severity_summary.csv',index=False)

# Crossdomain summary for 5/10/15 only
ac=sm[sm.severity_pct>0].groupby('domain').agg(median_rho=('spearman_q_nrmse','median'),
 mean_rho=('spearman_q_nrmse','mean'),median_regret=('nrmse_regret_ratio','median'),
 wins=('qmax_ties_include_rmse_min','sum'),top5_hits=('top5_contains_optimum','sum'),
 n=('qmax_ties_include_rmse_min','size'), median_best_nrmse=('best_nrmse','median'),
 median_qwinner_nrmse=('nrmse_best_q','median')).reset_index()
ac.to_csv(OUT/'domain_overview.csv',index=False)

component_agg=comps.groupby(['domain','severity_pct','component'],as_index=False).agg(median_spearman=('spearman_component_nrmse','median'),valid_seeds=('spearman_component_nrmse','count'))
component_agg.to_csv(OUT/'dimension_correlations.csv',index=False)

# Counterexamples by q-selected-to-oracle gap, no 0%.
smv=sm[(sm.severity_pct>0)&np.isfinite(sm.nrmse_regret_ratio)].copy()
smv.sort_values('nrmse_regret_ratio',ascending=False).head(25).to_csv(OUT/'largest_misrankings.csv',index=False)
# Top25 by score and error for representative seed 42 at 10% for each domain.
for domain in ['industrial','energy','physiological','finance','network']:
 d=res[(res.domain==domain)&(res.seed==42)&(res.severity_pct==10)&(res.valid_rmse)]
 d.sort_values('q_score',ascending=False).head(25).to_csv(OUT/f'top25_score_{domain}_seed42_10pct.csv',index=False)
 d.sort_values('nrmse').head(25).to_csv(OUT/f'top25_nrmse_{domain}_seed42_10pct.csv',index=False)

# Q and NRMSE per 25 original specs, match with old as cross-validity proof.
merge[['domain','seed','severity_pct','imputation','outlier','smoothing','q_score','Q','nrmse','NRMSE']].to_csv(OUT/'phase1_reproduction_1875_rows.csv',index=False)

# Keep matrix of 336 configurations as a source-controlled shared candidate definition.

if not (OUT/'all_336_cleaning_pipelines.csv').exists():
    from itertools import product
    cfg=pd.DataFrame([{'imputation':im,'outlier':o,'smoothing':s,'scaling':'none','order':order}
        for order,im,o,s in product(('imputation_outlier_smoothing_scaling','outlier_imputation_smoothing_scaling'),
                  ('none','mean','median','locf','linear','knn','mice'),
                  ('none','zscore','iqr','mad','winsorize','isolation_forest'),
                  ('none','moving_median','savgol','butterworth'))])
    cfg.to_csv(OUT/'all_336_cleaning_pipelines.csv',index=False)

# Charts: using matplotlib defaults and separate figures.
order=['industrial','energy','physiological','finance','network']
D={'industrial':'Industrial','energy':'Energy','physiological':'Physiological','finance':'Finance','network':'Network'}
sub=agg[agg.severity_pct>0]
fig,ax=plt.subplots(figsize=(9,4.7))
for dom in order:
 g=sub[sub.domain==dom].sort_values('severity_pct')
 ax.plot(g.severity_pct,g.median_spearman,marker='o',label=D[dom])
ax.axhline(0,linestyle='--',linewidth=1)
ax.set_xticks([5,10,15]);ax.set_xlabel('Artificial corruption (%)');ax.set_ylabel('Spearman(Q, NRMSE)')
ax.set_title('Intrinsic quality index vs. reconstruction error')
ax.legend(ncol=3,fontsize=9);ax.grid(True,alpha=.2);fig.tight_layout();fig.savefig(OUT/'spearman_by_domain.png',dpi=180);plt.close(fig)

fig,ax=plt.subplots(figsize=(9,4.7))
for dom in order:
 g=sub[sub.domain==dom].sort_values('severity_pct')
 ax.plot(g.severity_pct,g.median_regret_ratio,marker='o',label=D[dom])
ax.axhline(1,linestyle='--',linewidth=1)
ax.set_xticks([5,10,15]);ax.set_xlabel('Artificial corruption (%)');ax.set_ylabel('Error ratio: Q-selected / NRMSE-optimal')
ax.set_title('Reconstruction cost of the highest intrinsic quality score')
ax.legend(ncol=3,fontsize=9);ax.grid(True,alpha=.2);fig.tight_layout();fig.savefig(OUT/'selection_regret_by_domain.png',dpi=180);plt.close(fig)

fig,ax=plt.subplots(figsize=(9,4.8))
for dom in order:
 g=agg[agg.domain==dom].sort_values('severity_pct')
 ax.plot(g.severity_pct,g.median_corrupted_q,marker='o',label=D[dom])
ax.set_xticks([0,5,10,15]);ax.set_xlabel('Artificial corruption (%)');ax.set_ylabel('Q of corrupted input data')
ax.set_title('Quality index response to increasing artificial corruption')
ax.legend(ncol=3,fontsize=9);ax.grid(True,alpha=.2);fig.tight_layout();fig.savefig(OUT/'dirty_q_by_severity.png',dpi=180);plt.close(fig)

# Environment and audit metadata.
metadata={
 'experiment':'exhaustive quality index evaluation of 5 non-eye-tracking domains',
 'date':'2026-11-05',
 'upstream_repository':'https://github.com/eyetracking-data/CIRRUS_Plus_Plus',
 'upstream_core_blob_sha_as_inspected':'8d39b51e933874c8d48d4f4d545290aada65172c',
 'implementation':'local faithful transcription; direct upstream import/parity check pending',
 'synthetic_data_source':'same cirrus_eval.py generate_domain_series(n=480,seed=42..46) as phase 1',
 'corruption':'same cirrus_eval.py corrupt_nested(clean,severity,seed+1000) as phase 1',
 'samples_per_series':480,'seeds':[42,43,44,45,46],'severities_pct':[0,5,10,15],
 'domains':order,'pipelines_per_scenario':336,'scaling':'none','candidate_count':33600,
 'model':'same feature-standardized RMSE vs noiseless synthetic clean signal; all points',
 'old_25_pipeline_reproduction_count':len(merge),
 'old_25_pipeline_reproduction_max_abs_Q_diff':max_q_diff,
 'old_25_pipeline_reproduction_max_abs_nrmse_diff':max_rmse_diff,
 'versions':{'python':sys.version.split()[0], 'numpy':np.__version__,'pandas':pd.__version__}
}
(OUT/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n')
for name in ['cirrus_eval.py','run_full_five_domains.py','test_upstream_parity.py','orchestrate.py','summarize_and_package.py']:
 if BASE/name != OUT/name:
  shutil.copy(BASE/name,OUT/name)
(OUT/'requirements.txt').write_text('numpy>=1.26,<3\npandas>=2.1,<3\nscipy>=1.11,<2\nscikit-learn>=1.4,<2\nmatplotlib>=3.7\nnbformat>=5.8\nnbclient>=0.8\n')

print('\nCross-domain overview:')
print(ac.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
print('\nBy severity:')
print(agg.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
print('\nTop counterexamples:\n',smv[['domain','seed','severity_pct','quality_max','nrmse_best_q','best_nrmse','nrmse_regret_ratio']].head(10).to_string(index=False))

# A compact Markdown README with computed summaries.
md=['# CIRRUS++ — Exhaustive synthetic benchmark of five additional domains','',
 '## Scope',
 'Uses the **same original 480-sample synthetic data generators** and exactly the **same seeded nested half-MCAR / half-spike corruption** as phase 1 (`cirrus_eval.py`). The five additional domains are industrial, energy, physiological, finance, and network. The first full eye-tracking evaluation was performed separately.',
 '',f'**Total:** {len(res):,} pipeline evaluations = 5 domains × 5 seeds × 4 severities × 336 cleaning configurations.',
 '', '## How to interpret results',
 '- Q = CIRRUS++ domain-weighted six-dimension intrinsic index (local transcription of the source implementation).',
 '- NRMSE = mean over signal columns of RMSE(clean, processed) / std(clean); evaluated over every time sample.',
 '- 336 = 7 imputation × 6 outlier × 4 smoothing × 2 ordering; scaling disabled to preserve RMSE units.',
 '- Corrupted percentage applies per signal; ~half missing at random and ~half additive spikes at 4–9× signal standard deviation.',
 '- 0% denotes the synthetic reference with its built-in background noise but no additional artificial corruption.',
 '- Some pipeline outputs contain NaN: their full-coverage NRMSE is explicitly missing and they are excluded from ranking.',
 '- Rankings and Spearman correlations count all configuration candidates, including output-equivalent combinations; outputs are **not independent observations**.',
 '- Strong monotonic correlation does not guarantee that the Q-maximizing pipeline minimizes NRMSE.',
 '', '## Domain-level outcome (5–15% artificial corruption, 15 scenarios/domain)',
 '', '| Domain | Median Spearman | Q-optimum hits / 15 | Top-5 hits / 15 | Median error factor of Q-winner |',
 '|:--|--:|--:|--:|--:|']
for r in ac.itertuples(): md.append(f'| {r.domain} | {r.median_rho:.3f} | {r.wins}/15 | {r.top5_hits}/15 | {r.median_regret:.2f}× |')
md.extend(['', '## Verification status',
 f'- **Phase-1 reproducibility PASS:** all {len(merge):,} overlapping previously tested configurations reproduced exactly: maximum absolute Q difference `{max_q_diff:g}`, maximum absolute NRMSE difference `{max_rmse_diff:g}`.',
 '- **Upstream parity pending:** Original `cirruspp_core.py` could not be downloaded into this execution environment; the functions are a source-matching local transcription. Do not claim validated parity against the upstream module without running `test_upstream_parity.py` on a local checkout of the repository.',
 '- The datasets are synthetically generated by simple models. They are not observed sensor records; physiological, energy and network series are illustrative extensions and not clinically/operationally validated models.',
 '- Only one composite corruption family, five seeds and 480 samples per series were evaluated. Strong domain-general conclusions would be premature.',
 '- Scores on output-scaled signals were not recomputed for every setting; this study tests the exhaustive **cleaning** search space only.',
 '', '## Reproduce',
 'From this directory with the requirements installed:',
 '```bash',
 'python run_full_five_domains.py --domain industrial --seed 42 --n 480 --output runs',
 'python orchestrate.py',
 'python summarize_and_package.py',
 '```',
 'The default `orchestrate.py` schedules all five domains and five seeds, each with 0/5/10/15% corruption.',
 'To verify upstream source parity after checking out the original GitHub repository:',
 '```bash',
 'python test_upstream_parity.py --module /path/to/CIRRUS_Plus_Plus/cirruspp_core.py',
 '```',
 '', '## Main output files',
 '- `results_all.csv`: all pipeline-level scores, six component scores, per-channel NRMSE/RMSE, and invalid flags.',
 '- `summary_all.csv`: scenario-level correlations, hit rates, winner errors and regret.',
 '- `domain_severity_summary.csv`: median statistics for each domain and severity.',
 '- `domain_overview.csv`: compact comparison at 5/10/15% severity.',
 '- `baselines_all.csv`: Q before corruption, after corruption, and six raw components.',
 '- `dimension_correlations.csv`: ability of each intrinsic score component to rank pipelines.',
 '- `phase1_reproduction_1875_rows.csv`: exact previous-pilot cross-check.',
 '- `figures`: PNG charts in the folder root.',
 '', '## Research next steps',
 '1. Run upstream parity test and pin a Git commit, plus dependency versions.',
 '2. Add other artifact classes (bursty missingness, drift, clipping, persistent offsets, realistic events).',
 '3. Increase seeds and compare by held-out generator regimes, including real datasets.',
 '4. Report event preservation and domain-specific loss rather than interpreting NRMSE as universal application quality.',
 '5. Evaluate transformed-data diagnostics separately from ground-truth reconstruction.',
 ''])
(OUT/'README.md').write_text('\n'.join(md),encoding='utf-8')
print('READY',OUT)
