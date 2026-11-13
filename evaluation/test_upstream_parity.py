"""Optional parity test against upstream CIRRUS++.
Usage: python test_upstream_parity.py --module /path/to/CIRRUS_Plus_Plus/cirruspp_core.py
Does not download or modify any upstream code.
"""
import argparse,importlib.util,itertools
import numpy as np
import pandas as pd
from cirrus_eval import generate_domain_series,corrupt_nested,aggregate_profile,quality_components,DOMAIN_WEIGHTS
from run_full_five_domains import (original_equivalent_impute,stage_cache,q_profile,IMP,OUT,SMOOTH,ORDERS)
from cirrus_eval import apply_smoothing

p=argparse.ArgumentParser();p.add_argument('--module',required=True);args=p.parse_args()
spec=importlib.util.spec_from_file_location('cirruspp_core_original',args.module)
up=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=up
spec.loader.exec_module(up)
assert len(IMP)==7 and len(OUT)==6 and len(SMOOTH)==4
for domain in ('industrial','energy','physiological','finance','network'):
 clean,fs=generate_domain_series(domain,480,42)
 dirty=corrupt_nested(clean,.10,1042)
 local_q,_=q_profile(dirty,fs,domain)
 original_q=up.quality_index(up.aggregate_profile(dirty,fs),up.DOMAIN_CONFIG[domain]['quality_weights'])
 assert np.isclose(local_q,original_q,rtol=0,atol=1e-9),(domain,local_q,original_q)
 cached=stage_cache(dirty.drop(columns='timestamp'),fs)
 # Compare every pipeline, both stage orders, full transformed arrays and Q
 for order,im,out,smooth in itertools.product(ORDERS,IMP,OUT,SMOOTH):
  transformed=dirty.copy();transformed.loc[:,transformed.columns!='timestamp']=apply_smoothing(cached[order][(im,out)],smooth,fs)
  orig_out,meta=up.execute_pipeline(dirty,up.PipelineSpec(im,out,smooth,'none'),fs,order=order,seed=42)
  cols=[c for c in dirty.columns if c!='timestamp']
  assert np.allclose(transformed[cols].to_numpy(float),orig_out[cols].to_numpy(float),rtol=1e-11,atol=1e-9,equal_nan=True),(domain,order,im,out,smooth,'values')
  q0,_=q_profile(transformed,fs,domain)
  q1=up.quality_index(up.aggregate_profile(orig_out,fs),up.DOMAIN_CONFIG[domain]['quality_weights'])
  assert np.isclose(q0,q1,rtol=0,atol=1e-8),(domain,order,im,out,smooth,'score')
 print(f'PASS {domain}: 336 pipeline outputs + scores match upstream module.')
print('PARITY VERIFIED for 5 domains, seed42, 10% severity.')
