import concurrent.futures,subprocess,sys,time,os
from pathlib import Path
work=Path(__file__).resolve().parent
jobs=[(d,s) for d in ('industrial','energy','physiological','finance','network') for s in range(42,47)]
def run(job):
 d,s=job
 env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
 p=subprocess.run([sys.executable,str(work/'run_full_five_domains.py'),'--domain',d,'--seed',str(s),'--n','480','--output',str(work/'runs')],env=env,capture_output=True,text=True)
 (work/'logs').mkdir(exist_ok=True)
 (work/'logs'/f'{d}_{s}.log').write_text(p.stdout+'\n'+p.stderr)
 return d,s,p.returncode,p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[:200]
t0=time.monotonic(); ok=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
 for i,f in enumerate(concurrent.futures.as_completed([ex.submit(run,j) for j in jobs]),1):
  d,s,rc,info=f.result();print(f'COMPLETE {i}/{len(jobs)} {d} seed={s} rc={rc}; time={time.monotonic()-t0:.1f}s; {info}',flush=True)
  ok.append((d,s,rc))
print('END',time.monotonic()-t0,'failed',[(d,s) for d,s,rc in ok if rc],flush=True)
