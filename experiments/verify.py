"""Numerical checks of the invariant and lease certificate."""
import json,time
from pathlib import Path
import numpy as np
from core import PLANTS
ROOT=Path(__file__).resolve().parents[1]

def main():
 rng=np.random.default_rng(7301);results={}
 for name,p in PLANTS.items():
  cases=0;positive=0;largest=-1.;times=[]
  for _ in range(5000):
   x=float(rng.uniform(0,.999));v=float(rng.uniform(0,p.gain));u=float(rng.uniform(0,1))
   if p.recoverability(x+p.error,v+p.error*p.gain)>=1:continue
   begin=time.perf_counter_ns();L=p.lease(x,v,u);times.append((time.perf_counter_ns()-begin)/1e3)
   # True state is anywhere below the upper observation box.
   xx=x+float(rng.uniform(-p.error,p.error));xx=max(0.,xx);vv=max(0.,v+float(rng.uniform(-p.error*p.gain,p.error*p.gain)))
   positive+=int(L>0)
   hold=int(round((L+p.reserve)/p.dt)) if L>0 else 0
   for k in range(hold+300):
    cmd=u if k<hold else 0.
    xx,vv=p.step(xx,vv,cmd,float(rng.uniform(0,.025)))
    largest=max(largest,p.recoverability(xx,vv));assert xx<=1+1e-10;assert p.recoverability(xx,vv)<=1+1e-10
   cases+=1
  results[name]={'validated_cases':cases,'positive_lease_cases':positive,'max_recoverability':largest,'lease_compute_us_median':float(np.median(times)),'lease_compute_us_p95':float(np.quantile(times,.95))}
 # A reproducible delayed-pump counterexample for a state-only lease.
 p=PLANTS['water'];x0=.90;v0=.03;u=1.;L=p.lease(x0,v0,u,error=0.)
 naive=p.dt*np.floor(max(0.,(1-x0)/p.gain-p.reserve)/p.dt);rows=[]
 for label,budget in [('recoverability',L),('state_only',naive)]:
  x,v=x0,v0
  for k in range(160):
   rows.append({'method':label,'t':k*p.dt,'x':x,'v':v,'b':p.recoverability(x,v),'lease':budget})
   x,v=p.step(x,v,u if k*p.dt<budget+p.reserve-1e-10 else 0.,0.)
 import pandas as pd
 pd.DataFrame(rows).to_csv(ROOT/'results/counterexample.csv',index=False,float_format='%.12g')
 (ROOT/'results/certificate_checks.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
if __name__=='__main__':main()
