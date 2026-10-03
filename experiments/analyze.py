"""Seed-level summaries and paired bootstrap contrasts for the experiments."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
METRICS=['rmse','tracking_fraction','discarded_fraction','accepted_per_min','fallback_fraction']

def ci(a,seed=1741):
 a=np.asarray(a,dtype=float);rng=np.random.default_rng(seed);means=a[rng.integers(0,len(a),(10000,len(a)))].mean(axis=1);lo,hi=np.quantile(means,[.025,.975]);return float(a.mean()),float(lo),float(hi)

def main():
 d=pd.read_csv(ROOT/'results/test.csv');d.loc[d.policy.str.startswith('fixed'),'policy']='fixed';out={}
 out['test_episodes']=len(d);out['violating_episodes']=int(d.episode_violation.sum());out['max_recoverability']=float(d.max_recoverability.max());out['per_cell_zero_failure_upper95']=float(1-.05**(1/100))
 aggregate=[]
 for p,g in d.groupby('policy'):
  z=g.groupby('seed')[METRICS].mean()
  for m in METRICS:
   mean,lo,hi=ci(z[m]);aggregate.append({'policy':p,'metric':m,'mean':mean,'lo95':lo,'hi95':hi,'seeds':len(z)})
 a=pd.DataFrame(aggregate);a.to_csv(ROOT/'results/aggregate_summary.csv',index=False,float_format='%.12g')
 contrasts=[]
 for reference in sorted(set(d.policy)-{'lease'}):
  for metric in METRICS:
   z=d.groupby(['seed','policy'])[metric].mean().unstack();mean,lo,hi=ci(z.lease-z[reference]);contrasts.append({'reference':reference,'metric':metric,'difference_lease_minus_reference':mean,'lo95':lo,'hi95':hi})
 pd.DataFrame(contrasts).to_csv(ROOT/'results/paired_contrasts.csv',index=False,float_format='%.12g')
 cells=[]
 for (model,plant,scale),g in d.groupby(['model','plant','scale']):
  for m in METRICS:
   z=g.pivot(index='seed',columns='policy',values=m);mean,lo,hi=ci(z.lease-z.reactive)
   cells.append({'model':model,'plant':plant,'scale':scale,'metric':m,'difference':mean,'lo95':lo,'hi95':hi})
 pd.DataFrame(cells).to_csv(ROOT/'results/cell_contrasts.csv',index=False,float_format='%.12g')
 d.groupby(['model','plant','scale','policy'])[METRICS+['episode_violation','timeout','stale','rejected']].mean().reset_index().to_csv(ROOT/'results/cell_summary.csv',index=False,float_format='%.12g')
 tune=pd.read_csv(ROOT/'results/tune.csv');best=tune.groupby(['model','plant','scale','policy']).rmse.mean().reset_index();best=best.loc[best.groupby(['model','plant','scale']).rmse.idxmin()];best.to_csv(ROOT/'results/selected_fixed.csv',index=False,float_format='%.12g')
 profiles=[];latency=[]
 for p in sorted((ROOT/'results').glob('*_profile.jsonl')):
  records=[json.loads(l) for l in p.read_text().splitlines()];profiles+=records
  for h in [None,0,16,64]:
   rs=[r for r in records if h is None or r['history_length']==h];times=[r['seconds'] for r in rs]
   latency.append({'model':rs[0]['model'],'history_length':'all' if h is None else h,'n':len(rs),'median_seconds':float(np.median(times)),'p95_seconds':float(np.quantile(times,.95)),'valid_fraction':float(np.mean([r['status']=='ok' for r in rs])),'agreement_fraction':float(np.mean([r['action'] is not None and abs(r['action']-r['oracle_action'])<1e-8 for r in rs])),'peak_memory_gib':max(r['peak_memory_bytes'] for r in rs)/2**30})
 pd.DataFrame(latency).to_csv(ROOT/'results/latency_summary.csv',index=False,float_format='%.12g');out['profile_jobs']=len(profiles)
 if (ROOT/'results/live.jsonl').exists():
  live=[json.loads(l) for l in (ROOT/'results/live.jsonl').read_text().splitlines()];ld=pd.DataFrame([{k:v for k,v in r.items() if k not in ('trace','jobs')} for r in live]);ld.to_csv(ROOT/'results/live_metrics.csv',index=False,float_format='%.12g')
  summary=[]
  for (plant,policy),g in ld.groupby(['plant','policy']):
   for m in ['rmse','tracking_fraction','discarded_fraction','applied','monitor_jitter_p99_ms']:
    mean,lo,hi=ci(g[m]);summary.append({'plant':plant,'policy':policy,'metric':m,'mean':mean,'lo95':lo,'hi95':hi,'seeds':len(g)})
  pd.DataFrame(summary).to_csv(ROOT/'results/live_summary.csv',index=False,float_format='%.12g')
  out['live_episodes']=len(live);out['live_violation_steps']=int(ld.violation_steps.sum());out['live_jitter_p99_range_ms']=[float(ld.monitor_jitter_p99_ms.min()),float(ld.monitor_jitter_p99_ms.max())];out['live_max_recoverability']=float(ld.max_recoverability.max())
  out['live_job_outcomes']=pd.Series([j['result'] for r in live for j in r['jobs']]).value_counts().to_dict()
 (ROOT/'results/summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));print(a.to_string(index=False))
if __name__=='__main__':main()
