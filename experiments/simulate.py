"""Deterministic trace-driven closed-loop scheduling experiments."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ.setdefault(k,'1')
import json,math,argparse,time,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import numpy as np
import pandas as pd
from core import PLANTS,candidate,scenario
ROOT=Path(__file__).resolve().parents[1]

def run_episode(plant_name,seed,policy,latencies,scale=1.,seconds=180.,save_trace=False,noise=True):
    p=PLANTS[plant_name];sc=scenario(seed,p,seconds);x,v,u=sc['x'],sc['v'],sc['u'];dt=p.dt
    rng=np.random.default_rng(seed+987654);idx=rng.integers(0,len(latencies),int(seconds/dt)+1)
    sensor=np.random.default_rng(seed+54321).uniform(-1,1,(int(seconds/dt)+1,2)) if noise else np.zeros((int(seconds/dt)+1,2))
    n=len(sc['target']);q90=float(np.quantile(latencies,.9))*scale
    active=None;epoch=0;jobs=0;accepted=0;timeouts=0;stale=0;rejected=0;switches=0;admissions=0
    spent=0.;wasted=0.;age_sum=0.;fallback=False;fallback_n=0;unsafe=0;maxb=0.;loss=0.;success=0;tr=[]
    def settle(job,now,waste=False):
        nonlocal spent,wasted
        elapsed=(now-job['start'])*dt;spent+=elapsed
        if waste:wasted+=elapsed
    for k in range(n):
        t=k*dt;target=float(sc['target'][k]);draw=float(sc['draw'][k]);ox=x+p.error*sensor[k,0];ov=max(0.,v+p.error*p.gain*sensor[k,1])
        changed=False
        if k and target!=sc['target'][k-1]:epoch+=1;changed=True
        if u>0 and not p.monitor(ox,ov,u):
            u=0.;epoch+=1;switches+=1;fallback=True;changed=True
        # Every policy rejects obsolete epochs at commit; proactive cancellation
        # is shared by all but the monitor-only diagnostic and no-cancel ablation.
        if active and active['epoch']!=epoch and policy not in ('monitor','lease_noepoch'):
            settle(active,k,True);active=None;stale+=1
        if active and k>=active['deadline'] and k<active['end']:
            settle(active,k,True);active=None;timeouts+=1
            if (policy in ('lease','lease_noepoch','lease_zeroreserve','reactive_handoff') or policy.startswith('fixedh')) and u>0:
                u=0.;epoch+=1;fallback=True;admissions+=1
        if active and k>=active['end']:
            age=(k-active['start'])*dt;bad=active['epoch']!=epoch
            allowed=(not bad) and p.admissible(ox,ov,active['action'])
            settle(active,k,not allowed)
            if allowed:u=active['action'];accepted+=1;age_sum+=age;fallback=False
            elif bad:stale+=1
            else:rejected+=1
            active=None
        if active is None:
            lease=p.lease(ox,ov,u)
            if policy in ('lease','lease_noepoch','lease_zeroreserve') and lease<dt and u>0:
                u=0.;epoch+=1;admissions+=1;fallback=True;lease=p.lease(ox,ov,u)
            if policy.startswith('fixedh'):
                budget=float(policy[6:])
            elif policy.startswith('fixed'):
                budget=float(policy[5:])
            elif policy=='quantile':budget=q90
            elif policy in ('lease','lease_noadmit','lease_noepoch'):budget=lease
            elif policy=='lease_zeroreserve':budget=p.lease(ox,ov,u,reserve=0.)
            elif policy=='margin':budget=max(0.,min(18.,(1-ox-p.error)/max(ov,p.gain*u,1e-5)-p.reserve))
            else:budget=18.
            if policy=='instant':duration=dt
            else:duration=float(latencies[idx[jobs]])*scale
            end=k+max(1,int(math.ceil(duration/dt-1e-10)))
            deadline=k+max(1,int(math.floor(budget/dt+1e-10)))
            action=candidate(p,ox,ov,target,draw)
            active={'start':k,'end':end,'deadline':deadline,'epoch':epoch,'action':action,'budget':budget};jobs+=1
        if save_trace:tr.append([t,x,v,u,target,float(p.recoverability(x,v)),active['budget'],float(fallback)])
        loss+=(x-target)**2;success+=abs(x-target)<=.08;unsafe+=x>1.+1e-10;maxb=max(maxb,p.recoverability(x,v));fallback_n+=fallback
        x,v=p.step(x,v,u,draw)
    unsafe+=x>1.+1e-10;maxb=max(maxb,p.recoverability(x,v))
    if active:settle(active,n,False)
    record={'plant':plant_name,'seed':seed,'policy':policy,'scale':scale,'seconds':seconds,'rmse':float(np.sqrt(loss/n)),'tracking_fraction':success/n,'violation_steps':unsafe,'episode_violation':int(unsafe>0),'max_recoverability':maxb,'jobs':jobs,'accepted':accepted,'accepted_per_min':accepted/(seconds/60),'timeout':timeouts,'stale':stale,'rejected':rejected,'monitor_switches':switches,'admission_switches':admissions,'fallback_fraction':fallback_n/n,'inference_seconds':spent,'discarded_seconds':wasted,'discarded_fraction':wasted/max(spent,1e-9),'mean_applied_age':age_sum/max(accepted,1)}
    return record,tr

def worker(a):
    model,plant,seed,policy,lat,scale,seconds=a
    r,_=run_episode(plant,seed,policy,lat,scale,seconds);r['model']=model;return r

def read_latencies(root=ROOT):
    result={}
    for f in sorted((root/'results').glob('*_profile.jsonl')):
        records=[json.loads(s) for s in f.read_text().splitlines()]
        result[f.name.replace('_profile.jsonl','')]=[r['seconds'] for r in records]
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['pilot','tune','test'],default='pilot');ap.add_argument('--workers',type=int,default=10);args=ap.parse_args()
    latencies=read_latencies();assert latencies,'No measured latency profiles'
    if args.phase=='pilot':seeds=range(9000,9012);policies=['fixed2','fixed4','fixed8','reactive','monitor','quantile','lease','lease_noadmit','margin'];scales=[1.,2.,4.];seconds=120.
    elif args.phase=='tune':seeds=range(12000,12024);policies=['fixed1','fixed2','fixed4','fixed8','fixed12','fixed18','fixedh1','fixedh2','fixedh4','fixedh8','fixedh12','fixedh18'];scales=[1.,2.,4.];seconds=180.
    else:seeds=range(20000,20100);policies=['fixed_best','quantile','reactive','reactive_handoff','monitor','lease','lease_noadmit','lease_noepoch','margin','instant'];scales=[1.,2.,4.];seconds=180.
    tuned={}
    if args.phase=='test':
        tf=pd.read_csv(ROOT/'results/tune.csv');means=tf.groupby(['model','plant','scale','policy']).rmse.mean();
        for k,g in means.groupby(level=[0,1,2]):tuned[k]=g.idxmin()[3]
    jobs=[]
    for model,lat in latencies.items():
        for plant in PLANTS:
            for scale in scales:
                for seed in seeds:
                    for pol in policies:
                        p=tuned[(model,plant,scale)] if pol=='fixed_best' else pol
                        jobs.append((model,plant,seed,p,lat,scale,seconds))
    started=time.time();results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,r in enumerate(pool.map(worker,jobs,chunksize=4)):
            results.append(r)
            if (i+1)%100==0:print(args.phase,i+1,len(jobs),round(time.time()-started,1),flush=True)
    df=pd.DataFrame(results);df.to_csv(ROOT/f'results/{args.phase}.csv',index=False,float_format='%.12g')
    (ROOT/f'results/{args.phase}_protocol.json').write_text(json.dumps({'phase':args.phase,'seeds':list(seeds),'policies':policies,'scales':scales,'seconds':seconds,'jobs':len(jobs),'workers':args.workers,'duration_seconds':time.time()-started,'latency_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'results').glob('*_profile.jsonl')}},indent=2))
    print(df.groupby(['plant','scale','policy'])[['rmse','tracking_fraction','discarded_fraction','episode_violation']].mean().to_string(),flush=True)
if __name__=='__main__':main()
