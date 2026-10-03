"""Local GPU agent interacting with a wall-clock-paced simulated plant."""
import json,time,threading,argparse
from pathlib import Path
import numpy as np
from core import PLANTS,scenario
from inference import Agent,state_dict
ROOT=Path(__file__).resolve().parents[1]

class Runtime:
 def __init__(self,name,seed,seconds=60.,speed=3.):
  self.p=PLANTS[name];self.sc=scenario(seed,self.p,seconds);self.x=self.sc['x'];self.v=self.sc['v'];self.u=self.sc['u'];self.n=len(self.sc['target']);self.k=0;self.epoch=0;self.speed=speed;self.lock=threading.RLock();self.rows=[];self.jitter=[];self.switches=0;self.fallback=False;self.done=False
  self.noise=np.random.default_rng(seed+54321).uniform(-1,1,(self.n+1,2));self.start=time.perf_counter();self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
 def obs(self):
  i=min(self.k,self.n-1);p=self.p
  return self.x+p.error*self.noise[i,0],max(0.,self.v+p.error*p.gain*self.noise[i,1])
 def loop(self):
  while not self.done:
   due=self.start+(self.k+1)*self.p.dt/self.speed
   delay=due-time.perf_counter()
   if delay>0:time.sleep(min(delay,.005));continue
   with self.lock:
    if self.k>=self.n:self.done=True;break
    k=self.k;p=self.p;target=float(self.sc['target'][k]);draw=float(self.sc['draw'][k]);ox,ov=self.obs();self.jitter.append(max(0.,time.perf_counter()-due))
    if self.u>0 and not p.monitor(ox,ov,self.u):self.u=0.;self.epoch+=1;self.switches+=1;self.fallback=True
    self.rows.append([k*p.dt,self.x,self.v,self.u,target,p.recoverability(self.x,self.v),float(self.fallback)])
    self.x,self.v=p.step(self.x,self.v,self.u,draw);self.k+=1
    if self.k<self.n and self.sc['target'][self.k]!=self.sc['target'][self.k-1]:self.epoch+=1
 def state(self):
  with self.lock:
   x,v=self.obs();return state_dict(self.p,x,v,float(self.sc['draw'][min(self.k,self.n-1)]))
 def goal(self):
  with self.lock:return {'target':round(float(self.sc['target'][min(self.k,self.n-1)]),4)}
 def prepare(self,policy,fixed):
  with self.lock:
   ox,ov=self.obs();L=self.p.lease(ox,ov,self.u)
   if policy=='lease' and L<self.p.dt and self.u>0:self.u=0.;self.epoch+=1;self.fallback=True;L=self.p.lease(ox,ov,0.)
   B=L if policy=='lease' else (fixed if policy=='fixed' else 18.)
   return self.epoch,B/self.speed
 def cancel(self,epoch):
  with self.lock:return self.done or self.epoch!=epoch
 def finish(self,record,epoch,policy):
  with self.lock:
   if self.done:return 'censored'
   if self.epoch!=epoch:return 'obsolete'
   if record['status']=='timeout' and policy=='lease':self.u=0.;self.epoch+=1;self.fallback=True;return 'timeout'
   if record['action'] is None:return record['status']
   x,v=self.obs()
   if not self.p.admissible(x,v,record['action']):return 'rejected'
   self.u=record['action'];self.fallback=False;return 'applied'

def episode(agent,name,seed,policy,fixed,speed):
 rt=Runtime(name,seed,speed=speed);jobs=[]
 while not rt.done:
  epoch,budget=rt.prepare(policy,fixed);before=rt.k;job_wall_start=time.perf_counter()
  rec=agent.run(rt.p,rt.state,rt.goal,deadline=budget,cancel=lambda:rt.cancel(epoch));rec['result']=rt.finish(rec,epoch,policy);rec['start_step']=before;rec['end_step']=rt.k;rec['budget_wall_seconds']=budget;rec['occupied_seconds']=max(0.,min(time.perf_counter(),rt.start+60./speed)-job_wall_start);jobs.append(rec)
 rt.thread.join();a=np.array(rt.rows);err=a[:,1]-a[:,4];spent=sum(r['occupied_seconds'] for r in jobs);discarded=sum(r['occupied_seconds'] for r in jobs if r['result'] not in ('applied','censored'))
 return {'model':Path(agent.path).name,'plant':name,'seed':seed,'policy':policy,'speed':speed,'fixed_sim_seconds':fixed,'rmse':float(np.sqrt(np.mean(err**2))),'tracking_fraction':float(np.mean(abs(err)<=.08)),'violation_steps':int(np.sum(a[:,1]>1+1e-10)),'max_recoverability':float(a[:,5].max()),'applied':sum(r['result']=='applied' for r in jobs),'jobs_count':len(jobs),'discarded_fraction':discarded/max(spent,1e-9),'inference_seconds':spent,'fallback_fraction':float(a[:,6].mean()),'monitor_switches':rt.switches,'monitor_jitter_p99_ms':float(np.quantile(rt.jitter,.99)*1000),'wall_seconds':time.perf_counter()-rt.start,'trace':rt.rows,'jobs':jobs}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',default='Qwen3-4B-4bit');ap.add_argument('--seeds',type=int,default=10);ap.add_argument('--speed',type=float,default=3.);ap.add_argument('--pilot',action='store_true');args=ap.parse_args()
 agent=Agent(ROOT/'models'/args.model);p=PLANTS['water'];agent.run(p,state_dict(p,.4,.01,.01),{'target':.65})
 profile=[json.loads(l) for l in (ROOT/f'results/{args.model}_profile.jsonl').read_text().splitlines()];fast=[r['seconds'] for r in profile if r['history_length']==0]
 # Fixed budget is calibrated from independent short-context profiling records.
 fixed=float(np.quantile(fast,.95))*args.speed
 path=ROOT/('results/live_pilot.jsonl' if args.pilot else 'results/live.jsonl');done=set()
 if path.exists():done={(r['plant'],r['seed'],r['policy']) for r in map(json.loads,path.read_text().splitlines())}
 rng=np.random.default_rng(7711)
 for seed in range(31000,31000+args.seeds):
  for name in ['water','thermal']:
   policies=list(rng.permutation(['fixed','reactive','lease']))
   for policy in policies:
    if (name,seed,policy) in done:continue
    r=episode(agent,name,seed,policy,fixed,args.speed)
    with path.open('a') as f:f.write(json.dumps(r)+'\n')
    print(name,seed,policy,'rmse',round(r['rmse'],4),'applied',r['applied'],'jobs',r['jobs_count'],'jitter99ms',round(r['monitor_jitter_p99_ms'],2),flush=True)
if __name__=='__main__':main()
