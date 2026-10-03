"""Positive-system benchmark and recoverability leases (SI seconds)."""
from dataclasses import dataclass
from functools import cached_property, lru_cache
import numpy as np

@dataclass(frozen=True)
class Plant:
    name: str
    dt: float = 0.1
    gain: float = 0.035
    tau: float = 2.0
    cooling: float = 0.0
    limit: float = 1.0
    error: float = 0.004
    reserve: float = 0.5
    @cached_property
    def rho(self): return float(np.exp(-self.dt/self.tau))
    @cached_property
    def c(self): return self.dt/(1-self.rho)
    def recoverability(self,x,v): return x+self.c*v
    def step(self,x,v,u,draw=0.0):
        # The nonnegative boundary models exhausted storage; only upper-limit
        # protection is asserted. The same discretization defines the simulator.
        return max(0.,(1-self.cooling*self.dt)*x+self.dt*(v-draw)), self.rho*v+(1-self.rho)*self.gain*u
    def lease(self,x,v,u,cap=18.0,error=None,reserve=None):
        # Upper initial box and zero withdrawal dominate all allowed trajectories.
        e=self.error if error is None else error
        r=self.reserve if reserve is None else reserve
        xx=x+e; vv=v+e*self.gain
        safe=0
        for k in range(int((cap+r)/self.dt)+1):
            if self.recoverability(xx,vv)>self.limit-1e-12: break
            safe=k
            xx,vv=self.step(xx,vv,u,0.)
        return max(0.,min(cap,safe*self.dt-r))
    def admissible(self,x,v,u):
        return self.lease(x,v,u,cap=self.reserve+0.3)>=self.dt
    def monitor(self,x,v,u):
        # Check the currently held command throughout the entire reaction reserve.
        return self.lease(x,v,u,cap=self.reserve+0.3)>0.0

PLANTS={'water':Plant('water'), 'thermal':Plant('thermal',gain=0.045,tau=3.5,cooling=0.035)}
ACTIONS=np.array([0.,.25,.5,.75,1.])

def action_scores(plant,x,v,target,draw,horizon=8.):
    xx=np.full(5,x);vv=np.full(5,v);loss=np.zeros(5)
    rho=plant.rho;dt=plant.dt
    for _ in range(int(horizon/dt)):
        xx=np.maximum(0.,(1-plant.cooling*dt)*xx+dt*(vv-draw))
        vv=rho*vv+(1-rho)*plant.gain*ACTIONS
        loss+=(xx-target)**2+.002*ACTIONS**2
    return loss

@lru_cache(maxsize=16)
def prediction_basis(plant,horizon):
    j=np.arange(1,int(horizon/plant.dt)+1,dtype=float)
    a=1-plant.cooling*plant.dt
    aj=a**j
    sj=j if a==1 else (1-aj)/(1-a)
    rj=(aj-plant.rho**j)/(a-plant.rho)
    return aj[:,None],sj[:,None],rj[:,None]

def fast_action_scores(plant,x,v,target,draw,horizon=8.):
    """Exact reflected linear recurrence, vectorized over time and actions."""
    aj,sj,rj=prediction_basis(plant,horizon)
    y=aj*x+plant.dt*(rj*v+plant.gain*(sj-rj)*ACTIONS-sj*draw)
    # Running reflection implements max(0, ...) at every intermediate step.
    xx=y+aj*np.maximum.accumulate(np.maximum(0.,-y/aj),axis=0)
    return np.sum((xx-target)**2+.002*ACTIONS**2,axis=0)

def candidate(plant,x,v,target,draw,horizon=8.):
    """Finite-action planner used only for scheduler isolation."""
    return float(ACTIONS[int(np.argmin(fast_action_scores(plant,x,v,target,draw,horizon)))])

def scenario(seed,plant,seconds=180):
    rng=np.random.default_rng(seed)
    n=int(seconds/plant.dt)
    # Independent exogenous seed streams are shared across every policy.
    target=np.empty(n);draw=np.empty(n)
    for st in range(0,n,250):
        target[st:st+250]=rng.uniform(.45,.89)
        base=rng.uniform(.003,.023) if plant.name=='water' else rng.uniform(.0,.008)
        draw[st:st+250]=base
    draw=np.maximum(0.,draw+rng.uniform(-.001,.001,n))
    x=float(rng.uniform(.25,.70));v=float(rng.uniform(0.,plant.gain*.65))
    return {'x':x,'v':v,'u':float(rng.choice(ACTIONS)),'target':target,'draw':draw}
