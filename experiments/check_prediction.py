"""Validate the vectorized scorer against the stepwise plant recurrence."""
import numpy as np
from core import PLANTS,action_scores,fast_action_scores
rng=np.random.default_rng(329)
maximum=0.
for p in PLANTS.values():
 for _ in range(2000):
  x=float(rng.uniform(0,1));v=float(rng.uniform(0,p.gain));q=float(rng.uniform(.2,.95));d=float(rng.uniform(0,.1))
  a=action_scores(p,x,v,q,d);b=fast_action_scores(p,x,v,q,d)
  maximum=max(maximum,float(np.max(np.abs(a-b))))
  np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-11)
  assert a.argmin()==b.argmin()
print('PASS: 4000 exact-recurrence comparisons; maximum absolute score difference',maximum)
