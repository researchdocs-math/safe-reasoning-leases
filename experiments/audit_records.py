"""Check recorded experiment integrity and independently audit live trajectories."""
from pathlib import Path
import hashlib,json
import numpy as np
from core import PLANTS,scenario
ROOT=Path(__file__).resolve().parents[1]

def main():
 protocol=json.loads((ROOT/'configs/protocol.json').read_text())
 for name,expected in protocol['sha256'].items():
  actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
  assert actual==expected,('Frozen input changed',name)
 profiles=[]
 for p in (ROOT/'results').glob('*_profile.jsonl'):
  records=[json.loads(s) for s in p.read_text().splitlines()]
  assert len(records)==120 and len({r['id'] for r in records})==120
  assert all(r['seconds']>0 for r in records)
  profiles+=records
 assert len(profiles)==240
 live=[json.loads(s) for s in (ROOT/'results/live.jsonl').read_text().splitlines()]
 assert len(live)==60 and len({(r['plant'],r['seed'],r['policy']) for r in live})==60
 max_x=0.;max_b=0.;transition_error=0.
 for r in live:
  p=PLANTS[r['plant']];sc=scenario(r['seed'],p,60);a=np.asarray(r['trace'])
  assert len(a)==600
  np.testing.assert_allclose(a[:,4],sc['target'],rtol=0,atol=1e-12)
  for k,row in enumerate(a):
   x,v=p.step(row[1],row[2],row[3],sc['draw'][k])
   if k+1<len(a):
    err=max(abs(x-a[k+1,1]),abs(v-a[k+1,2]));transition_error=max(transition_error,err)
    assert err<1e-12
   max_x=max(max_x,float(row[1]),x);max_b=max(max_b,float(row[5]),p.recoverability(x,v))
  np.testing.assert_allclose(r['rmse'],np.sqrt(np.mean((a[:,1]-a[:,4])**2)),rtol=0,atol=1e-12)
  assert r['violation_steps']==int(np.sum(a[:,1]>1+1e-10))
 assert max_x<=1+1e-10
 report={'profile_jobs':len(profiles),'live_episodes':len(live),'live_transitions_checked':len(live)*600,'max_transition_error':transition_error,'max_x_including_terminal':max_x,'max_recoverability_including_terminal':max_b,'frozen_input_hashes_checked':len(protocol['sha256'])}
 (ROOT/'results/record_audit.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
