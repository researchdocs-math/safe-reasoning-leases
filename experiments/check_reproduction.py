"""Compare replay outputs against the archived numerical reference results."""
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SPECS={'tune.csv':['model','plant','scale','seed','policy'],'test.csv':['model','plant','scale','seed','policy'],'aggregate_summary.csv':['policy','metric'],'paired_contrasts.csv':['reference','metric'],'cell_contrasts.csv':['model','plant','scale','metric'],'selected_fixed.csv':['model','plant','scale']}

def main():
 for name,keys in SPECS.items():
  expected=pd.read_csv(ROOT/'expected'/name).sort_values(keys).reset_index(drop=True)
  actual=pd.read_csv(ROOT/'results'/name).sort_values(keys).reset_index(drop=True)
  assert actual.shape==expected.shape,(name,actual.shape,expected.shape)
  assert list(actual.columns)==list(expected.columns),name
  for c in expected:
   if pd.api.types.is_numeric_dtype(expected[c]):np.testing.assert_allclose(actual[c],expected[c],rtol=1e-9,atol=1e-10,err_msg=f'{name}:{c}')
   else:assert actual[c].equals(expected[c]),(name,c)
  print('PASS',name,len(actual),'rows')
 print('All deterministic experiment outputs match the reference results.')
if __name__=='__main__':main()
