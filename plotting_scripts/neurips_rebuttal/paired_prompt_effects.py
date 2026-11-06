"""Compute four-step minus standard effects on shared valid systems.

Condition means use all available records. Paired differences use only matched
sample IDs; counts are reported separately to make missingness explicit.
Intervals are exploratory, unadjusted percentile bootstrap intervals from
4,000 shared-system resamples (seed 0).
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np

REPO=Path(__file__).resolve().parents[2]
import argparse
parser=argparse.ArgumentParser(description='Paired bootstrap changes in accuracy and exploration under four-step prompting.')
parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'step_instruction_judge_analysis')
args=parser.parse_args()
OUT=args.output_dir;OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(REPO))
from plotting_scripts.neurips.archived_experiment_data import discover_experiments,select_preferred_experiments,gather_occupancy_entropy
from plotting_scripts.neurips_rebuttal.mechanism_info_accuracy_comparison import gather_step5_accuracy
from plotting_scripts.neurips_rebuttal.step_instruction_reasoning_accuracy_correlation import STEP_RUNS

def ci(values):
 values=np.asarray(values,dtype=float);rng=np.random.default_rng(0)
 means=values[rng.integers(0,len(values),size=(4000,len(values)))].mean(axis=1)
 return values.mean(),*np.percentile(means,[2.5,97.5])

exps=discover_experiments(str(REPO/'results'))
standard={e.label:e for e in select_preferred_experiments(exps,[f'{m} with Mechanism Info' for m,_ in STEP_RUNS])}
by_source={e.source_tar:e for e in exps}
rows=[]
try:
 for model,step_archive in STEP_RUNS:
  a=standard[f'{model} with Mechanism Info'];b=by_source[step_archive]
  sources={'Accuracy':(gather_step5_accuracy(a).rename(columns={'accuracy':'value'}),gather_step5_accuracy(b).rename(columns={'accuracy':'value'}))}
  aa=pd.DataFrame(gather_occupancy_entropy(a));bb=pd.DataFrame(gather_occupancy_entropy(b))
  for metric in ['Occupancy Entropy','S-cat Corner Mass']:
   sources[metric]=(aa[(aa.step==5)&(aa.metric==metric)],bb[(bb.step==5)&(bb.metric==metric)])
  for metric,(one,two) in sources.items():
   pairs=one[['sample_id','value']].merge(two[['sample_id','value']],on='sample_id',suffixes=('_standard','_step'),validate='one_to_one')
   mean,lo,hi=ci(pairs.value_step-pairs.value_standard)
   rows.append(dict(model=model,metric=metric,n_standard=len(one),n_step=len(two),n_paired=len(pairs),standard_mean=one.value.mean(),step_mean=two.value.mean(),paired_change=mean,change_ci_low=lo,change_ci_high=hi))
finally:
 for e in exps:e.close()
df=pd.DataFrame(rows);df.to_csv(OUT/'nine_model_paired_prompt_effects.csv',index=False)
print(df.to_string(index=False),flush=True)
