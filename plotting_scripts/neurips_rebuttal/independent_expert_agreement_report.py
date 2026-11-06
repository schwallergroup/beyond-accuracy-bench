"""Reproduce the two-expert agreement table and the consensus-extreme audit.

Expert identities are assigned by the two documented input files, not their
legacy anonymous-ID column, which is identical in both saved exports.
"""
from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from plotting_scripts.neurips.judge_rubric_scores_by_model import RUBRIC_LABELS, _strip_mechanism_suffix
from plotting_scripts.neurips_rebuttal.judge_filtering import canonical_model_id
from plotting_scripts.neurips_rebuttal.iteration_reasoning_score_analysis import JUDGE_INPUTS,spearman_with_bootstrap_ci

KEYS=['sample_id','iteration','model_name','rubric_key']

PROJECT_ROOT = ROOT
HUMAN_INPUT = ROOT / "bench/judge/human_runs/human_judge_results_f6e6ad1c5a1f.csv"


def load_human_and_debiased_llm_scores() -> pd.DataFrame:
    """Return paired human scores and self-judge-excluded LLM means."""
    human = pd.read_csv(HUMAN_INPUT)
    human["score"] = pd.to_numeric(human["score"], errors="raise")
    human["model_name"] = human["model_label"].map(_strip_mechanism_suffix)
    key_columns = ["sample_id", "iteration", "model_name", "rubric_key"]
    human = human[key_columns + ["score"]].rename(columns={"score": "human_score"})

    llm_frames = []
    for judge_id, path in JUDGE_INPUTS.items():
        frame = pd.read_csv(path)
        observed = {canonical_model_id(value) for value in frame["judge_model"].dropna().astype(str)}
        if observed != {judge_id}:
            raise RuntimeError(f"Expected judge {judge_id!r} in {path}, found {sorted(observed)}")
        frame["score"] = pd.to_numeric(frame["score"], errors="raise")
        frame["model_name"] = frame["model_label"].map(_strip_mechanism_suffix)
        frame["judge_id"] = judge_id
        llm_frames.append(frame[key_columns + ["score", "judge_id"]])
    llm = pd.concat(llm_frames, ignore_index=True)

    # Restrict before averaging so every comparison has the exact human trace,
    # model, iteration, and rubric key.  Exclude GPT-5.4 for GPT-5.4 targets
    # and Claude Opus 4.7 for Claude Opus 4.7 targets.
    shared_keys = human[key_columns].drop_duplicates()
    llm = llm.merge(shared_keys, on=key_columns, how="inner")
    target_ids = llm["model_name"].map(canonical_model_id)
    llm = llm[llm["judge_id"] != target_ids]
    llm = (
        llm.groupby(key_columns, as_index=False)
        .agg(llm_score=("score", "mean"), n_eligible_judges=("judge_id", "nunique"))
    )
    paired = human.merge(llm, on=key_columns, how="inner", validate="one_to_one")
    if len(paired) != len(human):
        raise RuntimeError(f"Only {len(paired)} of {len(human)} human judgments matched LLM results.")
    return paired


def build_tables():
    paired=load_human_and_debiased_llm_scores()
    second=pd.read_csv(ROOT/'bench/judge/human_runs/human_judge_results_2.csv')
    second['model_name']=second.model_label.str.replace(' with Mechanism Info','',regex=False)
    paired=paired.merge(second[KEYS+['score']].rename(columns={'score':'expert_2'}),on=KEYS,validate='one_to_one')
    if len(paired)!=120 or paired[KEYS].duplicated().any():
        raise ValueError('Expected 60 matched traces x two rubrics from both experts.')
    rows=[]
    for rubric,group in paired.groupby('rubric_key',sort=False):
        models=group.groupby('model_name')[['human_score','expert_2','llm_score']].mean().reset_index()
        for level,frame in [('trace',group),('model',models)]:
            for comparison,x,y in [('independent_annotators','human_score','expert_2'),('annotator_1_vs_llm','human_score','llm_score'),('annotator_2_vs_llm','expert_2','llm_score')]:
                rho,p,lo,hi,n=spearman_with_bootstrap_ci(frame[x],frame[y])
                rows.append(dict(comparison=comparison,agreement_level=level,rubric_key=rubric,reasoning_score=RUBRIC_LABELS[rubric].replace("\n", " "),n_pairs=n,spearman_rho=rho,bootstrap_ci_low=lo,bootstrap_ci_high=hi,two_sided_p_value=p))
    audited = load_audited_scores(paired)
    extremes=[]
    for rubric,group in audited.groupby('rubric_key',sort=False):
        for score in [0,2]:
            subset=group[(group['gpt-5.4']==score)&(group['claude-opus-4.7']==score)]
            extremes.append(dict(rubric_key=rubric,unanimous_llm_score=score,n_traces=len(subset),expert_1_mean=subset.human_score.mean(),expert_2_mean=subset.expert_2.mean(),expert_1_exact_match=float((subset.human_score==score).mean()) if len(subset) else np.nan,expert_2_exact_match=float((subset.expert_2==score).mean()) if len(subset) else np.nan))
    return pd.DataFrame(rows),pd.DataFrame(extremes)

def load_audited_scores(paired):
    raw=[]
    for judge,path in JUDGE_INPUTS.items():
        frame=pd.read_csv(path)
        frame['model_name']=frame.model_label.str.replace(' with Mechanism Info','',regex=False)
        raw.append(frame[KEYS+['score']].rename(columns={'score':judge}))
    raw=raw[0].merge(raw[1],on=KEYS,validate='one_to_one')
    audited=paired.merge(raw,on=KEYS,validate='one_to_one')
    # Consensus requires two non-self judges. Drop the 12 traces generated by
    # either judge model, leaving 48 traces from eight models for this audit.
    audited=audited[~audited.model_name.isin(['GPT-5.4','Claude Opus 4.7'])]
    return audited


def build_saved_audit_tables():
    paired = load_human_and_debiased_llm_scores()
    second = pd.read_csv(ROOT / 'bench/judge/human_runs/human_judge_results_2.csv')
    second['model_name'] = second.model_label.map(_strip_mechanism_suffix)
    paired = paired.merge(second[KEYS + ['score']].rename(columns={'score': 'expert_2'}), on=KEYS, validate='one_to_one')
    audited = load_audited_scores(paired)
    high_low, consensus = [], []
    for axis, rubric in [('PR', 'predictive_sharpness'), ('DED', 'feature_to_action_faithfulness')]:
        group = audited[audited.rubric_key == rubric]
        cases = [(0, 0), (2, 2)] if axis == 'PR' else [(0, 0), (0, 1), (1, 0), (2, 2)]
        for gpt, claude in cases:
            subset = group[(group['gpt-5.4'] == gpt) & (group['claude-opus-4.7'] == claude)]
            label = f'Both judges: {gpt}' if gpt == claude else f'GPT: {gpt}, Claude: {claude}'
            high_low.append(dict(axis=axis, llm_judge_scores=label, n=len(subset), expert_1_mean=subset.human_score.mean(), expert_2_mean=subset.expert_2.mean()))
        for score in ([0, 2] if axis == 'PR' else [1, 2]):
            subset = group[(group['gpt-5.4'] == score) & (group['claude-opus-4.7'] == score)]
            consensus.append(dict(axis=axis, llm_judge_consensus=f'Both judges: {score}', n=len(subset), expert_1_mean=subset.human_score.mean(), expert_2_mean=subset.expert_2.mean(), expert_consensus_mean=(subset.human_score.mean() + subset.expert_2.mean()) / 2, expert_1_exact_agreement=(subset.human_score == score).mean(), expert_2_exact_agreement=(subset.expert_2 == score).mean()))
    return pd.DataFrame(high_low), pd.DataFrame(consensus)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'judge_agreement_analysis')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    agreement,audit=build_tables()
    agreement.to_csv(args.output_dir/'independent_annotator_and_llm_agreement_spearman.csv',index=False,float_format='%.6f')
    high_low, consensus = build_saved_audit_tables()
    high_low.to_csv(args.output_dir/'blinded_expert_audit_llm_high_low.csv',index=False,float_format='%.6f')
    consensus.to_csv(args.output_dir/'blinded_expert_audit_llm_consensus_extremes.csv',index=False,float_format='%.6f')
    print(agreement.to_string(index=False));print(audit.to_string(index=False))

if __name__=='__main__':main()
