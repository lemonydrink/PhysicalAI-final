"""Estimate local job finish from observed update and rollout wall times."""
import argparse
import datetime
import json
from pathlib import Path
from run import HERE,read_json,write_json


def estimate(job):
    now=datetime.datetime.now(datetime.timezone.utc);plan=read_json(job/'plan.json');total=plan['optimizer_steps']
    step=0;train_seconds=0.;train_updates=0;full_validation=False
    for metrics in job.glob('train-*/metrics.jsonl'):
        lines=metrics.read_text().splitlines()
        if not lines:continue
        rows=[json.loads(l) for l in lines];step=max(step,rows[-1]['global_step'])
        run=read_json(metrics.parent/'run.json');start=metrics.parent/'run.json'
        elapsed=metrics.stat().st_mtime-start.stat().st_mtime
        for d in job.glob('eval-*'):
            summary=d/'summary.json';protocol=d/'protocol.json'
            if summary.exists() and protocol.exists():
                left=max(start.stat().st_mtime,protocol.stat().st_mtime);right=min(metrics.stat().st_mtime,summary.stat().st_mtime)
                elapsed-=max(0.,right-left)
        train_seconds+=max(0.,elapsed);train_updates+=len(rows)
        full_validation|=bool(rows[-1].get('validation/full'))
    source='BurgQ measured' if train_updates>=30 else 'previous batch32 measured; BurgQ training not yet measured'
    if train_updates<30:
        p=HERE/'runs/staged-batch32-20261101/train-001/metrics.jsonl'
        previous=p.read_text().splitlines()
        seconds_per_update=(p.stat().st_mtime-(p.parent/'run.json').stat().st_mtime)/len(previous)
    else:seconds_per_update=train_seconds/train_updates
    observed=0;span=0.;intervals=0
    for directory in [job/'baseline80',*job.glob('eval-*')]:
        times=[]
        for p in directory.glob('*.json'):
            if 'trial_id' in read_json(p):times.append(p.stat().st_mtime)
        observed+=len(times)
        if len(times)>1:span+=max(times)-min(times);intervals+=len(times)-1
    if intervals<1:return {'phase':'insufficient_rollout_measurements'}
    trial_seconds=span/intervals
    checkpoints=len(range(plan['checkpoint_interval'],total,plan['checkpoint_interval']))+1
    pending_trials=max(0,(checkpoints+1)*80-observed)
    remaining_training=max(0,total-step)*seconds_per_update
    remaining_rollout=pending_trials*trial_seconds
    validation=0. if full_validation else 600.  # Explicit allowance, not claimed as directly timed BurgQ validation.
    seconds=remaining_training+remaining_rollout+validation
    tz=datetime.timezone(datetime.timedelta(hours=9))
    record={'generated_at_jst':now.astimezone(tz).isoformat(),'optimizer_step':step,'total_optimizer_steps':total,
            'seconds_per_optimizer_update':seconds_per_update,'training_timing_source':source,'new_model_measured_updates':train_updates,
            'completed_rollout_trials':observed,'rollout_timing_intervals':intervals,'seconds_per_rollout_trial':trial_seconds,'remaining_rollout_trials':pending_trials,
            'remaining_training_seconds':remaining_training,'remaining_rollout_seconds':remaining_rollout,'full_validation_allowance_seconds':validation,
            'estimated_finish_jst':(now+datetime.timedelta(seconds=seconds)).astimezone(tz).isoformat(),
            'limitations':'Rollout difficulty and post-SFT success change episode lengths. Full validation allowance is provisional. New-model update speed replaces predecessor proxy after 30 measured updates.'}
    write_json(job/'eta.json',record);return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path,required=True);a=p.parse_args();print(json.dumps(estimate(a.job),ensure_ascii=False,indent=2))
