#!/usr/bin/env python3
"""Reserve a best-checkpoint ROS 2 evaluation: two simulators, 80 held-out trials."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime,fcntl,json,os,subprocess,time
from pathlib import Path
from urllib.request import Request,urlopen
from run import HERE,ROOT,read_json,write_json,digest
from staged_job import environment,status,command


def rpc(url,route,payload):
    with urlopen(Request(url+route,data=json.dumps(payload,allow_nan=False).encode(),headers={'Content-Type':'application/json'}),timeout=65) as response:return json.load(response)


def healthy(child,url):
    for _ in range(120):
        if child.poll() is not None:raise RuntimeError('ROS/simulator child exited: '+url)
        try:
            with urlopen(url+'/health',timeout=2) as response:
                if response.status==200 and json.load(response).get('ready'):return
        except OSError:time.sleep(1)
    raise TimeoutError('startup '+url)


def execute(job,smoke=False,policy_smoke=False):
    job=job.resolve();plan=read_json(job/'plan.json');env=environment(job);final=plan['final_evaluation']
    if final['workers']!=2 or final['transport']!='ros2':raise ValueError('final evaluation requires ROS2/two workers')
    output=job/('ros2-preflight' if smoke else 'final-ros2-best80');output.mkdir(exist_ok=True)
    lock=(output/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    checkpoint=Path(read_json(job/'best_checkpoint.json')['checkpoint']);bundle=digest(checkpoint/'model.safetensors')
    children=[];logs=[];urls=[];ros_env=dict(env,ROS_DOMAIN_ID=str(final['ros_domain_id']),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    ros_env.pop('VIRTUAL_ENV',None);ros_env.pop('CONDA_PREFIX',None)
    def launch(args,child_env,name):
        log=(output/(name+'.log')).open('a');logs.append(log)
        child=subprocess.Popen([str(x) for x in args],cwd=ROOT,env=child_env,stdout=log,stderr=subprocess.STDOUT);children.append(child);return child
    try:
        for index in range(2):
            port=final['sim_port_base']+index;sim=f'http://127.0.0.1:{port}';namespace=f'/physicalai/final/worker_{index}'
            child=launch([ROOT/'.venv/bin/vla','serve','--kind','sim','--port',port],dict(env,PHYSICALAI_SIM_INSTANCE=str(port),OMP_NUM_THREADS='1'),'sim-'+str(index));healthy(child,sim)
            common=['--namespace',namespace,'--bundle-hash',bundle,'--max-age',final['max_observation_age_seconds']]
            launch(['bash',ROOT/'scripts/run_smolvla_ros.sh','--role','robot','--backend',sim,*common],ros_env,'robot-'+str(index))
            port=final['gateway_port_base']+index;url=f'http://127.0.0.1:{port}'
            child=launch(['bash',ROOT/'scripts/run_smolvla_ros.sh','--role','gateway','--port',port,*common],ros_env,'gateway-'+str(index));healthy(child,url);urls.append(url)
        if smoke:
            def trial(index):
                url=urls[index];payload={'task':'milk_basket','index':index,'seed':42+index};obs=rpc(url,'/reset',payload);assert obs['transport']=='ros2'
                for _ in range(3):
                    result=rpc(url,'/step',{'episode':obs['episode'],'action':[0.,0.,0.,0.,0.,0.,-1.],'observation_time':obs['observation_time'],'bundle_hash':bundle});obs=result['observation']
                assert obs['ros2_counts']['actions']==3 and result['steps']==3
                try:rpc(url,'/step',{'episode':obs['episode'],'action':[0.]*7,'observation_time':obs['observation_time']-10,'bundle_hash':bundle})
                except OSError:pass
                else:raise AssertionError('expired action accepted')
                assert rpc(url,'/stop',{})['stopped'];obs=rpc(url,'/reset',payload);assert obs['transport']=='ros2'
                return dict(worker=index,namespace=obs['ros2_namespace'],dds_actions_verified=3,expired_action_rejected=True,stop_acknowledged=True,reset_recovers=True)
            with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(trial,range(2)))
            record=dict(passed=True,workers=2,transport='ros2_dds',results=results,hardware_tested=False,timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),transport_code_sha256=digest(ROOT/'scripts/smolvla_ros_transport.py'),tested_checkpoint=str(checkpoint),tested_weight_sha256=bundle)
            if policy_smoke:
                attempt=1
                while (job/f'ros2-policy-smoke-{attempt:03d}').exists():attempt+=1
                policy_output=job/f'ros2-policy-smoke-{attempt:03d}'
                command([HERE/'.venv/bin/python',HERE/'run.py','evaluate','--manifest',plan['evaluation_manifest_path'],'--output',policy_output,'--checkpoint',checkpoint,'--worker-urls',*urls,'--task','milk_basket','--episodes',2,'--max-steps',5,'--resume','--initial-state-offset',0,'--transport','ros2'],env)
                rows=[read_json(p) for p in policy_output.glob('*.json')];rows=[v for v in rows if 'trial_id' in v]
                if len(rows)!=2 or not all(v['ros2_counts']['actions']==v['steps'] and v['transport']=='ros2_dds' for v in rows):raise ValueError('policy/DDS smoke evidence missing')
                record['policy_smoke_output']=str(policy_output)
                record['actual_policy_rollouts']=len(rows);record['actual_policy_actions']=sum(v['steps'] for v in rows)
            write_json(output/'report.json',record);print(json.dumps(record),flush=True);return record
        preflight=read_json(job/'ros2-preflight/report.json')
        if not preflight.get('passed') or preflight['transport_code_sha256']!=digest(ROOT/'scripts/smolvla_ros_transport.py'):raise ValueError('matching ROS 2 preflight required')
        status(job,'final_ros2_evaluation',checkpoint=str(checkpoint),planned_trials=80,simulator_workers=2,initial_state_offset=final['initial_state_offset'])
        args=[HERE/'.venv/bin/python',HERE/'run.py','evaluate','--manifest',plan['evaluation_manifest_path'],'--output',output,'--checkpoint',checkpoint,'--worker-urls',*urls,'--episodes',10,'--max-steps',300,'--resume','--initial-state-offset',final['initial_state_offset'],'--transport','ros2']
        command(args,env);command([ROOT/'.venv/bin/vla','score',output,'--per-task',10],env)
        summary=read_json(output/'summary.json');rows=[read_json(p) for p in output.glob('*.json')];rows=[t for t in rows if 'trial_id' in t]
        if len(rows)!=80 or not summary['all_trials_present'] or not all(t.get('transport')=='ros2_dds' and (t['infra_error'] or t['ros2_counts']['actions']==t['steps']) for t in rows):raise ValueError('missing actual DDS control evidence')
        command([ROOT/'.venv/bin/vla','log-eval',output,'--wandb-mode','offline'],env)
        infra_errors=sum(t['infra_error'] for t in rows)
        record=dict(infra_errors=infra_errors,benchmark_valid=infra_errors==0,summary=summary,checkpoint=str(checkpoint),transport='ros2_dds',simulator_workers=2,initial_state_offset=final['initial_state_offset'],all_80_trials_have_dds_action_evidence=infra_errors==0,hardware_tested=False,physical_robot_success_claim=False)
        write_json(job/'final_ros2_result.json',record)
        write_final_report(job,record)
        status(job,'complete',training_complete=True,final_ros2_complete=True,final_macro_success_rate=summary['macro_success_rate'],final_accepted=summary['accepted'] and infra_errors==0,best_checkpoint=str(checkpoint),hardware_tested=False)
        return record
    finally:
        for child in reversed(children):
            if child.poll() is None:child.terminate()
        for child in reversed(children):
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        for log in logs:log.close()



def write_final_report(job,record):
    summary=record['summary']
    rows='\n'.join(f"| {t['task']} | {t['successes']}/10 | {t['success_rate']:.1%} | {t['infra_errors']} |" for t in summary['tasks'])
    report=("# 最終ROS 2評価結果\n\n"
            f"Checkpoint: `{record['checkpoint']}`\n\n"
            f"ROS 2 DDS、2並列、8タスク各10回。初期状態index20〜29。macro成功率 **{summary['macro_success_rate']:.2%}**。\n\n"
            f"インフラエラー: {record['infra_errors']}件。通信検証を含めた評価有効性: {record['benchmark_valid']}。\n\n"
            "| タスク | 成功 | 成功率 | インフラエラー |\n|---|---:|---:|---:|\n"+rows+
            "\n\n失敗・インフラエラーも分母80に含め、成功するまでの再試行で置き換えていない。"
            "実機は未検証であり、この結果は実機でのタスク成功を保証しない。"
            "動画とtrial JSONは同じfinal-ros2-best80ディレクトリに保存。\n")
    (job/'final-ros2-best80/RESULT.md').write_text(report)


def wait_then_execute(job):
    plan=read_json(job/'plan.json');snapshot=job/'training-complete.json';last=None
    while True:
        if (job/'final_ros2_result.json').exists():return
        record=read_json(job/'status.json');query=subprocess.run(['systemctl','--user','show',plan['final_evaluation']['training_unit'],'--property=ActiveState','--value'],capture_output=True,text=True)
        if query.returncode==0 and query.stdout.strip()=='inactive' and (record.get('stage')=='complete' or snapshot.exists()):
            if not snapshot.exists():write_json(snapshot,record)
            break
        state=(record.get('stage'),query.stdout.strip())
        if state!=last:print(json.dumps({'waiting_for_training':state,'final_evaluation':'ROS2/two workers/best checkpoint/80 trials'}),flush=True);last=state
        time.sleep(15)
    try:execute(job)
    except BaseException as exc:status(job,'final_ros2_failed',error=repr(exc),final_ros2_complete=False);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',required=True,type=Path);p.add_argument('--smoke',action='store_true');p.add_argument('--policy-smoke',action='store_true');p.add_argument('--wait',action='store_true');a=p.parse_args()
    if a.wait:wait_then_execute(a.job.resolve())
    else:execute(a.job,a.smoke or a.policy_smoke,a.policy_smoke)
