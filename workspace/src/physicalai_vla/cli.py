from __future__ import annotations
import argparse
from pathlib import Path
import json
from .io import ROOT,config,read_json,write_json


def main(argv=None):
    p=argparse.ArgumentParser(description='2toINF X-VLA: eight-task simulation, training and ROS 2')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor')
    a=sub.add_parser('assets');a.add_argument('kind',choices=['model','metadata'])
    sub.add_parser('resolve-tasks')
    a=sub.add_parser('split');a.add_argument('source',type=Path);a.add_argument('output',type=Path)
    a=sub.add_parser('prepare');a.add_argument('--max-per-task',type=int,default=0);a.add_argument('--output-dir',type=Path)
    a=sub.add_parser('train');a.add_argument('--output',type=Path,required=True);a.add_argument('--steps',type=int)
    a.add_argument('--wandb-mode',choices=['online','offline','disabled']);a.add_argument('--smoke',action='store_true')
    a.add_argument('--resume',type=Path);a.add_argument('--manifest',type=Path);a.add_argument('--dry-run',action='store_true')
    a=sub.add_parser('evaluate');a.add_argument('--output',required=True,type=Path);a.add_argument('--adapter',type=Path)
    a.add_argument('--per-task',type=int);a.add_argument('--task');a.add_argument('--offset',type=int);a.add_argument('--frame',choices=['body','controller'],default='controller')
    a=sub.add_parser('score');a.add_argument('directory',type=Path);a.add_argument('--per-task',type=int,default=10)
    a=sub.add_parser('log-eval');a.add_argument('directory',type=Path);a.add_argument('--wandb-mode',choices=['online','offline','disabled'],default='offline')
    a=sub.add_parser('export');a.add_argument('--output',required=True,type=Path);a.add_argument('--adapter',type=Path)
    a=sub.add_parser('verify-bundle');a.add_argument('directory',type=Path)
    a=sub.add_parser('serve');a.add_argument('--kind',choices=['policy','sim'],required=True);a.add_argument('--port',type=int,default=8010);a.add_argument('--adapter',type=Path)
    a=sub.add_parser('ros');a.add_argument('--role',choices=['policy','bridge'],required=True);a.add_argument('--url',default='http://127.0.0.1:8010');a.add_argument('--task',default='milk_basket')
    args=p.parse_args(argv)
    if args.command=='doctor':
        import importlib.util
        print(json.dumps({'backend':'simulation','batch_size':config()['batch_size'],
            'model_present':(ROOT/'models/base/model.safetensors').is_file(),
            'resolved_tasks':(ROOT/'data/tasks.resolved.json').is_file(),
            'cuda':__import__('torch').cuda.is_available() if importlib.util.find_spec('torch') else None,
            'codec_status':read_json(ROOT/'configs/codec.json')['validation_status']},indent=2))
    elif args.command=='assets':
        from .assets import download_model,download_metadata
        print(download_model() if args.kind=='model' else download_metadata())
    elif args.command=='resolve-tasks':
        from .data import resolve_tasks
        print(json.dumps(resolve_tasks(),indent=2))
    elif args.command=='split':
        from .data import split_manifest
        split_manifest(args.source,args.output)
    elif args.command=='prepare':
        from .data import prepare
        print(json.dumps(prepare(args.max_per_task,output_dir=args.output_dir)['counts'],indent=2))
    elif args.command=='train':
        if args.dry_run:
            print(json.dumps({'config':config(),'output':str(args.output),'steps':args.steps,'smoke':args.smoke,'wandb_mode':args.wandb_mode,'execution':False},indent=2));return
        from .training import train
        train(args.output,args.steps,args.wandb_mode,args.smoke,args.resume,args.manifest)
    elif args.command=='evaluate':
        from .simulation import evaluate
        evaluate(args.output,args.adapter,args.per_task,args.task,args.offset,args.frame)
    elif args.command=='score':
        from .evaluation import summarize
        records=[read_json(x) for x in args.directory.glob('*.json')]
        result=summarize([r for r in records if 'trial_id' in r],[t['key'] for t in read_json(ROOT/'data/tasks.resolved.json')['tasks']],args.per_task,config()['target_macro_success'])
        write_json(args.directory/'summary.json',result);print(json.dumps(result,indent=2))
    elif args.command=='log-eval':
        from .evaluation import log_evaluation
        print(log_evaluation(args.directory,args.wandb_mode))
    elif args.command=='export':
        from .artifacts import export_bundle
        print(export_bundle(args.output,args.adapter))
    elif args.command=='verify-bundle':
        from .artifacts import verify_bundle
        print('Verified',len(verify_bundle(args.directory)['files']),'files')
    elif args.command=='serve':
        from .services import serve
        serve(args.kind,args.port,args.adapter)
    elif args.command=='ros':
        import subprocess,os
        env=os.environ.copy()
        env.pop('VIRTUAL_ENV',None);env.pop('CONDA_PREFIX',None)
        subprocess.run(['bash',str(ROOT/'scripts/run_ros.sh'),'--role',args.role,'--url',args.url,'--task',args.task],check=True,env=env)

if __name__=='__main__':main()
