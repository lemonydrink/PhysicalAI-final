#!/usr/bin/env python3
"""Compose eight complete episodes, with exact model prompts, after final ROS 2 evaluation."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap
import time

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text())


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)


def select_episodes(evaluation,tasks):
    if len(tasks)!=8 or len({t['key'] for t in tasks})!=8:raise ValueError('exactly eight distinct tasks required')
    records=[read(p) for p in Path(evaluation).glob('*.json')];records=[r for r in records if 'trial_id' in r]
    chosen=[]
    for task in tasks:
        candidates=[r for r in records if r['task']==task['key'] and not r.get('infra_error') and r.get('video') and Path(r['video']).is_file() and Path(r['video']).stat().st_size>0]
        if not candidates:raise ValueError('No complete, non-infrastructure episode video for '+task['key'])
        record=min(candidates,key=lambda r:(not r['success'],r['episode_id']))
        chosen.append(dict(task=task['key'],prompt=task['dataset_instruction'],success=record['success'],
                           no_success_available=not any(r['success'] for r in candidates),episode_id=record['episode_id'],
                           source=str(Path(record['video']).resolve()),source_sha256=digest(record['video']),
                           trial_id=record['trial_id'],bundle_hash=record['bundle_hash'],protocol_hash=record['protocol_hash'],steps=record['steps']))
    if len({r['bundle_hash'] for r in chosen})!=1 or len({r['protocol_hash'] for r in chosen})!=1:raise ValueError('mixed checkpoint/protocol videos are not permitted')
    return chosen


def compose(evaluation,output,tasks_path,preview=False):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    manifest_path=output.with_suffix('.json');chosen=select_episodes(evaluation,read(tasks_path)['tasks'])
    identity=[(x['source_sha256'],x['prompt'],x['episode_id']) for x in chosen]
    if output.exists() and manifest_path.exists():
        old=read(manifest_path)
        if [(x['source_sha256'],x['prompt'],x['episode_id']) for x in old['chapters']]==identity and digest(output)==old['output_sha256']:return old
        raise ValueError('output already exists with different provenance')
    partial=output.with_name(output.stem+'.partial.mp4');fps=20;offset=0
    font_file=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    if not font_file.exists():font_file=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc')
    font=ImageFont.truetype(str(font_file),26);small=ImageFont.truetype(str(font_file),20)
    # CPU encoding, two threads; no additional model/GPU process.
    with imageio.get_writer(partial,fps=fps,codec='libx264',pixelformat='yuv420p',quality=8,
                            output_params=['-movflags','+faststart','-threads','2']) as writer:
        for index,chapter in enumerate(chosen):
            reader=imageio.get_reader(chapter['source']);meta=reader.get_meta_data()
            if abs(float(meta.get('fps',0))-fps)>.01:raise ValueError('source is not native 20 fps')
            prompt='Prompt: '+chapter['prompt'];lines=textwrap.wrap(prompt,width=85,break_long_words=False)
            # Exact prompt text is reflowed only, never translated or paraphrased.
            overlay=Image.new('RGBA',(1280,720));draw=ImageDraw.Draw(overlay)
            height=20+len(lines)*34;draw.rectangle((0,0,1280,height),fill=(0,0,0,190))
            draw.multiline_text((16,12),'\n'.join(lines),font=font,fill='white',spacing=5)
            outcome='SUCCESS' if chapter['success'] else 'FAILURE - no successful episode available'
            label=f"{index+1}/8 | {chapter['task']} | {outcome} | episode {chapter['episode_id']}"
            draw.rectangle((0,676,1280,720),fill=(0,0,0,200));draw.text((16,686),label,font=small,fill='white')
            if preview:draw.text((900,640),'BASELINE PREVIEW',font=small,fill=(255,220,80,255))
            count=0
            try:
                for frame in reader:
                    image=Image.fromarray(frame).convert('RGB');scale=min(1280/image.width,720/image.height)
                    image=image.resize((round(image.width*scale),round(image.height*scale)),Image.Resampling.LANCZOS)
                    canvas=Image.new('RGB',(1280,720),'black');canvas.paste(image,((1280-image.width)//2,(720-image.height)//2))
                    writer.append_data(np.asarray(Image.alpha_composite(canvas.convert('RGBA'),overlay).convert('RGB')));count+=1
            finally:reader.close()
            if count!=chapter['steps']+1:raise ValueError(f'Incomplete episode {chapter["task"]}: frames={count}, steps={chapter["steps"]}')
            chapter.update(start_frame=offset,end_frame_exclusive=offset+count,frames=count,start_seconds=offset/fps,end_seconds=(offset+count)/fps)
            offset+=count;print(chapter['task'],count,'frames',outcome,flush=True)
    # Decode the generated MP4 before exposing it as a completed deliverable.
    reader=imageio.get_reader(partial)
    try:
        if reader.count_frames()!=offset:raise ValueError('encoded video frame count mismatch')
    finally:reader.close()
    partial.replace(output)
    result=dict(output=str(output.resolve()),output_sha256=digest(output),width=1280,height=720,fps=fps,frames=offset,duration_seconds=offset/fps,
                chapters=chosen,task_count=8,all_selected_episodes_successful=all(r['success'] for r in chosen),
                selection_rule='first successful non-infrastructure episode by episode_id; otherwise first complete failed episode with FAILURE label',
                prompt_source='tasks.resolved.json dataset_instruction: exact string supplied to policy',
                no_episode_trimming=True,preview_baseline=preview,created_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(manifest_path,result);return result


def wait_and_compose(job):
    plan=read(job/'plan.json')['video_compilation'];status_path=job/'video-status.json';previous=None
    while True:
        ready=(job/'final_ros2_result.json').exists()
        query=subprocess.run(['systemctl','--user','show',plan['final_evaluation_unit'],'--property=ActiveState','--value'],capture_output=True,text=True)
        if ready and query.returncode==0 and query.stdout.strip()=='inactive':break
        stage=read(job/'status.json').get('stage')
        if stage!=previous:
            write(status_path,dict(stage='waiting_for_final_ros2',evaluation_stage=stage));print('Waiting for final ROS2:',stage,flush=True);previous=stage
        time.sleep(15)
    try:
        write(status_path,dict(stage='rendering',output=plan['output']))
        result=compose(job/'final-ros2-best80',plan['output'],ROOT/'data/tasks.resolved.json')
        write(status_path,dict(stage='complete',output=result['output'],duration_seconds=result['duration_seconds'],all_eight_successful=result['all_selected_episodes_successful']))
    except BaseException as exc:
        write(status_path,dict(stage='failed',error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path);p.add_argument('--wait',action='store_true')
    p.add_argument('--evaluation',type=Path);p.add_argument('--output',type=Path);p.add_argument('--tasks',type=Path,default=ROOT/'data/tasks.resolved.json');p.add_argument('--preview',action='store_true');a=p.parse_args()
    if a.wait:
        if not a.job:p.error('--job required with --wait')
        wait_and_compose(a.job.resolve())
    else:
        if not a.evaluation or not a.output:p.error('--evaluation and --output required')
        compose(a.evaluation,a.output,a.tasks,a.preview)
