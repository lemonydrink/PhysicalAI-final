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
        candidates=[r for r in candidates if r['success'] and r.get('collision_available') is True and type(r.get('collision')) is bool]
        if not candidates:raise ValueError('No successful episode with a verified collision result for '+task['key'])
        for r in candidates:
            if 'collision_free_success' in r and r['collision_free_success']!=(not r['collision']):raise ValueError('inconsistent collision flags: '+task['key'])
        record=min(candidates,key=lambda r:(r['collision'],r['episode_id']))
        chosen.append(dict(task=task['key'],prompt=task['dataset_instruction'],success=record['success'],
                           collision=record['collision'],selected_collision_free=not record['collision'],
                           collision_free_success_candidates=sum(not r['collision'] for r in candidates),
                           colliding_success_candidates=sum(r['collision'] for r in candidates),episode_id=record['episode_id'],
                           source=str(Path(record['video']).resolve()),source_sha256=digest(record['video']),
                           trial_id=record['trial_id'],bundle_hash=record['bundle_hash'],protocol_hash=record['protocol_hash'],steps=record['steps']))
    if len({r['bundle_hash'] for r in chosen})!=1 or len({r['protocol_hash'] for r in chosen})!=1:raise ValueError('mixed checkpoint/protocol videos are not permitted')
    return chosen


def compose(evaluation,output,tasks_path,preview=False,intro_seconds=0.,transition_hold_seconds=0.):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    manifest_path=output.with_suffix('.json');chosen=select_episodes(evaluation,read(tasks_path)['tasks'])
    if intro_seconds<0 or transition_hold_seconds<0:raise ValueError('timing must be nonnegative')
    intro_frames=round(intro_seconds*20);hold_frames=round(transition_hold_seconds*20)
    recipe='collision-aware-success-only-v4-plain-serif-intro' if intro_frames or hold_frames else 'collision-aware-success-only-v2'
    identity=[(x['source_sha256'],x['prompt'],x['episode_id']) for x in chosen]
    if output.exists() and manifest_path.exists():
        old=read(manifest_path)
        if old.get('render_recipe')==recipe and old.get('intro_frames',0)==intro_frames and old.get('transition_hold_frames',0)==hold_frames and [(x['source_sha256'],x['prompt'],x['episode_id']) for x in old['chapters']]==identity and digest(output)==old['output_sha256']:return old
        raise ValueError('output already exists with different provenance')
    partial=output.with_name(output.stem+'.partial.mp4');fps=20;offset=0
    font_file=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    if not font_file.exists():font_file=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc')
    font=ImageFont.truetype(str(font_file),26);small=ImageFont.truetype(str(font_file),20)
    # CPU encoding, two threads; no additional model/GPU process.
    with imageio.get_writer(partial,fps=fps,codec='libx264',pixelformat='yuv420p',quality=8,
                            output_params=['-movflags','+faststart','-threads','2']) as writer:
        if intro_frames:
            title=Image.new('RGB',(1280,720),'white');draw=ImageDraw.Draw(title)
            serif=Path('/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf')
            if not serif.exists():serif=Path('/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf')
            heading=ImageFont.truetype(str(serif),48);body=ImageFont.truetype(str(serif),27)
            subheading=ImageFont.truetype(str(serif),26)
            draw.text((640,40),'SmolVLA + ROS 2',font=heading,fill='black',anchor='mt')
            draw.text((640,102),'Eight-task manipulation experiment in simulation',font=subheading,fill='black',anchor='mt')
            draw.line((80,150,1200,150),fill='black',width=1)
            for index,chapter in enumerate(chosen):
                x=80+(index//4)*620;y=194+(index%4)*112
                draw.text((x,y),f'{index+1}.',font=body,fill='black')
                lines=[];line=''
                for word in chapter['prompt'].split():
                    candidate=(line+' '+word).strip()
                    if line and draw.textlength(candidate,font=body)>510:lines.append(line);line=word
                    else:line=candidate
                if line:lines.append(line)
                if len(lines)>3:raise ValueError('intro prompt exceeds list entry')
                draw.multiline_text((x+36,y),'\n'.join(lines),font=body,fill='black',spacing=4)
            for _ in range(intro_frames):writer.append_data(np.asarray(title))
            offset=intro_frames
        for index,chapter in enumerate(chosen):
            reader=imageio.get_reader(chapter['source']);meta=reader.get_meta_data()
            if abs(float(meta.get('fps',0))-fps)>.01:raise ValueError('source is not native 20 fps')
            prompt='Prompt: '+chapter['prompt'];lines=textwrap.wrap(prompt,width=85,break_long_words=False)
            # Exact prompt text is reflowed only, never translated or paraphrased.
            overlay=Image.new('RGBA',(1280,720));draw=ImageDraw.Draw(overlay)
            height=20+len(lines)*34;draw.rectangle((0,0,1280,height),fill=(0,0,0,190))
            draw.multiline_text((16,12),'\n'.join(lines),font=font,fill='white',spacing=5)
            outcome='SUCCESS / NO COLLISION' if chapter['selected_collision_free'] else 'SUCCESS / COLLISION'
            label=f"{index+1}/8 | {chapter['task']} | {outcome} | episode {chapter['episode_id']}"
            draw.rectangle((0,676,1280,720),fill=(0,0,0,200));draw.text((16,686),label,font=small,fill='white')
            if preview:draw.text((900,640),'BASELINE PREVIEW',font=small,fill=(255,220,80,255))
            count=0
            try:
                for frame in reader:
                    image=Image.fromarray(frame).convert('RGB');scale=min(1280/image.width,720/image.height)
                    image=image.resize((round(image.width*scale),round(image.height*scale)),Image.Resampling.LANCZOS)
                    canvas=Image.new('RGB',(1280,720),'black');canvas.paste(image,((1280-image.width)//2,(720-image.height)//2))
                    last_frame=np.asarray(Image.alpha_composite(canvas.convert('RGBA'),overlay).convert('RGB'))
                    writer.append_data(last_frame);count+=1
            finally:reader.close()
            if count!=chapter['steps']+1:raise ValueError(f'Incomplete episode {chapter["task"]}: frames={count}, steps={chapter["steps"]}')
            chapter.update(start_frame=offset,end_frame_exclusive=offset+count,frames=count,start_seconds=offset/fps,end_seconds=(offset+count)/fps)
            offset+=count
            gap=hold_frames if index<len(chosen)-1 else 0
            chapter.update(hold_frames=gap,hold_start_frame=offset,hold_end_frame_exclusive=offset+gap)
            for _ in range(gap):writer.append_data(last_frame)
            offset+=gap;print(chapter['task'],count,'motion frames',gap,'hold frames',outcome,flush=True)
    # Decode the generated MP4 before exposing it as a completed deliverable.
    reader=imageio.get_reader(partial)
    try:
        if reader.count_frames()!=offset:raise ValueError('encoded video frame count mismatch')
    finally:reader.close()
    partial.replace(output)
    result=dict(output=str(output.resolve()),output_sha256=digest(output),width=1280,height=720,fps=fps,frames=offset,duration_seconds=offset/fps,
                render_recipe=recipe,intro_frames=intro_frames,intro_seconds=intro_frames/fps,
                transition_hold_frames=hold_frames,transition_hold_seconds=hold_frames/fps,total_transition_holds=7,
                motion_speed_multiplier=1.,chapters=chosen,task_count=8,
                all_selected_collision_free=all(r['selected_collision_free'] for r in chosen),
                collision_free_task_count=sum(r['selected_collision_free'] for r in chosen),
                collision_fallback_tasks=[r['task'] for r in chosen if r['collision']],
                collision_metric='simulation non-target object displacement >1mm, not a physical contact sensor',
                all_selected_episodes_successful=all(r['success'] for r in chosen),
                selection_rule='first collision-free success by episode_id; only if none, first success with collision; no failures/unknown collision results',
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
        write(status_path,dict(stage='complete',output=result['output'],duration_seconds=result['duration_seconds'],all_eight_successful=result['all_selected_episodes_successful'],collision_free_tasks=result['collision_free_task_count'],collision_fallback_tasks=result['collision_fallback_tasks']))
    except BaseException as exc:
        write(status_path,dict(stage='failed',error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path);p.add_argument('--wait',action='store_true')
    p.add_argument('--evaluation',type=Path);p.add_argument('--output',type=Path);p.add_argument('--tasks',type=Path,default=ROOT/'data/tasks.resolved.json');p.add_argument('--preview',action='store_true');p.add_argument('--intro-seconds',type=float,default=0.);p.add_argument('--transition-hold-seconds',type=float,default=0.);a=p.parse_args()
    if a.wait:
        if not a.job:p.error('--job required with --wait')
        wait_and_compose(a.job.resolve())
    else:
        if not a.evaluation or not a.output:p.error('--evaluation and --output required')
        compose(a.evaluation,a.output,a.tasks,a.preview,a.intro_seconds,a.transition_hold_seconds)
