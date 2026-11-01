"""Local-only HTTP IPC between Python/ROS environments (simulation only)."""
from __future__ import annotations
import base64
import io
import threading
import time
import uuid
import numpy as np
from PIL import Image
from .io import ROOT,config,read_json
from .scheduler import Chunk,Scheduler
from .codec import absolute_to_delta


def encode_observation(obs,episode,sequence):
    out={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in obs.items() if k not in ('front','wrist')}
    for key in ('front','wrist'):
        stream=io.BytesIO();Image.fromarray(obs[key]).save(stream,format='PNG')
        out[key]=base64.b64encode(stream.getvalue()).decode()
    out.update(episode=episode,sequence=sequence,observation_time=time.monotonic())
    return out


def decode_observation(obs):
    out=dict(obs)
    for key in ('front','wrist'):
        data=base64.b64decode(obs[key],validate=True)
        if len(data)>2_000_000:raise ValueError('image payload too large')
        image=Image.open(io.BytesIO(data))
        if image.size!=(256,256) or image.mode!='RGB':raise ValueError('invalid image format')
        out[key]=np.asarray(image)
    out['proprio_mode']='measured'
    return out


def serve(kind,port,adapter=None):
    import uvicorn
    from fastapi import FastAPI,HTTPException
    app=FastAPI();lock=threading.Lock();state={}
    codec=read_json(ROOT/'configs/inference.json')['profile_id']
    if kind=='policy':
        from .model import Policy
        state['policy']=Policy(adapter=adapter)
    else:state['scheduler']=Scheduler(codec,max_age=read_json(ROOT/'configs/ros.json')['max_observation_age_seconds'])

    @app.get('/health')
    def health():return {'backend':'simulation','kind':kind,'ready':True}

    @app.post('/reset')
    def reset(payload:dict):
        if kind!='sim':raise HTTPException(400,'sim only')
        from .simulation import Environment
        with lock:
            tasks=read_json(ROOT/'data/tasks.resolved.json')['tasks']
            matches=[t for t in tasks if t['key']==payload['task']]
            if not matches:raise HTTPException(400,'unknown task')
            if state.get('env'):state['env'].close()
            state['env']=Environment(matches[0],pose_frame='controller');state['episode']=str(uuid.uuid4());state['sequence']=0
            state['obs']=state['env'].reset(int(payload.get('index',0)),int(payload.get('seed',42)))
            state['scheduler'].reset(state['episode']);state['done']=False;state['bundle_hash']=None
            return encode_observation(state['obs'],state['episode'],0)

    @app.post('/step')
    def relative_step(payload:dict):
        if kind!='sim':raise HTTPException(400,'sim only')
        with lock:
            if 'env' not in state or payload.get('episode')!=state.get('episode'):raise HTTPException(409,'episode mismatch')
            if not state['done']:state['obs'],state['done']=state['env'].step(payload['action'])
            state['sequence']+=1
            return {'observation':encode_observation(state['obs'],state['episode'],state['sequence']),
                    'success':state['done'],'steps':state['env'].steps,
                    'initial_state_hash':state['env'].initial_hash,
                    'collision':any(v>.001 for v in state['env'].max_object_displacement.values())}

    @app.post('/infer')
    def infer(payload:dict):
        if kind!='policy':raise HTTPException(400,'policy only')
        if not lock.acquire(blocking=False):raise HTTPException(409,'inference busy')
        try:
            if payload['episode']!=state.get('episode'):
                state['policy'].reset(42);state['episode']=payload['episode']
            actions=state['policy'].predict(decode_observation(payload))
            return {'episode':payload['episode'],'sequence':payload['sequence'],'observation_time':payload['observation_time'],
                    'dt':config()['action_dt'],'actions':actions.tolist(),'codec':codec,'bundle_hash':state['policy'].bundle_hash}
        finally:lock.release()

    @app.post('/tick')
    def tick(payload:dict):
        if kind!='sim':raise HTTPException(400,'sim only')
        with lock:
            if 'env' not in state:raise HTTPException(409,'reset required')
            now=time.monotonic();result=None
            if payload.get('chunk'):
                c=payload['chunk']
                if state['bundle_hash'] is not None and c['bundle_hash']!=state['bundle_hash']:
                    state['scheduler'].pause('model_changed');raise HTTPException(409,'model identity changed')
                state['bundle_hash']=c['bundle_hash']
                result=state['scheduler'].accept(Chunk(c['episode'],c['sequence'],c['observation_time'],c['dt'],np.asarray(c['actions']),c['codec']),now)
            target=state['scheduler'].next(now)
            # PAUSED freezes simulation time rather than extending a move.
            if target is not None and not state['done']:
                obs=state['obs'];action=absolute_to_delta(target,obs['position'],obs['matrix'])
                state['obs'],state['done']=state['env'].step(action)
            state['sequence']+=1
            return {'observation':encode_observation(state['obs'],state['episode'],state['sequence']),
                    'state':state['scheduler'].state,'reason':state['scheduler'].reason,'success':state['done'],'chunk_result':result}
    uvicorn.run(app,host='127.0.0.1',port=port,workers=1)
