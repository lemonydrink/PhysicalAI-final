#!/usr/bin/env python3
"""ROS 2 DDS control/observation transport, isolated from the ML interpreter.

Gateway HTTP is local IPC into ROS. Every reset/action/observation crosses DDS.
Robot-side HTTP is an adapter boundary; this implementation uses the simulator.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
from urllib.request import Request,urlopen
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile,ReliabilityPolicy,HistoryPolicy
from std_msgs.msg import String
from vla_msgs.msg import Observation,ActionChunk
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from physicalai_vla.action_guard import ActionGuard,CODEC,require_hardware_readiness


def post(url,payload):
    with urlopen(Request(url,data=json.dumps(payload,allow_nan=False).encode(),headers={'Content-Type':'application/json'}),timeout=60) as response:return json.load(response)


class Transport(Node):
    def __init__(self,args):
        super().__init__('smolvla_'+args.role,namespace=args.namespace)
        self.args=args;self.guard=ActionGuard(args.bundle_hash,args.max_age);self.counts={'actions':0,'observations':0,'stops':0}
        self.waiters={};self.request_id=0;self.rpc_lock=threading.Lock();self.waiter_lock=threading.Lock();self.in_flight=False;self.worker=ThreadPoolExecutor(max_workers=1)
        qos=QoSProfile(depth=1,history=HistoryPolicy.KEEP_LAST,reliability=ReliabilityPolicy.RELIABLE)
        if args.role=='gateway':
            self.control=self.create_publisher(String,'control',qos);self.actions=self.create_publisher(ActionChunk,'action',qos)
            self.create_subscription(Observation,'observation',self.response,qos)
        else:
            with urlopen(args.backend+'/health',timeout=10) as response:health=json.load(response)
            if health.get('backend')!='simulation':
                if not args.allow_commissioned_hardware:raise ValueError('hardware requires explicit commissioned-adapter activation')
                require_hardware_readiness(health)
            self.observations=self.create_publisher(Observation,'observation',qos)
            self.create_subscription(String,'control',self.control_received,qos)
            self.create_subscription(ActionChunk,'action',self.action_received,qos)
            self.create_timer(.1,self.watchdog)

    def stop(self,reason):
        if not self.guard.fault:self.guard.fault=reason
        self.counts['stops']+=1
        # Simulation freezes without further /step. Hardware requires independent driver watchdog.
        try:post(self.args.backend+'/stop',{'reason':reason})
        except Exception as exc:self.get_logger().error('driver stop acknowledgement failed: '+str(exc))

    def watchdog(self):
        if not self.in_flight and not self.guard.fault and self.guard.watchdog():self.stop('observation watchdog expired')

    def publish_result(self,request_id,result=None,error=None):
        if result is not None:
            obs=result.get('observation',result);obs['transport']='ros2';obs['ros2_namespace']=self.args.namespace
            obs['ros2_counts']=dict(self.counts);self.guard.observation_time=obs['observation_time']
        msg=Observation();msg.episode=self.guard.episode;msg.sequence=request_id;msg.observation_time=time.monotonic()
        msg.payload_json=json.dumps({'request_id':request_id,'ok':error is None,'result':result,'error':error},allow_nan=False)
        self.observations.publish(msg);self.counts['observations']+=1

    def control_received(self,msg):
        data={}
        try:
            data=json.loads(msg.data);rid=data['request_id'];route=data['route']
            if route=='/stop':self.stop('gateway stop');self.publish_result(rid,{'stopped':True,'observation_time':time.monotonic()});return
            if route!='/reset':raise ValueError('unsupported control route')
            self.in_flight=True
            result=post(self.args.backend+'/reset',data['payload']);self.guard.reset(result['episode'],rid,result['observation_time']);self.counts={'actions':0,'observations':0,'stops':0}
            self.publish_result(rid,result)
        except Exception as exc:
            self.stop(str(exc));self.publish_result(data.get('request_id',0),error=str(exc))
        finally:self.in_flight=False

    def action_received(self,msg):
        try:
            self.in_flight=True
            if msg.horizon!=1 or msg.action_dim!=7:raise ValueError('invalid action shape')
            action=self.guard.accept(episode=msg.episode,sequence=msg.sequence,observation_time=msg.observation_time,
                                     codec=msg.codec,bundle_hash=msg.bundle_hash,dt=msg.action_dt,actions=msg.actions)
            result=post(self.args.backend+'/step',{'episode':msg.episode,'action':action})
            self.counts['actions']+=1;self.publish_result(msg.sequence,result)
        except Exception as exc:self.stop(str(exc));self.publish_result(msg.sequence,error=str(exc))
        finally:self.in_flight=False

    def response(self,msg):
        data=json.loads(msg.payload_json)
        with self.waiter_lock:
            if msg.sequence in self.waiters:
                slot=self.waiters[msg.sequence];slot['response']=data;slot['event'].set()

    def rpc(self,route,payload):
        with self.rpc_lock:
            self.request_id+=1;rid=self.request_id;slot={'event':threading.Event()}
            with self.waiter_lock:self.waiters[rid]=slot
            try:
                if route=='/step':
                    msg=ActionChunk();msg.episode=payload['episode'];msg.sequence=rid;msg.observation_time=float(payload['observation_time'])
                    msg.codec=CODEC;msg.bundle_hash=payload['bundle_hash'];msg.action_dt=.05;msg.horizon=1;msg.action_dim=7;msg.actions=payload['action'];self.actions.publish(msg)
                elif route in ('/reset','/stop'):self.control.publish(String(data=json.dumps({'request_id':rid,'route':route,'payload':payload},allow_nan=False)))
                else:raise ValueError('unsupported route')
                if not slot['event'].wait(65):
                    self.control.publish(String(data=json.dumps({'request_id':rid,'route':'/stop','payload':{}})));raise TimeoutError('DDS request timeout')
                response=slot['response']
                if not response['ok']:raise ValueError(response['error'])
                return response['result']
            finally:
                with self.waiter_lock:self.waiters.pop(rid,None)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--role',choices=['gateway','robot'],required=True)
    parser.add_argument('--namespace',required=True);parser.add_argument('--backend');parser.add_argument('--port',type=int)
    parser.add_argument('--allow-commissioned-hardware',action='store_true');parser.add_argument('--bundle-hash',required=True);parser.add_argument('--max-age',type=float,default=2.)
    args=parser.parse_args();rclpy.init();node=Transport(args)
    def spin_ros():
        try:rclpy.spin(node)
        except ExternalShutdownException:pass
    spin=threading.Thread(target=spin_ros,daemon=True);spin.start()
    server=None
    try:
        if args.role=='gateway':
            class Handler(BaseHTTPRequestHandler):
                def send(self,code,value):
                    data=json.dumps(value,allow_nan=False).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
                def do_GET(self):
                    ready=node.control.get_subscription_count()>0 and node.actions.get_subscription_count()>0 and node.count_publishers('observation')>0
                    self.send(200 if ready else 503,{'ready':ready,'transport':'ros2','namespace':args.namespace})
                def do_POST(self):
                    try:
                        length=int(self.headers['Content-Length'])
                        if not 0<length<100000:raise ValueError('invalid payload size')
                        self.send(200,node.rpc(self.path,json.loads(self.rfile.read(length))))
                    except Exception as exc:self.send(409,{'error':str(exc)})
                def log_message(self,*args):pass
            server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
            node.context.on_shutdown(lambda:threading.Thread(target=server.shutdown,daemon=True).start())
            server.serve_forever()
        else:spin.join()
    except KeyboardInterrupt:pass
    finally:
        if server:server.server_close()
        node.worker.shutdown(wait=False,cancel_futures=True);node.destroy_node();rclpy.try_shutdown()
if __name__=='__main__':main()
