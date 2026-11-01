#!/usr/bin/env python3
"""Run with ROS Python; ML/simulator remain in their dedicated interpreter.

No torch/numpy imports: usable from the independently installed Jazzy/Pixi.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from urllib.request import Request,urlopen
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,ReliabilityPolicy,HistoryPolicy
from std_msgs.msg import String
from vla_msgs.msg import Observation,ActionChunk


def post(url,data):
    req=Request(url,data=json.dumps(data,allow_nan=False).encode(),headers={'Content-Type':'application/json'})
    with urlopen(req,timeout=5) as response:return json.load(response)

class Proxy(Node):
    def __init__(self,args):
        super().__init__('vla_'+args.role)
        self.args=args;self.pending=None;self.latest=None;self.chunk=None
        self.pool=ThreadPoolExecutor(max_workers=1)
        qos=QoSProfile(depth=1,history=HistoryPolicy.KEEP_LAST,reliability=ReliabilityPolicy.RELIABLE)
        self.status=self.create_publisher(String,'/vla/status',qos)
        if args.role=='policy':
            self.publisher=self.create_publisher(ActionChunk,'/vla/action_chunk',qos)
            self.create_subscription(Observation,'/vla/observation',self.observe,qos)
            self.create_timer(.02,self.poll)
        else:
            self.publisher=self.create_publisher(Observation,'/vla/observation',qos)
            self.create_subscription(ActionChunk,'/vla/action_chunk',self.accept,qos)
            self.initial=post(args.url+'/reset',{'task':args.task,'index':0,'seed':42})
            self.create_timer(.05,self.tick)
    def observe(self,msg):
        self.latest=json.loads(msg.payload_json)
    def poll(self):
        if self.pending is not None and self.pending.done():
            try:
                c=self.pending.result();msg=ActionChunk()
                msg.episode=c['episode'];msg.sequence=c['sequence'];msg.observation_time=c['observation_time']
                msg.codec=c['codec'];msg.bundle_hash=c['bundle_hash'];msg.action_dt=c['dt']
                msg.horizon=len(c['actions']);msg.action_dim=10;msg.actions=[float(v) for a in c['actions'] for v in a]
                self.publisher.publish(msg)
            except Exception as exc:self.get_logger().error(str(exc))
            self.pending=None
        if self.pending is None and self.latest is not None:
            self.pending=self.pool.submit(post,self.args.url+'/infer',self.latest);self.latest=None
    def accept(self,msg):
        if msg.action_dim!=10 or msg.horizon>128 or len(msg.actions)!=msg.horizon*10:
            self.get_logger().error('invalid chunk dimensions');return
        self.chunk={'episode':msg.episode,'sequence':msg.sequence,'observation_time':msg.observation_time,
                    'dt':msg.action_dt,'codec':msg.codec,'bundle_hash':msg.bundle_hash,
                    'actions':[list(msg.actions[i:i+10]) for i in range(0,len(msg.actions),10)]}
    def tick(self):
        try:
            payload={'chunk':self.chunk};self.chunk=None
            result=post(self.args.url+'/tick',payload);obs=result['observation']
            msg=Observation();msg.episode=obs['episode'];msg.sequence=obs['sequence'];msg.observation_time=obs['observation_time']
            msg.payload_json=json.dumps(obs,allow_nan=False);self.publisher.publish(msg)
            self.status.publish(String(data=json.dumps({k:result[k] for k in ('state','reason','success')})))
        except Exception as exc:self.get_logger().error(str(exc))
    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role',choices=['policy','bridge'],required=True)
    p.add_argument('--url',default='http://127.0.0.1:8010');p.add_argument('--task',default='milk_basket')
    args=p.parse_args();rclpy.init();node=None
    try:node=Proxy(args);rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        if node:node.close();node.destroy_node()
        rclpy.shutdown()
if __name__=='__main__':main()
