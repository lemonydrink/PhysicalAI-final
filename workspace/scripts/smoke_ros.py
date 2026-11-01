#!/usr/bin/env python3
"""Real DDS roundtrip of generated Observation/ActionChunk types."""
import time
import rclpy
from vla_msgs.msg import Observation,ActionChunk
rclpy.init();pub_node=rclpy.create_node('vla_smoke_publisher');sub_node=rclpy.create_node('vla_smoke_subscriber')
seen=[]
sub=sub_node.create_subscription(ActionChunk,'/vla/smoke',lambda m:seen.append(m),1)
pub=pub_node.create_publisher(ActionChunk,'/vla/smoke',1)
try:
 deadline=time.monotonic()+10
 while not seen and time.monotonic()<deadline:
  m=ActionChunk();m.episode='smoke';m.sequence=1;m.horizon=1;m.action_dim=10;m.actions=[0.,0.,0.,1.,0.,0.,0.,1.,0.,1.]
  pub.publish(m);rclpy.spin_once(sub_node,timeout_sec=.1)
 assert seen and seen[0].episode=='smoke' and len(seen[0].actions)==10
 print('PASS: custom ROS 2 ActionChunk DDS roundtrip')
finally:sub_node.destroy_node();pub_node.destroy_node();rclpy.shutdown()
