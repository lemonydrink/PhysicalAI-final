#!/usr/bin/env bash
set -eo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PHYSICALAI_ROS_ENV:-$PROJECT_ROOT/../ros2.env}"
source "$PROJECT_ROOT/ros2_ws/install/setup.bash"
set -u
exec python "$PROJECT_ROOT/scripts/smolvla_ros_transport.py" "$@"
