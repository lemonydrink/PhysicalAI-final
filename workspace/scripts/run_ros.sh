#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/../ros2.env"
if [[ ! -f "$PROJECT_ROOT/ros2_ws/install/setup.bash" ]]; then
  echo '先に bash scripts/build_ros.sh を実行してください。' >&2
  exit 1
fi
source "$PROJECT_ROOT/ros2_ws/install/setup.bash"
exec python "$PROJECT_ROOT/scripts/ros_nodes.py" "$@"
