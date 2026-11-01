#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/../ros2.env"
cd "$PROJECT_ROOT/ros2_ws"
colcon build --base-paths src --symlink-install
