#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
uv sync --locked --extra ml --extra sim --extra dev
.venv/bin/vla assets metadata
.venv/bin/vla resolve-tasks
.venv/bin/vla assets model
.venv/bin/vla doctor
