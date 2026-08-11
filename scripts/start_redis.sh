#!/usr/bin/env bash
# Start Redis idempotently (for WSL / Linux without systemd).
#
# Usage:
#   ./scripts/start_redis.sh
#
# WSL auto-start: add this to /etc/wsl.conf under [boot]:
#   [boot]
#   command = /home/<user>/rag-pipeline/scripts/start_redis.sh
#
# (Or run once manually; the script is idempotent — it no-ops if Redis is up.)

set -u

log() { echo "[redis-start] $*"; }

if command -v redis-cli >/dev/null 2>&1 && redis-cli ping >/dev/null 2>&1; then
  log "Redis already running — nothing to do."
  exit 0
fi

if ! command -v redis-server >/dev/null 2>&1; then
  log "redis-server not installed. Install with: sudo apt-get install -y redis-server"
  exit 1
fi

# Start daemonized, bound to loopback (the RAG server runs in the same WSL).
redis-server --daemonize yes --bind 127.0.0.1
sleep 1

if redis-cli ping >/dev/null 2>&1; then
  log "Redis started (PID $(pgrep -f redis-server | head -1))."
else
  log "Failed to start Redis."
  exit 1
fi
