#!/bin/sh

set -eu

SERVER_PID=""
MCP_PID=""

# Stops all child processes started by this entrypoint.
cleanup() {
  if [ -n "$MCP_PID" ]; then
    kill "$MCP_PID" 2>/dev/null || true
    wait "$MCP_PID" 2>/dev/null || true
    MCP_PID=""
  fi

  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}

trap cleanup EXIT INT TERM

# Start DIP Studio HTTP and MCP servers as core services. If either exits,
# the container exits with the same status and cleanup stops the remaining
# child processes.
node dist/server.js &
SERVER_PID=$!

node dist/mcp-server.js &
MCP_PID=$!

while :; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    wait "$SERVER_PID" 2>/dev/null || SERVER_EXIT_CODE=$?
    SERVER_EXIT_CODE=${SERVER_EXIT_CODE:-1}
    exit "$SERVER_EXIT_CODE"
  fi

  if ! kill -0 "$MCP_PID" 2>/dev/null; then
    wait "$MCP_PID" 2>/dev/null || MCP_EXIT_CODE=$?
    MCP_EXIT_CODE=${MCP_EXIT_CODE:-1}
    exit "$MCP_EXIT_CODE"
  fi

  sleep 1
done
