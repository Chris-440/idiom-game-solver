#!/bin/bash
# Auto-restarting TensorBoard daemon
LOG=/root/成语接龙/logs/tensorboard.log
PORT=6006
LOGDIR=/root/成语接龙/checkpoints/tensorboard
mkdir -p "$(dirname "$LOG")" "$LOGDIR"

# Kill existing only if port in use
if ss -tlnp | grep -q ":$PORT "; then
    echo "TensorBoard already on port $PORT"
    exit 0
fi

nohup tensorboard --logdir="$LOGDIR" --port="$PORT" --bind_all \
    --reload_multifile=true > "$LOG" 2>&1 &
echo "TensorBoard started on port $PORT (PID $!)"
