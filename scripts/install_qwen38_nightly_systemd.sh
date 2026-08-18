#!/usr/bin/env bash
set -euo pipefail

# Install a per-user systemd timer that starts the resumable Qwen3.8 workflow at
# local midnight every day. qwen38_nightly_runner.sh independently enforces the
# 07:00 local wall-clock deadline; RuntimeMaxSec and KillMode are defense in
# depth so every quantizer child is stopped with the service.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/gptq-pro"
SERVICE="$UNIT_DIR/gptq-pro-qwen38-nightly.service"
TIMER="$UNIT_DIR/gptq-pro-qwen38-nightly.timer"
ENV_FILE="$CONFIG_DIR/qwen38-nightly.env"

mkdir -p "$UNIT_DIR" "$CONFIG_DIR"

if [[ ! -x "$REPO_ROOT/.venv/bin/python" ]]; then
  echo "Expected GPTQ-Pro virtualenv at $REPO_ROOT/.venv" >&2
  echo "Create it and install the repository before installing the timer." >&2
  exit 2
fi

if [[ ! -f "$ENV_FILE" ]]; then
  cat > "$ENV_FILE" <<EOF
# systemd EnvironmentFile syntax. Edit these paths before the first midnight.
MODEL=Qwen/Qwen3.8-27B
CALIBRATION_JSONL=/data/qwen38-calibration.jsonl
WORKDIR=/models/qwen38-gptq-pro-resume
FINAL_OUT=/models/Qwen3.8-27B-GPTQ-Pro-INT4-g64-longctx
GPU_LIST=0,1,2
NSAMPLE=128
GROUP_SIZE=64
BENCH_GPU=0
BENCH_CONTEXTS=2048,8192,32768
BENCH_NEW_TOKENS=128
STOP_GRACE_SECONDS=120
MIN_RUN_SECONDS=300
EOF
  chmod 600 "$ENV_FILE"
  echo "[install] created $ENV_FILE"
else
  echo "[install] preserving existing $ENV_FILE"
fi

cat > "$SERVICE" <<EOF
[Unit]
Description=GPTQ-Pro Qwen3.8 nightly resumable quantization
Documentation=https://github.com/groxaxo/GPTQ-Pro
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory=$REPO_ROOT
EnvironmentFile=$ENV_FILE
Environment=PATH=$REPO_ROOT/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/usr/bin/env bash $REPO_ROOT/scripts/qwen38_nightly_runner.sh
# The runner computes the exact local 07:00 deadline. These settings make sure
# an unexpected runner failure cannot leave CUDA children alive indefinitely.
RuntimeMaxSec=7h
TimeoutStopSec=3min
KillSignal=SIGTERM
KillMode=control-group

[Install]
WantedBy=default.target
EOF

cat > "$TIMER" <<'EOF'
[Unit]
Description=Start GPTQ-Pro Qwen3.8 quantization at midnight

[Timer]
OnCalendar=*-*-* 00:00:00
Persistent=true
AccuracySec=1s
Unit=gptq-pro-qwen38-nightly.service

[Install]
WantedBy=timers.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now gptq-pro-qwen38-nightly.timer

echo "[install] enabled user timer"
systemctl --user list-timers gptq-pro-qwen38-nightly.timer --no-pager || true

# A user timer must have a user manager alive while logged out. Enable linger
# automatically when passwordless sudo is available; otherwise print the one
# privileged command required. The timer itself never runs as root.
if loginctl show-user "$USER" -p Linger --value 2>/dev/null | grep -qx yes; then
  echo "[install] linger already enabled for $USER"
elif sudo -n true 2>/dev/null; then
  sudo loginctl enable-linger "$USER"
  echo "[install] enabled linger for $USER"
else
  echo
  echo "[action required once] keep the user timer alive while logged out/rebooted:"
  echo "  sudo loginctl enable-linger $USER"
fi

echo
echo "Configuration: $ENV_FILE"
echo "Timer:         $TIMER"
echo "Service:       $SERVICE"
echo
echo "Useful commands:"
echo "  systemctl --user status gptq-pro-qwen38-nightly.timer"
echo "  systemctl --user start gptq-pro-qwen38-nightly.service   # manual window-respecting test"
echo "  journalctl --user -u gptq-pro-qwen38-nightly.service -f"
echo "  systemctl --user disable --now gptq-pro-qwen38-nightly.timer"
