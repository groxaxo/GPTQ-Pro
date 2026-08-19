#!/usr/bin/env bash
set -euo pipefail

# Install user-level systemd start/stop timers for the resumable Qwen3.8 job.
# The start timer fires at 00:00 in the configured IANA timezone. The runner and
# an independent stop timer both enforce the configured morning boundary.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/gptq-pro"
SERVICE="$UNIT_DIR/gptq-pro-qwen38-nightly.service"
START_TIMER="$UNIT_DIR/gptq-pro-qwen38-nightly.timer"
STOP_SERVICE="$UNIT_DIR/gptq-pro-qwen38-nightly-stop.service"
STOP_TIMER="$UNIT_DIR/gptq-pro-qwen38-nightly-stop.timer"
ENV_FILE="$CONFIG_DIR/qwen38-nightly.env"

mkdir -p "$UNIT_DIR" "$CONFIG_DIR"

if [[ ! -x "$REPO_ROOT/.venv/bin/python" ]]; then
  echo "Expected GPTQ-Pro virtualenv at $REPO_ROOT/.venv" >&2
  echo "Create it and install the repository before installing the timers." >&2
  exit 2
fi

if [[ ! -f "$ENV_FILE" ]]; then
  cat > "$ENV_FILE" <<'EOF'
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
NIGHTLY_TIMEZONE=Pacific/Auckland
NIGHTLY_STOP_TIME=07:00:00
EOF
  echo "[install] created $ENV_FILE"
else
  echo "[install] preserving existing $ENV_FILE"
fi
chmod 600 "$ENV_FILE"

ensure_env_default() {
  local key="$1"
  local value="$2"
  if ! grep -Eq "^[[:space:]]*${key}=" "$ENV_FILE"; then
    printf '%s=%s\n' "$key" "$value" >> "$ENV_FILE"
    echo "[install] added default $key=$value"
  fi
}

# Upgrade older installations without replacing any user-edited paths/settings.
ensure_env_default NIGHTLY_TIMEZONE Pacific/Auckland
ensure_env_default NIGHTLY_STOP_TIME 07:00:00

read -r SCHEDULE_TIMEZONE SCHEDULE_STOP_TIME < <(
  "$REPO_ROOT/.venv/bin/python" - "$ENV_FILE" <<'PY'
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

path = Path(sys.argv[1])
values = {}
for raw in path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    values[key.strip()] = value

timezone_name = values.get("NIGHTLY_TIMEZONE", "Pacific/Auckland")
stop_value = values.get("NIGHTLY_STOP_TIME", "07:00:00")
try:
    ZoneInfo(timezone_name)
except ZoneInfoNotFoundError as exc:
    raise SystemExit(f"invalid NIGHTLY_TIMEZONE={timezone_name!r}") from exc

parsed = None
for fmt in ("%H:%M:%S", "%H:%M"):
    try:
        parsed = datetime.strptime(stop_value, fmt)
        break
    except ValueError:
        pass
if parsed is None:
    raise SystemExit(f"invalid NIGHTLY_STOP_TIME={stop_value!r}; use HH:MM or HH:MM:SS")

print(timezone_name, parsed.strftime("%H:%M:%S"))
PY
)

echo "[install] schedule timezone: $SCHEDULE_TIMEZONE"
echo "[install] nightly window:   00:00:00-$SCHEDULE_STOP_TIME"

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
# The runner and 07:00 stop timer enforce the exact local wall-clock boundary.
# Nine hours permits the eight-hour elapsed window on Auckland's DST fall-back
# night while still bounding any unexpected runner failure.
RuntimeMaxSec=9h
TimeoutStopSec=3min
KillSignal=SIGTERM
FinalKillSignal=SIGKILL
KillMode=control-group
UMask=0077

[Install]
WantedBy=default.target
EOF

cat > "$START_TIMER" <<EOF
[Unit]
Description=Start GPTQ-Pro Qwen3.8 quantization at midnight ($SCHEDULE_TIMEZONE)

[Timer]
OnCalendar=*-*-* 00:00:00 $SCHEDULE_TIMEZONE
Persistent=true
AccuracySec=1s
Unit=gptq-pro-qwen38-nightly.service

[Install]
WantedBy=timers.target
EOF

cat > "$STOP_SERVICE" <<'EOF'
[Unit]
Description=Stop GPTQ-Pro Qwen3.8 nightly quantization at the morning boundary

[Service]
Type=oneshot
ExecStart=/usr/bin/systemctl --user stop gptq-pro-qwen38-nightly.service
EOF

cat > "$STOP_TIMER" <<EOF
[Unit]
Description=Stop GPTQ-Pro Qwen3.8 quantization at $SCHEDULE_STOP_TIME ($SCHEDULE_TIMEZONE)

[Timer]
OnCalendar=*-*-* $SCHEDULE_STOP_TIME $SCHEDULE_TIMEZONE
Persistent=true
AccuracySec=1s
Unit=gptq-pro-qwen38-nightly-stop.service

[Install]
WantedBy=timers.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now \
  gptq-pro-qwen38-nightly.timer \
  gptq-pro-qwen38-nightly-stop.timer

echo "[install] enabled start and stop timers"
systemctl --user list-timers \
  gptq-pro-qwen38-nightly.timer \
  gptq-pro-qwen38-nightly-stop.timer \
  --no-pager || true

# User timers require a persistent user manager while logged out. Enable linger
# automatically with passwordless sudo; otherwise print the one privileged step.
if loginctl show-user "$USER" -p Linger --value 2>/dev/null | grep -qx yes; then
  echo "[install] linger already enabled for $USER"
elif sudo -n true 2>/dev/null; then
  sudo loginctl enable-linger "$USER"
  echo "[install] enabled linger for $USER"
else
  echo
  echo "[action required once] keep the user timers alive while logged out/rebooted:"
  echo "  sudo loginctl enable-linger $USER"
fi

echo
echo "Configuration: $ENV_FILE"
echo "Start timer:   $START_TIMER"
echo "Stop timer:    $STOP_TIMER"
echo "Service:       $SERVICE"
echo
echo "Useful commands:"
echo "  systemctl --user status gptq-pro-qwen38-nightly.timer gptq-pro-qwen38-nightly-stop.timer"
echo "  systemctl --user start gptq-pro-qwen38-nightly.service   # manual window-respecting test"
echo "  systemctl --user stop gptq-pro-qwen38-nightly.service"
echo "  journalctl --user -u gptq-pro-qwen38-nightly.service -f"
echo "  systemctl --user disable --now gptq-pro-qwen38-nightly.timer gptq-pro-qwen38-nightly-stop.timer"
