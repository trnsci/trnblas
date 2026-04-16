#!/usr/bin/env bash
#
# Capture a Neuron profiler trace of _mp2_energy_kernel on the trnblas CI
# instance via SSM, using Neuron Profiler 2.0 (Neuron 2.29 new API).
#
# Usage:
#   AWS_PROFILE=aws ./scripts/run_neuron_profile.sh --probe   # Phase A: tool discovery
#   AWS_PROFILE=aws ./scripts/run_neuron_profile.sh           # Phase B/C: full capture (medium)
#   AWS_PROFILE=aws ./scripts/run_neuron_profile.sh --shape medium
#
# ## API History
#
# 2026-04-14 (#33 attempt 1):  BLOCKED on Neuron 2.29 DLAMI —
#   - neuron-profile inspect → ntrace.pb (NTFF v130) rejected by show-session
#   - neuron-profile view --disable-ui --ingest-only requires InfluxDB (not installed)
#
# 2026-04-25 (#33 attempt 2, this script):
#   Phase A probe confirmed the new Neuron Profiler 2.0 API in Neuron 2.29.18.0:
#     --output-format=[db|summary-text|summary-json|json|perfetto|parquet]
#   capture --io-from=neff (default) allocates IO tensors from NEFF declarations —
#   no separate input files needed. Old April-14 traces are ntrace.pb (inspect
#   format), incompatible with the new view command — re-capture required.
#
# Strategy:
#   1. Write a minimal Python script that compiles ONLY _mp2_energy_kernel to NEFF
#      (clears compile cache first for isolation).
#   2. neuron-profile capture -n <neff> -s <ntff>  — executes NEFF, records trace.
#   3. neuron-profile view   -n <neff> -s <ntff> --output-format summary-text
#      → per-engine utilization + top ops, returned in SSM stdout (answers #33 B.1-B.4).
#
# Both probe and capture bodies are base64-encoded locally before being sent to
# SSM to bypass all shell quoting / heredoc-in-pipe-to-bash issues.
#
# Note: neuron-profile requires HOME to be set. Commands run as
# `sudo -u ubuntu HOME=/home/ubuntu` to ensure proper user context.

set -euo pipefail

EXTRA_ARGS=()
PROBE=false
INSTANCE_TYPE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --probe)
      PROBE=true
      shift
      ;;
    --shape)
      EXTRA_ARGS+=("--shape" "$2")
      shift 2
      ;;
    trn1|trn2|inf2)
      INSTANCE_TYPE="$1"
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

INSTANCE_TYPE="${INSTANCE_TYPE:-trn1}"
TAG="trnblas-ci-${INSTANCE_TYPE}"
REGION="${AWS_REGION:-us-east-1}"
SHA="$(git rev-parse HEAD)"

: "${AWS_PROFILE:?Set AWS_PROFILE, e.g. AWS_PROFILE=aws ./scripts/run_neuron_profile.sh}"

NP="/opt/aws/neuron/bin/neuron-profile"

echo "Looking up instance with Name=$TAG in $REGION..."
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=$TAG" \
            "Name=instance-state-name,Values=stopped,stopping,running,pending" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text \
  --region "$REGION")

if [[ -z "$INSTANCE_ID" || "$INSTANCE_ID" == "None" ]]; then
  echo "ERROR: No instance found with Name=$TAG" >&2
  exit 1
fi
echo "Instance: $INSTANCE_ID"

cleanup() {
  local exit_code=$?
  echo ""
  echo "Stopping $INSTANCE_ID..."
  aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
  exit "$exit_code"
}
trap cleanup EXIT

STATE=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$REGION" \
  --query 'Reservations[0].Instances[0].State.Name' --output text)

if [[ "$STATE" == "stopping" ]]; then
  echo "Instance is stopping — waiting for it to reach stopped..."
  aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID" --region "$REGION"
  STATE=stopped
fi

if [[ "$STATE" == "stopped" ]]; then
  echo "Starting instance..."
  aws ec2 start-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
fi

echo "Waiting for instance-running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"
echo "Waiting for SSM agent..."
for _ in $(seq 1 60); do
  PING=$(aws ssm describe-instance-information \
    --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
    --region "$REGION" \
    --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || true)
  [[ "$PING" == "Online" ]] && break
  sleep 5
done
if [[ "$PING" != "Online" ]]; then
  echo "ERROR: SSM agent not Online after 5 minutes (last PingStatus=$PING)" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Phase A — probe: discover API surface, check for old traces and cached NEFFs
# ---------------------------------------------------------------------------
if [[ "$PROBE" == "true" ]]; then
  echo "Running Phase A probe (SHA=$SHA)..."
  PROBE_SCRIPT=$(cat <<'PROBE_EOF'
set -euo pipefail
NP=/opt/aws/neuron/bin/neuron-profile
printf '%s\n' ==NP_HELP==
$NP --help 2>&1 || true
printf '%s\n' ==NP_CAPTURE_HELP==
$NP capture --help 2>&1 || true
printf '%s\n' ==VIEW_OUTPUT_FORMAT==
$NP view --help 2>&1 | grep -i "output.format" || echo "NOT FOUND"
printf '%s\n' ==OLD_TRACES==
find /home/ubuntu/profiles -type f 2>/dev/null | head -20 || echo none
printf '%s\n' ==NEFF_CACHE==
find /var/tmp/neuron-compile-cache -name model.neff 2>/dev/null | head -10 || echo none
printf '%s\n' ==NEURON_VERSION==
$NP --version 2>&1 || true
PROBE_EOF
)
  B64=$(printf '%s' "$PROBE_SCRIPT" | base64 | tr -d '\n')

  CMD_ID=$(aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --comment "trnblas neuron-profile probe @ $SHA" \
    --parameters "commands=[\"printf '%s' $B64 | base64 -d | bash\"]" \
    --region "$REGION" \
    --output text --query 'Command.CommandId')

  echo "Command ID: $CMD_ID"
  echo "Waiting for probe to complete (poll every 10s, up to 5min)..."

  STATUS=InProgress
  for _ in $(seq 1 30); do
    STATUS=$(aws ssm get-command-invocation \
      --command-id "$CMD_ID" \
      --instance-id "$INSTANCE_ID" \
      --region "$REGION" \
      --query 'Status' --output text 2>/dev/null || echo "InProgress")
    [[ "$STATUS" != "InProgress" && "$STATUS" != "Pending" ]] && break
    sleep 10
  done

  echo ""
  echo "=== PROBE STDOUT ==="
  aws ssm get-command-invocation \
    --command-id "$CMD_ID" \
    --instance-id "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'StandardOutputContent' --output text

  echo ""
  echo "=== PROBE STDERR ==="
  aws ssm get-command-invocation \
    --command-id "$CMD_ID" \
    --instance-id "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'StandardErrorContent' --output text

  echo ""
  echo "=== Status: $STATUS ==="
  [[ "$STATUS" == "Success" ]]
  exit 0
fi

# ---------------------------------------------------------------------------
# Phase B/C — full capture using Neuron Profiler 2.0
#
# Double-base64 encoding strategy:
#   1. Python warmup script → PY_B64 (first encoding)
#   2. Bash capture body with PY_B64 embedded → B64 (second encoding)
#   3. SSM sends: printf '%s' B64 | base64 -d | bash
#   4. Bash body contains: printf '%s' PY_B64 | base64 -d > /tmp/mp2_warmup.py
# This avoids heredoc-in-pipe-to-bash issues and all shell quoting problems.
# ---------------------------------------------------------------------------
echo "Building capture command (SHA=$SHA)..."

# Python warmup: compiles ONLY _mp2_energy_kernel at medium bench shape.
# ic=nocc=64, nvir=448 matches _BENCH_SHAPES["medium"] (nbasis=512, nocc=64).
# (The actual bench uses i_block=29 for memory; this compiles the kernel once
# in the ic=nocc=64 all-pairs form for a clean isolated NEFF.)
PY_WARMUP=$(cat <<'PYEOF'
import sys
sys.path.insert(0, '/home/ubuntu/trnblas')
import torch
import trnblas
trnblas.set_backend('nki')

ic, nocc, nvir = 64, 64, 448

T_flat        = torch.zeros(ic * nvir, nocc * nvir)
eps_occ_chunk = torch.full((ic,), -0.5)
eps_occ_full  = torch.full((nocc,), -0.5)
eps_vir       = torch.full((nvir,), 0.5)

print(f"Compiling _mp2_energy_kernel: T_flat=({ic*nvir},{nocc*nvir})", flush=True)
result = trnblas.nki.nki_mp2_energy(T_flat, eps_occ_chunk, eps_occ_full, eps_vir)
print(f"Done. result={float(result):.6e}", flush=True)
PYEOF
)
PY_B64=$(printf '%s' "$PY_WARMUP" | base64 | tr -d '\n')

# Bash capture body (placeholders __SHA__ and __PY_B64__ substituted below)
CAPTURE_BODY=$(cat <<'CAPTURE_EOF'
set -euo pipefail
NP=/opt/aws/neuron/bin/neuron-profile
NEURON_VENV=$(ls -d /opt/aws_neuronx_venv_pytorch_* 2>/dev/null | head -1)
test -n "$NEURON_VENV" || { echo "ERROR: no Neuron venv" >&2; exit 1; }
PYTHON="$NEURON_VENV/bin/python"

cd /home/ubuntu
sudo -u ubuntu git -C /home/ubuntu/trnblas fetch --all --quiet
sudo -u ubuntu git -C /home/ubuntu/trnblas checkout __SHA__
sudo -u ubuntu env PATH="$NEURON_VENV/bin:/usr/bin:/bin" \
  "$PYTHON" -m pip install -e /home/ubuntu/trnblas[dev] --quiet

PROFILE_DIR=/home/ubuntu/profiles/run-$(date +%s)
sudo -u ubuntu mkdir -p "$PROFILE_DIR"
chown -R ubuntu:ubuntu /home/ubuntu/profiles

printf '%s\n' ==STEP1_WRITE_WARMUP==
printf '%s' __PY_B64__ | base64 -d > /tmp/mp2_warmup.py
chown ubuntu:ubuntu /tmp/mp2_warmup.py
echo "Warmup script written."

printf '%s\n' ==STEP2_CLEAR_CACHE_AND_COMPILE==
rm -rf /var/tmp/neuron-compile-cache/* 2>/dev/null || true
echo "Compile cache cleared."
sudo -u ubuntu env \
  PATH="$NEURON_VENV/bin:/opt/aws/neuron/bin:/usr/bin:/bin" \
  "$PYTHON" /tmp/mp2_warmup.py 2>&1

printf '%s\n' ==STEP3_FIND_NEFF==
NEFF=$(find /var/tmp/neuron-compile-cache -name model.neff 2>/dev/null | head -1)
test -n "$NEFF" || { echo "ERROR: no model.neff after warmup" >&2; exit 1; }
echo "NEFF: $NEFF"
ls -lah "$NEFF"

printf '%s\n' ==STEP4_CAPTURE==
# neuron-profile requires HOME; run as ubuntu with explicit HOME.
sudo -u ubuntu HOME=/home/ubuntu "$NP" capture \
  -n "$NEFF" -s "$PROFILE_DIR/profile.ntff" 2>&1

printf '%s\n' ==STEP5_SUMMARY_TEXT==
sudo -u ubuntu HOME=/home/ubuntu "$NP" view \
  -n "$NEFF" -s "$PROFILE_DIR/profile.ntff" \
  --output-format summary-text 2>&1

printf '%s\n' ==STEP6_SUMMARY_JSON==
sudo -u ubuntu HOME=/home/ubuntu "$NP" view \
  -n "$NEFF" -s "$PROFILE_DIR/profile.ntff" \
  --output-format summary-json 2>&1 | head -300

printf '%s\n' ==ARTIFACTS==
ls -laR "$PROFILE_DIR" 2>&1 | head -40
CAPTURE_EOF
)

# Substitute SHA and Python base64 into the body
CAPTURE_BODY="${CAPTURE_BODY//__SHA__/$SHA}"
CAPTURE_BODY="${CAPTURE_BODY//__PY_B64__/$PY_B64}"

B64=$(printf '%s' "$CAPTURE_BODY" | base64 | tr -d '\n')

echo "Sending capture command (SHA=$SHA)..."
CMD_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --comment "trnblas neuron-profile 2.0 @ $SHA" \
  --parameters "commands=[\"printf '%s' $B64 | base64 -d | bash\"]" \
  --region "$REGION" \
  --output text --query 'Command.CommandId')

echo "Command ID: $CMD_ID"
echo "Waiting for command to complete (poll every 30s, up to 60min)..."

STATUS=InProgress
for _ in $(seq 1 120); do
  STATUS=$(aws ssm get-command-invocation \
    --command-id "$CMD_ID" \
    --instance-id "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Status' --output text 2>/dev/null || echo "InProgress")
  [[ "$STATUS" != "InProgress" && "$STATUS" != "Pending" ]] && break
  sleep 30
done

echo ""
echo "=== STDOUT ==="
aws ssm get-command-invocation \
  --command-id "$CMD_ID" \
  --instance-id "$INSTANCE_ID" \
  --region "$REGION" \
  --query 'StandardOutputContent' --output text

echo ""
echo "=== STDERR ==="
aws ssm get-command-invocation \
  --command-id "$CMD_ID" \
  --instance-id "$INSTANCE_ID" \
  --region "$REGION" \
  --query 'StandardErrorContent' --output text

echo ""
echo "=== Status: $STATUS ==="

[[ "$STATUS" == "Success" ]]
