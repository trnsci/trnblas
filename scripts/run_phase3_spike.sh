#!/usr/bin/env bash
#
# Run Phase 3 fused GEMM+energy spike (#38) on the trnblas CI instance via SSM.
#
# Usage:
#   AWS_PROFILE=aws ./scripts/run_phase3_spike.sh              # all three spikes
#   AWS_PROFILE=aws ./scripts/run_phase3_spike.sh --profile-spike-c  # + Neuron Profiler on spike C
#   AWS_PROFILE=aws ./scripts/run_phase3_spike.sh trn2         # explicit instance type
#
# Spike questions answered:
#   A: PSUM → SBUF → VE chain in one @nki.jit (no HBM intermediate for T_flat)
#   B: Two-GEMM strategy — T and T_T both SBUF-resident, no nl.load_transpose2d from HBM
#   C: TE/VE interleaved loop (correctness); --profile-spike-c adds Neuron Profiler capture
#      for TE/VE concurrency evidence (Perfetto trace + summary-json engine utilisation)
#
# Both the spike script and the profiler warmup are double-base64-encoded before SSM
# transmission to avoid heredoc-in-pipe-to-bash issues (see run_neuron_profile.sh header).

set -euo pipefail

PROFILE_SPIKE_C=false
INSTANCE_TYPE="trn1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile-spike-c) PROFILE_SPIKE_C=true; shift ;;
    trn1|trn2|inf2)    INSTANCE_TYPE="$1"; shift ;;
    *) shift ;;
  esac
done

TAG="trnblas-ci-${INSTANCE_TYPE}"
REGION="${AWS_REGION:-us-east-1}"
SHA="$(git rev-parse HEAD)"
NP="/opt/aws/neuron/bin/neuron-profile"

: "${AWS_PROFILE:?Set AWS_PROFILE}"

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
  echo "Instance is stopping — waiting for stopped..."
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
  echo "ERROR: SSM agent not Online (last PingStatus=$PING)" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Encode the spike script (double-base64 to bypass heredoc-in-pipe-to-bash)
# ---------------------------------------------------------------------------
SPIKE_PY=$(cat scripts/spike_phase3_fused_gemm_energy.py)
SPIKE_B64=$(printf '%s' "$SPIKE_PY" | base64 | tr -d '\n')

# ---------------------------------------------------------------------------
# Capture body
# ---------------------------------------------------------------------------
if [[ "$PROFILE_SPIKE_C" == "true" ]]; then
  PROFILE_EXTRA=$(cat <<'PROFILE_EOF'
printf '%s\n' ==STEP_PROFILE_SPIKE_C==
rm -rf /var/tmp/neuron-compile-cache/* 2>/dev/null || true

# Warmup script for spike C (compiles only _spike_c_te_ve_overlap).
WARMUP_PY=$(printf '%s' 'import sys
sys.path.insert(0, "/home/ubuntu/trnblas")
sys.path.insert(0, ".")
import torch
import trnblas
from scripts.spike_phase3_fused_gemm_energy import _spike_c_te_ve_overlap
import torch_xla.core.xla_model as xm
device = xm.xla_device()
TILE, NPAIRS = 128, 8
B = torch.randn(NPAIRS, TILE, TILE).to(device)
D = torch.ones(TILE, TILE).to(device)
O = torch.zeros(NPAIRS, TILE, 1).to(device)
print("Compiling spike C...", flush=True)
_spike_c_te_ve_overlap(B, D, O)
print("Done.", flush=True)
')
printf '%s' "$WARMUP_PY" > /tmp/spike_c_warmup.py
chown ubuntu:ubuntu /tmp/spike_c_warmup.py
sudo -u ubuntu env \
  PATH="$NEURON_VENV/bin:/opt/aws/neuron/bin:/usr/bin:/bin" \
  "$PYTHON" /tmp/spike_c_warmup.py 2>&1

NEFF=$(find /var/tmp/neuron-compile-cache -name model.neff 2>/dev/null | head -1)
test -n "$NEFF" || { echo "ERROR: no model.neff after spike C compile" >&2; exit 1; }
echo "NEFF: $NEFF  ($(ls -lah "$NEFF" | awk '{print $5}'))"

PROFILE_DIR=/home/ubuntu/profiles/spike-c-$(date +%s)
sudo -u ubuntu mkdir -p "$PROFILE_DIR"
sudo -u ubuntu HOME=/home/ubuntu "$NP" capture -n "$NEFF" -s "$PROFILE_DIR/profile.ntff" 2>&1
printf '%s\n' ==SPIKE_C_SUMMARY_JSON==
sudo -u ubuntu HOME=/home/ubuntu "$NP" view \
  -n "$NEFF" -s "$PROFILE_DIR/profile.ntff" --output-format summary-json 2>&1
printf '%s\n' ==SPIKE_C_ARTIFACTS==
ls -laR "$PROFILE_DIR" 2>&1 | head -20
PROFILE_EOF
)
else
  PROFILE_EXTRA="printf '%s\n' ==SKIP_PROFILE_SPIKE_C=="
fi

CAPTURE_BODY=$(cat <<CAPTURE_EOF
set -euo pipefail
NEURON_VENV=\$(ls -d /opt/aws_neuronx_venv_pytorch_* 2>/dev/null | head -1)
test -n "\$NEURON_VENV" || { echo "ERROR: no Neuron venv" >&2; exit 1; }
PYTHON="\$NEURON_VENV/bin/python"

cd /home/ubuntu
sudo -u ubuntu git -C /home/ubuntu/trnblas fetch --all --quiet
sudo -u ubuntu git -C /home/ubuntu/trnblas checkout $SHA
sudo -u ubuntu env PATH="\$NEURON_VENV/bin:/usr/bin:/bin" \
  "\$PYTHON" -m pip install -e /home/ubuntu/trnblas[dev] --quiet

printf '%s' __SPIKE_B64__ | base64 -d > /tmp/spike_phase3.py
chown ubuntu:ubuntu /tmp/spike_phase3.py

printf '%s\n' ==STEP_RUN_SPIKES==
sudo -u ubuntu env \
  PATH="\$NEURON_VENV/bin:/opt/aws/neuron/bin:/usr/bin:/bin" \
  TRNBLAS_REQUIRE_NKI=1 \
  "\$PYTHON" /tmp/spike_phase3.py 2>&1

$PROFILE_EXTRA
CAPTURE_EOF
)

# Substitute SHA and spike base64.
CAPTURE_BODY="${CAPTURE_BODY//__SPIKE_B64__/$SPIKE_B64}"
B64=$(printf '%s' "$CAPTURE_BODY" | base64 | tr -d '\n')

echo "Sending spike command (SHA=$SHA, profile_spike_c=$PROFILE_SPIKE_C)..."
CMD_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --comment "trnblas phase3 spike @ $SHA" \
  --parameters "commands=[\"printf '%s' $B64 | base64 -d | bash\"]" \
  --region "$REGION" \
  --output text --query 'Command.CommandId')

echo "Command ID: $CMD_ID"
echo "Waiting (poll every 30s, up to 60min)..."

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
