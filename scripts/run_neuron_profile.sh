#!/usr/bin/env bash
#
# Capture a Neuron profiler trace of the df_mp2 --fused-energy bench
# on the trnblas CI instance via SSM.
#
# Usage:
#   AWS_PROFILE=aws ./scripts/run_neuron_profile.sh                # medium
#   AWS_PROFILE=aws ./scripts/run_neuron_profile.sh --shape medium
#
# ## Status (2026-04-14, #33)
#
# Capture works — `neuron-profile inspect -o <dir> <userscript>` dumps
# `ntrace.pb` + `trace_info.pb` on the instance. Extraction to anything
# human-readable is blocked on the Neuron 2.29 DLAMI:
#
#   - `neuron-profile view --disable-ui --ingest-only` requires InfluxDB
#     to be set up as the timeseries store. The DLAMI doesn't pre-install
#     it.
#   - `neuron-profile show-session` rejects the trace-format version the
#     capturer produces (v130 — tool only supports v1–6 in the same
#     2.29.18.0-d5fe7ba42 release).
#
# This script leaves the capture on disk under /home/ubuntu/profiles/
# for later analysis once either InfluxDB is available or a newer
# aws-neuronx-tools release lands. See
# docs/design/mp2_energy_profile_findings.md for the full write-up.

set -euo pipefail

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
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
BENCH_ARGS="${EXTRA_ARGS[*]:---shape medium}"

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

BODY="printf %s\\\\n ==CAPTURE== && mkdir -p /home/ubuntu/profiles && chown -R ubuntu:ubuntu /home/ubuntu/profiles && PROFILE_DIR=/home/ubuntu/profiles/run-\$(date +%s) && sudo -u ubuntu mkdir -p \$PROFILE_DIR && sudo -u ubuntu env PATH=\$NEURON_VENV/bin:/opt/aws/neuron/bin:/usr/bin:/bin $NP inspect -o \$PROFILE_DIR \$NEURON_VENV/bin/python /home/ubuntu/trnblas/examples/df_mp2.py --bench --fused-energy $BENCH_ARGS 2>&1 | tail -40 && printf %s\\\\n ==ARTIFACTS== && ls -laR \$PROFILE_DIR 2>&1 | head -30 && printf %s\\\\n ==NOTE== && echo 'Raw ntrace.pb captured. Extraction via neuron-profile view requires InfluxDB setup on the instance (not pre-installed).'"

echo "Sending profile command (SHA=$SHA, args=$BENCH_ARGS)..."
CMD_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --comment "trnblas neuron-profile @ $SHA" \
  --parameters "commands=[
    \"bash -c 'set -euo pipefail; cd /home/ubuntu/trnblas && sudo -u ubuntu git fetch --all && sudo -u ubuntu git checkout $SHA && NEURON_VENV=\$(ls -d /opt/aws_neuronx_venv_pytorch_* | head -1) && sudo -u ubuntu \$NEURON_VENV/bin/pip install -e /home/ubuntu/trnblas[dev] --quiet && $BODY'\"
  ]" \
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
