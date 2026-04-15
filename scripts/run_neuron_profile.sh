#!/usr/bin/env bash
#
# Capture a Neuron profiler trace of the df_mp2 --fused-energy bench
# on the trnblas CI instance via SSM. Used to diagnose where the
# ~30 s/chunk wall time goes inside _mp2_energy_kernel (#33).
#
# Usage:
#   AWS_PROFILE=aws ./scripts/run_neuron_profile.sh                # medium
#   AWS_PROFILE=aws ./scripts/run_neuron_profile.sh --shape medium
#   AWS_PROFILE=aws ./scripts/run_neuron_profile.sh --probe        # print
#                                                                  # neuron-profile --help
#                                                                  # then exit
#
# Leaves raw `.ntff` on the instance under /home/ubuntu/profiles/
# and retrieves the text-format `neuron-profile show` summary via
# SSM stdout.

set -euo pipefail

PROBE=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --shape)
      EXTRA_ARGS+=("--shape" "$2")
      shift 2
      ;;
    --probe)
      PROBE=1
      shift
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

# Body command depends on --probe. In either case we always print
# `neuron-profile --help` at the top so the log pin-points CLI
# surface differences before any capture is attempted.
NP="/opt/aws/neuron/bin/neuron-profile"

if [[ "$PROBE" -eq 1 ]]; then
  BODY="printf %s\\\\n ==NEURON-PROFILE-HELP== && $NP --help 2>&1 && printf %s\\\\n ==CAPTURE-HELP== && $NP capture --help 2>&1 && printf %s\\\\n ==VIEW-HELP== && $NP view --help 2>&1 || true"
else
  BODY="printf %s\\\\n ==CAPTURE== && mkdir -p /home/ubuntu/profiles && chown ubuntu:ubuntu /home/ubuntu/profiles && NAME=trnblas-mp2-\$(date +%s) && sudo -u ubuntu env PATH=\$NEURON_VENV/bin:/opt/aws/neuron/bin:/usr/bin:/bin $NP capture -n \$NAME -s \"\$NEURON_VENV/bin/python /home/ubuntu/trnblas/examples/df_mp2.py --bench --fused-energy $BENCH_ARGS\" -o /home/ubuntu/profiles 2>&1 | tail -40 && printf %s\\\\n ==SHOW== && ls -la /home/ubuntu/profiles/ && $NP view /home/ubuntu/profiles/*.ntff 2>&1 | head -200 || true"
fi

echo "Sending profile command (SHA=$SHA, probe=$PROBE, args=$BENCH_ARGS)..."
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
