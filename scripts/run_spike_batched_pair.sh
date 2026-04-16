#!/usr/bin/env bash
#
# Run the batched-pair kernel spike (#43) on the trnblas CI trn1 instance.
#
# Usage:
#   AWS_PROFILE=aws ./scripts/run_spike_batched_pair.sh [--spike A|B|C|all]
#
# Default: --spike all  (runs all three spike kernels in sequence).
#
# This script:
#   1. Starts the tagged instance (if stopped/stopping)
#   2. Waits for SSM agent
#   3. Sends the spike script via SSM (installs trnblas, runs spike)
#   4. Prints stdout/stderr
#   5. Stops the instance (always, even on failure)

set -euo pipefail

SPIKE="${1:-all}"
if [[ "$SPIKE" == "--spike" ]]; then
    SPIKE="${2:-all}"
fi

INSTANCE_TYPE="${INSTANCE_TYPE:-trn1}"
TAG="trnblas-ci-${INSTANCE_TYPE}"
REGION="${AWS_REGION:-us-east-1}"
SHA="$(git rev-parse HEAD)"

: "${AWS_PROFILE:?Set AWS_PROFILE, e.g. AWS_PROFILE=aws ./scripts/run_spike_batched_pair.sh}"

echo "Looking up instance with Name=$TAG in $REGION..."
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=$TAG" \
            "Name=instance-state-name,Values=stopped,stopping,running,pending" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text \
  --region "$REGION")

if [[ -z "$INSTANCE_ID" || "$INSTANCE_ID" == "None" ]]; then
  echo "ERROR: No instance found with Name=$TAG" >&2
  echo "Provision with: cd infra/terraform && terraform apply" >&2
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
  echo "ERROR: SSM agent not Online after 5 minutes (last PingStatus=$PING)" >&2
  exit 1
fi

echo "Sending spike command (SHA=$SHA, spike=$SPIKE)..."
CMD_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --comment "trnblas spike_batched_pair @ $SHA" \
  --parameters "commands=[
    \"bash -c 'set -euo pipefail; cd /home/ubuntu/trnblas && sudo -u ubuntu git fetch --all && sudo -u ubuntu git checkout $SHA && NEURON_VENV=\$(ls -d /opt/aws_neuronx_venv_pytorch_* | head -1) && sudo -u ubuntu \$NEURON_VENV/bin/pip install -e /home/ubuntu/trnblas[dev] --quiet && sudo -u ubuntu env PATH=\$NEURON_VENV/bin:/usr/bin:/bin TRNBLAS_REQUIRE_NKI=1 \$NEURON_VENV/bin/python /home/ubuntu/trnblas/scripts/spike_batched_pair.py --spike $SPIKE'\"
  ]" \
  --region "$REGION" \
  --output text --query 'Command.CommandId')

echo "Command ID: $CMD_ID"
echo "Waiting for spike to complete (spikes A+B each compile 1-2 NEFFs; ~5-10 min)..."

# Poll every 30s for up to 20 min.
for _ in $(seq 1 40); do
  STATUS=$(aws ssm get-command-invocation \
    --command-id "$CMD_ID" \
    --instance-id "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Status' --output text 2>/dev/null || echo Pending)
  case "$STATUS" in
    Success|Failed|Cancelled|TimedOut|DeliveryTimedOut|ExecutionTimedOut)
      break ;;
  esac
  echo "  Status: $STATUS — waiting..."
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
