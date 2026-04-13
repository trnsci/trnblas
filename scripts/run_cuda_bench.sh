#!/usr/bin/env bash
#
# Run the DF-MP2 bench on the trnblas CUDA CI instance (A10G) via SSM.
# Vintage-matched to trn1: see infra/terraform-cuda/README.md.
#
# Usage:
#   AWS_PROFILE=aws ./scripts/run_cuda_bench.sh                 # all 3 shapes
#   AWS_PROFILE=aws ./scripts/run_cuda_bench.sh --shape medium  # one shape
#
# Mirrors run_df_mp2_bench.sh: starts the tagged instance, runs the
# bench with --device cuda, prints stdout/stderr, always stops the
# instance via trap.

set -euo pipefail

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --shape)
      EXTRA_ARGS+=("--shape" "$2")
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

TAG="${TRNBLAS_CUDA_TAG:-trnblas-ci-cuda-a10g}"
REGION="${AWS_REGION:-us-east-1}"
SHA="$(git rev-parse HEAD)"
BENCH_ARGS="${EXTRA_ARGS[*]:-}"

: "${AWS_PROFILE:?Set AWS_PROFILE, e.g. AWS_PROFILE=aws ./scripts/run_cuda_bench.sh}"

echo "Looking up instance with Name=$TAG in $REGION..."
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=$TAG" \
            "Name=instance-state-name,Values=stopped,running,pending" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text \
  --region "$REGION")

if [[ -z "$INSTANCE_ID" || "$INSTANCE_ID" == "None" ]]; then
  echo "ERROR: No instance found with Name=$TAG" >&2
  echo "       Provision it first: see infra/terraform-cuda/README.md" >&2
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

echo "Sending bench command (SHA=$SHA, args=$BENCH_ARGS)..."
CMD_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --comment "trnblas cuda bench @ $SHA" \
  --parameters "commands=[
    \"bash -c 'set -euo pipefail; cd /home/ubuntu/trnblas && sudo -u ubuntu git fetch --all && sudo -u ubuntu git checkout $SHA && sudo -u ubuntu /opt/pytorch/bin/pip install -e /home/ubuntu/trnblas[dev] --quiet && sudo -u ubuntu /opt/pytorch/bin/python /home/ubuntu/trnblas/examples/df_mp2.py --bench --device cuda $BENCH_ARGS'\"
  ]" \
  --region "$REGION" \
  --output text --query 'Command.CommandId')

echo "Command ID: $CMD_ID"
echo "Waiting for command to complete (poll every 30s, up to 30min)..."

STATUS=InProgress
for _ in $(seq 1 60); do
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
