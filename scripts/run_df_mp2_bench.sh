#!/usr/bin/env bash
#
# Run the DF-MP2 bench example on the trnblas CI instance via SSM.
#
# Usage:
#   AWS_PROFILE=aws ./scripts/run_df_mp2_bench.sh                 # all 3 shapes
#   AWS_PROFILE=aws ./scripts/run_df_mp2_bench.sh --shape medium  # one shape
#   AWS_PROFILE=aws ./scripts/run_df_mp2_bench.sh --compare       # torch vs --fused-energy, one session
#   AWS_PROFILE=aws ./scripts/run_df_mp2_bench.sh trn2            # different instance
#
# Mirrors run_neuron_tests.sh: starts the tagged instance, runs the
# bench, prints stdout/stderr, and stops the instance via trap.
# The bench itself runs each shape twice (cold / warm cache) inside one
# Python process — no need for a --warm flag at the script level.
#
# --compare runs the bench twice back-to-back in one SSM session: once
# with the torch energy path, once with --fused-energy. Avoids a second
# instance-start round-trip when doing A/B comparisons.

set -euo pipefail

COMPARE=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --shape)
      EXTRA_ARGS+=("--shape" "$2")
      shift 2
      ;;
    --compare)
      COMPARE=1
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
BENCH_ARGS="${EXTRA_ARGS[*]:-}"

: "${AWS_PROFILE:?Set AWS_PROFILE, e.g. AWS_PROFILE=aws ./scripts/run_df_mp2_bench.sh}"

echo "Looking up instance with Name=$TAG in $REGION..."
# Include 'stopping' so back-to-back runs don't race the previous run's
# cleanup trap. We'll wait for it to reach 'stopped' before starting.
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
  echo "Instance is stopping — waiting for it to reach stopped before starting..."
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

if [[ "$COMPARE" -eq 1 ]]; then
  # No single quotes — the whole SSM command is wrapped in bash -c '...',
  # so embedded single quotes would close the outer string and silently
  # truncate output. printf is unambiguous across quoting layers.
  BENCH_INVOCATION="printf %s\\\\n ==TORCH== && sudo -u ubuntu env PATH=\$NEURON_VENV/bin:/usr/bin:/bin \$NEURON_VENV/bin/python /home/ubuntu/trnblas/examples/df_mp2.py --bench $BENCH_ARGS && printf %s\\\\n ==FUSED== && sudo -u ubuntu env PATH=\$NEURON_VENV/bin:/usr/bin:/bin \$NEURON_VENV/bin/python /home/ubuntu/trnblas/examples/df_mp2.py --bench --fused-energy $BENCH_ARGS"
else
  BENCH_INVOCATION="sudo -u ubuntu env PATH=\$NEURON_VENV/bin:/usr/bin:/bin \$NEURON_VENV/bin/python /home/ubuntu/trnblas/examples/df_mp2.py --bench $BENCH_ARGS"
fi

echo "Sending bench command (SHA=$SHA, args=$BENCH_ARGS, compare=$COMPARE)..."
CMD_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --comment "trnblas df_mp2 bench @ $SHA" \
  --parameters "commands=[
    \"bash -c 'set -euo pipefail; cd /home/ubuntu/trnblas && sudo -u ubuntu git fetch --all && sudo -u ubuntu git checkout $SHA && NEURON_VENV=\$(ls -d /opt/aws_neuronx_venv_pytorch_* | head -1) && sudo -u ubuntu \$NEURON_VENV/bin/pip install -e /home/ubuntu/trnblas[dev] --quiet && $BENCH_INVOCATION'\"
  ]" \
  --region "$REGION" \
  --output text --query 'Command.CommandId')

echo "Command ID: $CMD_ID"
echo "Waiting for command to complete (poll every 30s, up to 60min)..."

# `aws ssm wait command-executed` caps at ~8min; the NEFF compile for
# the larger shapes can exceed that on first encounter. Poll manually.
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
