#!/usr/bin/env bash
#
# Run the df_mp2.py bench on the trnblas CI trn1 instance.
#
# Usage:
#   AWS_PROFILE=aws ./scripts/run_bench.sh                    # medium + large shapes
#   AWS_PROFILE=aws ./scripts/run_bench.sh --medium-only      # medium shape only
#   AWS_PROFILE=aws ./scripts/run_bench.sh --shape large      # one shape
#
# Runs `python examples/df_mp2.py --bench --batched-pair-energy` with cold
# and warm timing. Cold compile can take 30–90 min for large shapes; the
# script polls up to 120 min.
#
# The trnblas CI instance (trnblas-ci-trn1) must be provisioned via
# infra/terraform/ before running. See docs/aws_setup.md.

set -euo pipefail

SHAPES=("medium" "large")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --medium-only)
      SHAPES=("medium")
      shift
      ;;
    --shape)
      SHAPES=("$2")
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

INSTANCE_TYPE="${INSTANCE_TYPE:-trn1}"
TAG="trnblas-ci-${INSTANCE_TYPE}"
case "$INSTANCE_TYPE" in
  trn2*) REGION="${AWS_REGION:-sa-east-1}" ;;
  *)     REGION="${AWS_REGION:-us-east-1}" ;;
esac
SHA="$(git rev-parse HEAD)"

: "${AWS_PROFILE:?Set AWS_PROFILE, e.g. AWS_PROFILE=aws ./scripts/run_bench.sh}"

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

# Build the remote bench command: run df_mp2.py --bench for each shape.
# Large shape cold compile can take 30-90 min (chunked NEFF compilations).
SHAPES_ARG="${SHAPES[*]}"
echo "Sending bench command (SHA=$SHA, shapes=${SHAPES_ARG})..."

# Build SSM parameters via Python with a base64-encoded bash script.
# Same pattern as run_pyscf_tests.sh — see that file for rationale.
PARAMS_FILE=$(mktemp /tmp/trnblas-bench-XXXXXX.json)
SHA_VAL="$SHA" SHAPES_VAL="$SHAPES_ARG" python3 - <<'PYEOF' > "$PARAMS_FILE"
import base64, json, os

sha    = os.environ["SHA_VAL"]
shapes = os.environ["SHAPES_VAL"].split()

# Build the bench commands: one python invocation per shape.
bench_cmds = ""
for shape in shapes:
    bench_cmds += (
        f"echo '--- shape={shape} ---'\n"
        f"sudo -u ubuntu env PATH=$NEURON_VENV/bin:/usr/bin:/bin TMPDIR=/var/tmp"
        f" TRNBLAS_REQUIRE_NKI=1"
        f" $NEURON_VENV/bin/python /home/ubuntu/trnblas/examples/df_mp2.py"
        f" --bench --shape {shape} --batched-pair-energy\n"
    )

script = (
    "#!/bin/bash\n"
    "set -euo pipefail\n"
    "cd /home/ubuntu/trnblas\n"
    "sudo -u ubuntu git fetch --all\n"
    f"sudo -u ubuntu git checkout {sha}\n"
    "NEURON_VENV=$(ls -d /opt/aws_neuronx_venv_pytorch_* | head -1)\n"
    "sudo -u ubuntu $NEURON_VENV/bin/pip install -e '/home/ubuntu/trnblas[dev]' --quiet\n"
    + bench_cmds
)
encoded = base64.b64encode(script.encode()).decode()
print(json.dumps({"commands": [f"echo '{encoded}' | base64 -d | bash"]}))
PYEOF

CMD_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --comment "trnblas bench @ $SHA shapes=${SHAPES_ARG}" \
  --parameters "file://$PARAMS_FILE" \
  --timeout-seconds 7200 \
  --region "$REGION" \
  --output text --query 'Command.CommandId')
rm -f "$PARAMS_FILE"

echo "Command ID: $CMD_ID"
echo "Waiting for bench to complete (cold compile for large shape: 30-90 min)..."

# Poll every 30s, up to 120 min.
STATUS=InProgress
for _ in $(seq 1 240); do
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
