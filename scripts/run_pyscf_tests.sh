#!/usr/bin/env bash
#
# Run the PySCF precision tests (#20) on the trnblas CI trn1 instance.
#
# Usage:
#   AWS_PROFILE=aws ./scripts/run_pyscf_tests.sh            # fast tests only
#   AWS_PROFILE=aws ./scripts/run_pyscf_tests.sh --slow     # include slow (cc-pVTZ, glycine/cc-pVDZ)
#   AWS_PROFILE=aws ./scripts/run_pyscf_tests.sh --all      # all pyscf tests
#
# PySCF is not in the default trn1 user-data; this script installs
# trnblas[pyscf] in the Neuron venv before running. The tests use
# torch.matmul (CPU fallback) — no Neuron hardware needed, but the
# trn1 instance has enough RAM for the larger integral computations.
#
# Tests marked @pytest.mark.pyscf are in tests/test_df_mp2_pyscf.py.
# Tests marked @pytest.mark.slow include glycine/cc-pVDZ, h2o_trimer,
# and h2o/cc-pVTZ — these are the FP32 precision envelope cases (#20).

set -euo pipefail

SLOW=0
MARKER_EXPR="pyscf"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --slow)
      SLOW=1
      MARKER_EXPR="pyscf and slow"
      shift
      ;;
    --all)
      MARKER_EXPR="pyscf"
      shift
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

: "${AWS_PROFILE:?Set AWS_PROFILE, e.g. AWS_PROFILE=aws ./scripts/run_pyscf_tests.sh}"

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

echo "Sending PySCF test command (SHA=$SHA, marker='$MARKER_EXPR', slow=$SLOW)..."
CMD_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --comment "trnblas pyscf tests @ $SHA" \
  --parameters "commands=[
    \"bash -c 'set -euo pipefail; cd /home/ubuntu/trnblas && sudo -u ubuntu git fetch --all && sudo -u ubuntu git checkout $SHA && NEURON_VENV=\$(ls -d /opt/aws_neuronx_venv_pytorch_* | head -1) && sudo -u ubuntu \$NEURON_VENV/bin/pip install -e /home/ubuntu/trnblas[dev,pyscf] --quiet && sudo -u ubuntu env PATH=\$NEURON_VENV/bin:/usr/bin:/bin TMPDIR=/var/tmp TRNBLAS_REQUIRE_NKI=1 \$NEURON_VENV/bin/pytest /home/ubuntu/trnblas/tests/test_df_mp2_pyscf.py -v -s -m \"$MARKER_EXPR\" --tb=short'\"
  ]" \
  --region "$REGION" \
  --output text --query 'Command.CommandId')

echo "Command ID: $CMD_ID"
echo "Waiting for PySCF tests (slow tests take 2-5 min each; --slow adds ~20 min)..."

# Poll every 30s, up to 60 min.
STATUS=InProgress
for _ in $(seq 1 120); do
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
