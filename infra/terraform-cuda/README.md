# Terraform — trnblas CUDA (A10G) CI instance

Provisions a single-GPU NVIDIA A10G instance (`g5.xlarge` by default)
for running the DF-MP2 bench against cuBLAS. Vintage-matched to
Trainium1 — A10G (GA102 Ampere, Apr 2021) vs trn1 (Oct 2022).

Parallel to `infra/terraform/` (the Trainium CI module); use a
separate state dir so the two lifecycles are independent.

## What it creates

- **EC2 instance** — `g5.xlarge` by default (Deep Learning AMI, Ubuntu 24.04).
- **IAM instance profile** — SSM managed-instance-core permissions.
- **Security group** — SSM only, no inbound.

## Apply

```bash
cd infra/terraform-cuda

AWS_PROFILE=aws terraform init

AWS_PROFILE=aws terraform apply \
  -var="vpc_id=vpc-xxxxxx" \
  -var="subnet_id=subnet-xxxxxx"
```

Outputs include `instance_id` and `instance_tag`. Wait ~2–3 minutes for
user-data to finish (DLAMI already has PyTorch + CUDA; we just clone
and pip-install), then stop it:

```bash
AWS_PROFILE=aws aws ec2 stop-instances \
  --instance-ids $(terraform output -raw instance_id)
```

## Variables

| Variable | Default | Notes |
|----------|---------|-------|
| `aws_region` | `us-east-1` | |
| `instance_type` | `g5.xlarge` | Also: `g5.2xlarge` (more vCPU/RAM, same A10G) |
| `instance_tag` | `trnblas-ci-cuda-a10g` | Must match `scripts/run_cuda_bench.sh` |
| `vpc_id` | (required) | Can reuse the Trainium VPC |
| `subnet_id` | (required) | g5.* has broader AZ availability than trn1.* |

## Cost

- Stopped: ~$10/mo EBS (100 GB gp3).
- Running: `g5.xlarge` ≈ $1.006/hr on-demand in us-east-1.

A full DF-MP2 bench cycle (all three shapes, cold + warm) runs in
~10–15 minutes of instance-on time — ~$0.25 per run.
