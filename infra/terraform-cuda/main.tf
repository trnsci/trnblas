terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Parallel to ../terraform/ (Trainium CI). This module provisions an
# NVIDIA GPU CI instance for cuBLAS head-to-head benchmarks (#4).
# Vintage-matched to trn1: A10G (GA102 Ampere, Apr 2021) on g5.*.

variable "aws_region" {
  description = "AWS region for the CI instance"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type (single-GPU, A10G-class for trn1 parity)"
  type        = string
  default     = "g5.xlarge"
  # A10G (24GB) on g5.xlarge/2xlarge/4xlarge. Choose by vCPU/RAM need;
  # the GPU itself is identical across the small g5 sizes.
}

variable "instance_tag" {
  description = "Tag used by scripts/run_cuda_bench.sh to find the instance"
  type        = string
  default     = "trnblas-ci-cuda-a10g"
}

variable "vpc_id" {
  description = "VPC to place the instance in"
  type        = string
}

variable "subnet_id" {
  description = "Subnet for the instance (public or private with NAT)"
  type        = string
}

provider "aws" {
  region = var.aws_region
}

# ---------------------------------------------------------------------------
# AWS Deep Learning AMI with PyTorch + CUDA (cuBLAS) — Ubuntu 24.04.
# ---------------------------------------------------------------------------

data "aws_ami" "cuda" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 24.04*"]
  }
}

# ---------------------------------------------------------------------------
# IAM role for the EC2 instance (SSM access)
# ---------------------------------------------------------------------------

resource "aws_iam_role" "instance" {
  name = "${var.instance_tag}-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "instance" {
  name = "${var.instance_tag}-profile"
  role = aws_iam_role.instance.name
}

# ---------------------------------------------------------------------------
# Security group (SSM only, no inbound)
# ---------------------------------------------------------------------------

resource "aws_security_group" "instance" {
  name        = "${var.instance_tag}-sg"
  description = "SSM-only access for trnblas CUDA CI"
  vpc_id      = var.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ---------------------------------------------------------------------------
# EC2 instance
# ---------------------------------------------------------------------------

resource "aws_instance" "ci" {
  ami                         = data.aws_ami.cuda.id
  instance_type               = var.instance_type
  subnet_id                   = var.subnet_id
  iam_instance_profile        = aws_iam_instance_profile.instance.name
  vpc_security_group_ids      = [aws_security_group.instance.id]
  associate_public_ip_address = true

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
  }

  user_data = <<-EOF
    #!/bin/bash
    set -euxo pipefail
    cd /home/ubuntu
    sudo -u ubuntu git clone https://github.com/trnsci/trnblas.git trnblas
    # The AWS CUDA DLAMI (Ubuntu 24.04) ships its PyTorch venv at /opt/pytorch.
    sudo -u ubuntu /opt/pytorch/bin/pip install -e '/home/ubuntu/trnblas[dev]'
  EOF

  tags = {
    Name = var.instance_tag
  }
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output "instance_id" {
  value = aws_instance.ci.id
}

output "instance_tag" {
  value       = var.instance_tag
  description = "Name tag used by scripts/run_cuda_bench.sh"
}

output "aws_region" {
  value       = var.aws_region
  description = "Region to pass to AWS CLI commands"
}
