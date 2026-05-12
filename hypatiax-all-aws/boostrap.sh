#!/bin/bash
set -euo pipefail
exec > /var/log/hypatiax.log 2>&1

# Install deps
dnf install -y git python3.12 python3.12-pip
pip3.12 install julia

# Clone repo
git clone https://github.com/YOUR_ORG/LLM-HypatiaX-PAPERS-Public /opt/hypatiax
cd /opt/hypatiax

# Install Julia + Python deps
pip3.12 install -r requirements.txt
python3.12 -c "import julia; julia.install()"

# Pull checkpoint from S3 if resuming
aws s3 cp s3://hypatiax-results/checkpoints/ logs/ \
  --recursive --no-sign-request 2>/dev/null || true

# Set secrets from SSM Parameter Store
export ANTHROPIC_API_KEY=$(aws ssm get-parameter \
  --name /hypatiax/anthropic_key --with-decryption \
  --query Parameter.Value --output text)

# Run the pipeline
python3.12 run_all_checkpoint.py --resume --continue-on-fail
python3.12 run_exp2_symbolic_engine.py --resume
python3.12 run_exp2_hybrid_system.py --resume

# Upload results
aws s3 sync hypatiax/data/results/ s3://hypatiax-results/results/
aws s3 sync logs/ s3://hypatiax-results/checkpoints/

# Self-terminate to stop billing
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID"
