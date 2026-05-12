# 1. Create an S3 bucket for results
aws s3 mb s3://hypatiax-results

# 2. Create an IAM role with S3 write + self-terminate permissions
# (policy: s3:PutObject on your bucket, ec2:TerminateInstances on self)

# 3. Launch the instance
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \           # Amazon Linux 2023, us-east-1
  --instance-type r6i.xlarge \                  # 4 vCPU, 32 GB RAM
  --instance-market-options '{"MarketType":"spot"}' \
  --iam-instance-profile Name=hypatiax-runner \
  --key-name your-key \
  --user-data file://bootstrap.sh \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":40}}]'
