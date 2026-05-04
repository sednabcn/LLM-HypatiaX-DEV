# HypatiaX CI — Self-Hosted Runner Setup Guide
## GCP `n2-standard-8` and AWS EC2 `c5.4xlarge`

This guide covers everything needed to run `ci-gcp.yml` or `ci-aws.yml` instead of the GitHub-hosted `ci.yml`.  
The self-hosted variants unlock:
- No 6-hour per-job cap (suppB/instability need up to 12 h)
- 8 vCPUs + 32 GB RAM (vs 2 vCPU / 7 GB on `ubuntu-latest`)
- PySR with `multithreading` across 30 populations runs ~4× faster
- No GitHub Actions compute billing (you pay cloud directly)

---

## Which file goes where

| File | Purpose |
|---|---|
| `.github/workflows/ci.yml` | GitHub-hosted ubuntu-latest (v9 fixed — use for PRs/small runs) |
| `.github/workflows/ci-gcp.yml` | GCP n2-standard-8 self-hosted |
| `.github/workflows/ci-aws.yml` | AWS c5.4xlarge self-hosted |

All three files are identical except for `runs-on:` labels and the `supp-slow` timeout (350 min on GH-hosted to stay under the 6 h cap; 720 min on self-hosted for full suppB+instability headroom).

---

## Option 1 — GCP `n2-standard-8`

### Specs
| | |
|---|---|
| vCPUs | 8 |
| RAM | 32 GB |
| Disk (recommend) | 200 GB SSD (`pd-ssd`) |
| OS | Ubuntu 22.04 LTS |
| Region | `europe-west2` (London) or nearest to your team |

### Cost estimate (on-demand, May 2026)

| Scenario | Hours/run | $/hr | Cost/run |
|---|---|---|---|
| Full pipeline (75 h wall, 1 runner) | 75 h | ~$0.38 | **~$28.50** |
| Full pipeline (8 parallel runners) | ~10 h | $0.38 × 8 | **~$30.40** |
| Spot/preemptible instance | 75 h | ~$0.11 | **~$8.25** |

> **Tip**: Use a preemptible instance for main-branch runs. Checkpointing means a preemption restart picks up mid-experiment. Add `--restart-policy=on-failure` to the runner service.

### Step-by-step setup

#### 1. Create the VM

```bash
gcloud compute instances create hypatiax-runner \
  --machine-type=n2-standard-8 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --zone=europe-west2-a \
  --tags=github-runner \
  --scopes=cloud-platform
```

For **preemptible** (cheaper, restartable):
```bash
gcloud compute instances create hypatiax-runner \
  --machine-type=n2-standard-8 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --zone=europe-west2-a \
  --preemptible \
  --restart-on-failure
```

#### 2. SSH in and install dependencies

```bash
gcloud compute ssh hypatiax-runner --zone=europe-west2-a
```

```bash
# System packages
sudo apt-get update && sudo apt-get install -y \
  curl wget git build-essential libssl-dev \
  python3.12 python3.12-dev python3.12-venv python3-pip \
  julia \
  docker.io jq unzip

# Julia 1.11 (exact version)
curl -fsSL https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.0-linux-x86_64.tar.gz \
  | sudo tar -xz -C /usr/local --strip-components=1

# Verify
python3.12 --version   # Python 3.12.x
julia --version        # julia version 1.11.0
```

#### 3. Register the GitHub Actions runner

In your repo: **Settings → Actions → Runners → New self-hosted runner**  
Copy the registration token, then on the VM:

```bash
mkdir -p /home/ubuntu/actions-runner && cd /home/ubuntu/actions-runner

# Download runner (check https://github.com/actions/runner/releases for latest)
curl -o actions-runner-linux-x64.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.316.0/actions-runner-linux-x64-2.316.0.tar.gz
tar xzf actions-runner-linux-x64.tar.gz

# Configure — replace TOKEN and YOUR_ORG/YOUR_REPO
./config.sh \
  --url https://github.com/YOUR_ORG/YOUR_REPO \
  --token YOUR_REGISTRATION_TOKEN \
  --labels "self-hosted,linux,gcp-n2-standard-8" \
  --name "gcp-n2-hypatiax-01" \
  --work "/home/ubuntu/runner-work" \
  --unattended
```

> **The `--labels` flag must exactly match `runs-on:` in `ci-gcp.yml`**: `[self-hosted, linux, gcp-n2-standard-8]`

#### 4. Install as a systemd service (survives reboots)

```bash
sudo ./svc.sh install ubuntu
sudo ./svc.sh start
sudo systemctl enable actions.runner.YOUR_ORG-YOUR_REPO.gcp-n2-hypatiax-01.service
```

Verify it's running:
```bash
sudo systemctl status "actions.runner.*"
```

#### 5. Use `ci-gcp.yml` in your repo

```bash
# In your repo
cp ci-gcp.yml .github/workflows/ci-gcp.yml
git add .github/workflows/ci-gcp.yml
git commit -m "ci: add GCP self-hosted runner workflow"
git push
```

Trigger via GitHub UI: **Actions → HypatiaX Reproducibility CI — v9 [GCP] → Run workflow**

---

## Option 2 — AWS EC2 `c5.4xlarge`

### Specs
| | |
|---|---|
| vCPUs | 16 (hyperthreaded, 8 physical) |
| RAM | 32 GB |
| Storage | 200 GB `gp3` EBS |
| Network | Up to 10 Gbps |
| OS | Ubuntu 22.04 LTS (ami-0b45ae66668865cd8 in eu-west-2) |

### Cost estimate (on-demand, May 2026)

| Scenario | Hours/run | $/hr | Cost/run |
|---|---|---|---|
| Full pipeline, on-demand | 75 h | $0.680 | **~$51.00** |
| Full pipeline, Spot | 75 h | ~$0.20 | **~$15.00** |
| Savings Plan (1-yr compute) | 75 h | ~$0.43 | **~$32.25** |

> `c5.4xlarge` is compute-optimised (Intel Xeon Platinum, AVX-512). PySR benefits significantly from AVX-512 vectorisation vs `n2-standard-8` which is also good but uses different instruction scheduling. For pure symbol-regression throughput `c5.4xlarge` is typically 10-15% faster per core.  
> For **cost efficiency**, GCP preemptible n2-standard-8 wins (~$8/run vs ~$15/run Spot).  
> For **raw speed**, `c5.4xlarge` Spot is the better choice.

### Step-by-step setup

#### 1. Launch the instance

Via AWS CLI:
```bash
aws ec2 run-instances \
  --image-id ami-0b45ae66668865cd8 \
  --instance-type c5.4xlarge \
  --key-name YOUR_KEY_PAIR \
  --security-group-ids YOUR_SG_ID \
  --subnet-id YOUR_SUBNET_ID \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3","Iops":3000}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=hypatiax-runner}]' \
  --count 1
```

For **Spot** (cheaper):
```bash
aws ec2 run-instances \
  --image-id ami-0b45ae66668865cd8 \
  --instance-type c5.4xlarge \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"persistent","InstanceInterruptionBehavior":"stop"}}' \
  --key-name YOUR_KEY_PAIR \
  --security-group-ids YOUR_SG_ID \
  --subnet-id YOUR_SUBNET_ID \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' \
  --count 1
```

#### 2. SSH in and install dependencies

```bash
ssh -i YOUR_KEY.pem ubuntu@YOUR_INSTANCE_IP
```

```bash
sudo apt-get update && sudo apt-get install -y \
  curl wget git build-essential libssl-dev \
  python3.12 python3.12-dev python3.12-venv python3-pip \
  jq unzip

# Julia 1.11
curl -fsSL https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.0-linux-x86_64.tar.gz \
  | sudo tar -xz -C /usr/local --strip-components=1

julia --version   # julia version 1.11.0
python3.12 --version
```

#### 3. Register the runner

```bash
mkdir -p /home/ubuntu/actions-runner && cd /home/ubuntu/actions-runner

curl -o actions-runner-linux-x64.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.316.0/actions-runner-linux-x64-2.316.0.tar.gz
tar xzf actions-runner-linux-x64.tar.gz

./config.sh \
  --url https://github.com/YOUR_ORG/YOUR_REPO \
  --token YOUR_REGISTRATION_TOKEN \
  --labels "self-hosted,linux,aws-c5-4xlarge" \
  --name "aws-c5-hypatiax-01" \
  --work "/home/ubuntu/runner-work" \
  --unattended
```

> **Labels must exactly match `ci-aws.yml`**: `[self-hosted, linux, aws-c5-4xlarge]`

#### 4. systemd service

```bash
sudo ./svc.sh install ubuntu
sudo ./svc.sh start
sudo systemctl enable "actions.runner.*"
```

#### 5. Use `ci-aws.yml`

```bash
cp ci-aws.yml .github/workflows/ci-aws.yml
git add .github/workflows/ci-aws.yml
git commit -m "ci: add AWS self-hosted runner workflow"
git push
```

---

## GitHub repository secrets needed (all three variants)

Go to **Settings → Secrets and variables → Actions → New repository secret**:

| Secret | Value |
|---|---|
| `ANTHROPIC_API_KEY` | Your Anthropic API key |

No other secrets are required — GitHub Actions injects `GITHUB_TOKEN` automatically.

---

## Cost comparison summary

| Option | Full pipeline cost | Speed vs GH-hosted | Notes |
|---|---|---|---|
| GitHub-hosted `ubuntu-latest` | ~$0 (included minutes) | 1× (baseline) | Fails: 6h job cap, 2 vCPU |
| GCP n2-standard-8 on-demand | ~$28.50/run | ~4× | Good default |
| GCP n2-standard-8 preemptible | **~$8.25/run** | ~4× | Best value; checkpoint-safe |
| AWS c5.4xlarge on-demand | ~$51.00/run | ~4.5× | Fastest per-core |
| AWS c5.4xlarge Spot | ~$15.00/run | ~4.5× | Best speed/cost if AVX needed |

> All estimates based on May 2026 list prices in `europe-west2` (GCP) and `eu-west-2` (AWS). Prices vary by region and change over time — always verify at [cloud.google.com/compute/vm-instance-pricing](https://cloud.google.com/compute/vm-instance-pricing) and [aws.amazon.com/ec2/pricing](https://aws.amazon.com/ec2/pricing).

---

## Recommended workflow

1. **PRs** → use `ci.yml` (GitHub-hosted, fast, free, only runs lint + verify)
2. **main branch merges** → use `ci-gcp.yml` (preemptible, ~$8/run, full pipeline)
3. **Paper submission / reproducibility audit** → use `ci-aws.yml` (Spot, fastest, ~$15/run)

To restrict which workflow runs on which branch, add `branches:` filters or use separate workflow files as provided.
