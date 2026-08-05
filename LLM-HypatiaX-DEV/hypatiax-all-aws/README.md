## Reproducing HypatiaX Results

Click the button below. You need a free AWS account — no other software required.

[![Launch Stack](https://s3.amazonaws.com/cloudformation-examples/cloudformation-launch-stack.png)](https://console.aws.amazon.com/cloudformation/home#/stacks/create/review?templateURL=https://hypatiax-public.s3.amazonaws.com/hypatiax-reproducibility.yaml&stackName=hypatiax-repro)

### What happens
1. AWS provisions an `r6i.xlarge` instance (32 GB RAM, 4 vCPU) in your account
2. The instance clones this repository, runs the full pipeline (~4–6 hours), uploads results to S3, then **terminates itself** — you are not charged after it finishes
3. The **Results** link appears in the CloudFormation Outputs tab when done

### Parameters you may want to change
| Parameter | Default | Notes |
|-----------|---------|-------|
| `AnthropicApiKey` | *(empty)* | Paste your key for LLM steps; leave blank to run without |
| `PysrTimeout` | 1100 | Reduce to 500 for a faster (~2h) lower-fidelity run |
| `RepoBranch` | main | Pin to a specific commit SHA for exact reproducibility |

### Estimated cost
~$5–8 USD total (Spot instance + S3). The S3 bucket auto-deletes after 30 days.

### Troubleshooting
- **Stack stays in CREATE_IN_PROGRESS for >10 min:** the instance is provisioning normally; the pipeline runs inside the instance and takes 4–6 hours
- **Spot interruption:** re-launch the stack with the same parameters; the pipeline resumes from checkpoint automatically
- **No results URL after 8h:** open the EC2 console, find the instance named `hypatiax-runner-hypatiax-repro`, go to Actions → Monitor → Get system log
