# AWS EC2

This guide provisions and runs a CUDA benchmark through the AWS CLI. It assumes
an account policy that permits EC2 discovery, launch, tagging, stopping, and
termination.

## Choose hardware

Start with a GPU instance that also has enough CPU capacity to expose input
pipeline effects:

| Instance family | GPU | Initial use |
|---|---|---|
| `g6.2xlarge` | NVIDIA L4, 24 GB | 8 vCPUs; a balanced first experiment |
| `g5.2xlarge` | NVIDIA A10G, 24 GB | Similar alternative where L4 is unavailable |
| `p4d.24xlarge` | 8× NVIDIA A100, 40 GB | High-end replication after the workload is stable |

Use a Deep Learning AMI or Ubuntu image with a compatible NVIDIA driver and
CUDA runtime. Attach a persistent `gp3` EBS volume of at least 100 GB for the
repository and framework caches.

## Discover account resources

```bash
aws sts get-caller-identity

aws ec2 describe-instance-type-offerings \
  --location-type availability-zone \
  --filters Name=instance-type,Values=g6.2xlarge,g5.2xlarge

aws ec2 describe-subnets \
  --query 'Subnets[].{SubnetId:SubnetId,AZ:AvailabilityZone,VpcId:VpcId}' \
  --output table

aws ec2 describe-security-groups \
  --query 'SecurityGroups[].{GroupId:GroupId,Name:GroupName,VpcId:VpcId}' \
  --output table
```

## Launch and connect

Replace all placeholders with account-specific values. Prefer Systems Manager
when the account supports it; otherwise restrict SSH ingress to a known IP.

```bash
aws ec2 run-instances \
  --image-id ami-REPLACE_ME \
  --instance-type g6.2xlarge \
  --subnet-id subnet-REPLACE_ME \
  --security-group-ids sg-REPLACE_ME \
  --key-name REPLACE_ME \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3,DeleteOnTermination=false}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=model-scaffolding-bench}]' \
  --query 'Instances[0].InstanceId' \
  --output text

aws ec2 wait instance-running --instance-ids i-REPLACE_ME

aws ec2 describe-instances \
  --instance-ids i-REPLACE_ME \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text

ssh -i /path/to/key.pem ubuntu@PUBLIC_IP
```

For Systems Manager, launch with an instance profile containing
`AmazonSSMManagedInstanceCore`, then connect with:

```bash
aws ssm start-session --target i-REPLACE_ME
```

## Prepare and run

Clone this repository, install the required framework integrations, and verify
the selected accelerator before a measured run:

```bash
git clone REPOSITORY_URL
cd model-scaffolding-bench

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[transformers]'

python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
nvidia-smi
```

Install the relevant project extra before executing a case study. PaddleOCR
requires a CUDA-compatible PaddlePaddle wheel and uses `gpu:0`; PyTorch-based
YOLO and Transformers benchmarks use `cuda`.

Record environment metadata and write benchmark output under `results/`
before processing it into tables or figures.

## Stop or terminate

GPU instance charges continue while an instance runs. Stopping preserves the
attached EBS volume but still incurs storage charges. Terminate only after
preserving required artifacts.

```bash
aws ec2 stop-instances --instance-ids i-REPLACE_ME

# Destructive: removes the instance. The launch command retains its root EBS volume.
aws ec2 terminate-instances --instance-ids i-REPLACE_ME
```

## Next

Read [Running experiments](running-experiments.md) to run a case study and
save the measurements from the prepared instance.
