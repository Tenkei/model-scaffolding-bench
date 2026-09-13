# Benchmarking

The batch-decode examples compare native model façades with explicit input
pipelines. They are intended for steady-state measurements: model construction,
model download, and first-use compilation must not be part of a measured run.

## Provider-independent procedure

Use an NVIDIA CUDA machine for the main comparison and record the complete
environment with every result:

```bash
nvidia-smi
python --version
python -m pip freeze
```

Use a persistent disk for model caches. This prevents model downloads from
entering a benchmark or differing between routes:

```text
~/.cache/huggingface
~/.cache/ultralytics
~/.paddlex
```

Run a one-iteration smoke test after setup, then use a warm-up phase and
multiple measured runs. Keep input images, model checkpoint, model batch size,
and output validation identical between the compared routes.

The Hugging Face benchmark, for example, can run on CUDA with:

```bash
python -m ml_pipes benchmark examples.run_transformers_vit_batch_decode_comparison \
  --axis strategy=transformers-paths,scatter-transformers-decode,direct-model,direct-model-concurrent-preprocess \
  --axis device=cuda \
  --axis inference_batch_size=8 \
  --axis max_concurrency=8 \
  --data-axis batch_size=8 \
  --runs 20 --warmup 3
```

Do not combine first-run, cold-cache, and steady-state results. If cold-cache
behavior matters, report it as a distinct experiment.

## AWS EC2

AWS EC2 is suitable when raw CUDA access, persistent storage, and repeatable
shell-based execution are needed. The entire workflow below uses the AWS CLI;
no console interaction is required.

### Prerequisites

Configure credentials and choose a region:

```bash
aws configure
aws sts get-caller-identity
aws configure get region
```

The active identity needs EC2 permissions. At a minimum, a launch workflow
requires `ec2:Describe*`, `ec2:RunInstances`, `ec2:CreateTags`,
`ec2:StopInstances`, and `ec2:TerminateInstances`. It also needs permissions
for any security group, EBS volume, SSH key, or IAM instance profile it creates
or uses.

Start with a CUDA-capable instance that has enough CPU capacity to expose input
pipeline effects. `g6.2xlarge` (NVIDIA L4, 8 vCPUs) or `g5.2xlarge` (NVIDIA
A10G, 8 vCPUs) are sensible initial choices when available in the chosen region.
For high-end replication, use an A100 or H100 instance after the experiment is
stable.

### Find an image and network targets

Use an Ubuntu or Deep Learning AMI with a compatible NVIDIA driver and CUDA
runtime. Resolve its image ID and choose an existing subnet, security group, and
SSH key or SSM instance profile according to the account's security policy.

The following read-only commands help discover available resources:

```bash
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

### Launch and connect

Replace the placeholders with identifiers from the account. The EBS volume holds
the repository and model caches across instance stops.

```bash
aws ec2 run-instances \
  --image-id ami-REPLACE_ME \
  --instance-type g6.2xlarge \
  --subnet-id subnet-REPLACE_ME \
  --security-group-ids sg-REPLACE_ME \
  --key-name REPLACE_ME \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3,DeleteOnTermination=false}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ml-pipes-benchmark}]' \
  --query 'Instances[0].InstanceId' \
  --output text
```

Wait for the instance, obtain its address, and connect over SSH:

```bash
aws ec2 wait instance-running --instance-ids i-REPLACE_ME

aws ec2 describe-instances \
  --instance-ids i-REPLACE_ME \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text

ssh -i /path/to/key.pem ubuntu@PUBLIC_IP
```

If the account uses Systems Manager, attach an instance profile with
`AmazonSSMManagedInstanceCore` when launching and use this instead of opening an
SSH security-group rule:

```bash
aws ssm start-session --target i-REPLACE_ME
```

### Set up and run the benchmarks

On the GPU instance, clone the repository, install the project, and verify CUDA
before benchmarking:

```bash
git clone https://github.com/requiem4machines/ml-pipes-ultralytics.git
cd ml-pipes-ultralytics

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install transformers

python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
nvidia-smi
```

Install the CUDA-compatible PaddlePaddle wheel before running the PaddleOCR
comparison. Its device option uses `gpu:0`; PyTorch-based YOLO and Transformers
examples use `cuda`.

Save the benchmark output and environment metadata beside the raw result:

```bash
mkdir -p benchmark-results
nvidia-smi > benchmark-results/nvidia-smi.txt
python -m pip freeze > benchmark-results/requirements.txt
```

### Stop or terminate

GPU instance charges stop only after stopping or terminating the instance. A
stopped instance still incurs EBS charges; terminating it removes the instance
and, with the launch command above, preserves the root volume for deliberate
cleanup later.

```bash
aws ec2 stop-instances --instance-ids i-REPLACE_ME

# Use only when the instance and its retained volume are no longer needed.
aws ec2 terminate-instances --instance-ids i-REPLACE_ME
```

## Other providers

Future provider-specific sections should retain the same procedure: record the
hardware and software environment, use persistent model caches, run the same
commands, and preserve raw results. This keeps AWS, GCP, NVIDIA-hosted, and
other measurements comparable.
