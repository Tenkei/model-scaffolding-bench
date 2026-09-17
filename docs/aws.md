# AWS EC2

This guide provisions and runs a CUDA benchmark through the AWS CLI. It assumes
an account policy that permits EC2 discovery, launch, tagging, stopping, and
termination.

## Step 0: Choose hardware

Start with a GPU instance that also has enough CPU capacity to expose input
pipeline effects:

| Instance family | GPU | Initial use |
|---|---|---|
| `g6.2xlarge` | NVIDIA L4, 24 GB | 8 vCPUs; a balanced first experiment |
| `g5.2xlarge` | NVIDIA A10G, 24 GB | Similar alternative where L4 is unavailable |
| `p4d.24xlarge` | 8× NVIDIA A100, 40 GB | High-end replication after the workload is stable |

Use a Deep Learning AMI or Ubuntu image with a compatible NVIDIA driver and
CUDA runtime. Plan for at least 100 GB of EBS storage for the repository and
framework caches; add or resize storage only if the selected AMI's root volume
is not large enough.

## Step 1: Verify the account and GPU availability

```bash
aws sts get-caller-identity

aws ec2 describe-instance-type-offerings \
  --location-type availability-zone \
  --filters Name=instance-type,Values=g6.2xlarge,g5.2xlarge
```

## Step 2: Find the AMI

The following launch steps use the default VPC and subnet, then create a
dedicated security group for the benchmark.

The command selects the latest NVIDIA GPU Deep Learning AMI for Ubuntu 22.04
in the configured region.

```bash
AMI_ID=$(aws ec2 describe-images \
  --owners amazon \
  --filters 'Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) ????????' 'Name=state,Values=available' \
  --query 'reverse(sort_by(Images, &CreationDate))[:1].ImageId' \
  --output text)

echo "Using AMI: $AMI_ID"

# Example output in ap-northeast-1 on 2026-09-16:
# Using AMI: ami-015bb9fad84746391
```

The launch command uses `AMI_ID` automatically. See AWS's [DLAMI ID lookup guide](https://docs.aws.amazon.com/dlami/latest/devguide/find-dlami-id.html)
for other operating-system or framework images.

## Step 3: Create the SSH key pair

Run this block once to create the key pair; later runs reuse the local key file.

```bash
mkdir -p ~/.ssh

if [ -e ~/.ssh/model-scaffolding-bench-key.pem ]; then
  echo 'Key file already exists; reusing it.'
elif KEY_MATERIAL=$(aws ec2 create-key-pair \
  --key-name model-scaffolding-bench-key \
  --key-type ed25519 \
  --query 'KeyMaterial' \
  --output text); then
  printf '%s\n' "$KEY_MATERIAL" > ~/.ssh/model-scaffolding-bench-key.pem
  chmod 400 ~/.ssh/model-scaffolding-bench-key.pem
  unset KEY_MATERIAL
fi

# Confirm that the public key is registered in EC2.
aws ec2 describe-key-pairs \
  --key-names model-scaffolding-bench-key \
  --query 'KeyPairs[0].{Name:KeyName,Type:KeyType,Fingerprint:KeyFingerprint}' \
  --output table
```

> [!CAUTION]
> AWS returns the private key only at creation time, so keep the `.pem` file
> private and do not delete it. If the AWS key pair already exists but the local
> file does not, choose a new key name; the lost private key cannot be recovered.
> The verification command checks the public key registered with EC2, not the
> contents of the private `.pem` file.

## Step 4: Create the SSH security group

```bash
VPC_ID=$(aws ec2 describe-vpcs \
  --filters Name=is-default,Values=true \
  --query 'Vpcs[0].VpcId' \
  --output text)

SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=vpc-id,Values=$VPC_ID" 'Name=group-name,Values=model-scaffolding-bench-ssh' \
  --query 'SecurityGroups[0].GroupId' \
  --output text)

if [ "$SG_ID" = "None" ]; then
  SG_ID=$(aws ec2 create-security-group \
    --group-name model-scaffolding-bench-ssh \
    --description 'Temporary SSH access for model-scaffolding-bench' \
    --vpc-id "$VPC_ID" \
    --query 'GroupId' \
    --output text)
fi

MY_PUBLIC_IP=$(curl --fail --silent https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress \
  --group-id "$SG_ID" \
  --protocol tcp --port 22 --cidr "$MY_PUBLIC_IP/32"
```

> [!CAUTION]
> This rule permits SSH only from the computer's current public IP. Do not
> replace the `/32` suffix with `0.0.0.0/0`. If the command reports a duplicate
> rule, the correct rule already exists and the setup can continue.

## Step 5: Launch and connect

```bash
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type g6.2xlarge \
  --key-name model-scaffolding-bench-key \
  --security-group-ids "$SG_ID" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Application,Value=model-scaffolding-bench}]' \
  --query 'Instances[0].InstanceId' \
  --output text)

aws ec2 wait instance-status-ok --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

ssh -i ~/.ssh/model-scaffolding-bench-key.pem ubuntu@"$PUBLIC_IP"
```

The `Application` tag satisfies accounts that require an application tag when
launching instances. Use the value required by your organization's tag policy.

> [!TIP]
> Add `--instance-initiated-shutdown-behavior terminate` to the launch command
> if shutting down the operating system should also terminate the instance.
> Copy results off the instance first. This avoids leaving the instance running
> after an operating-system shutdown; separate EBS volumes may still remain.

## Step 6: Prepare and run

Clone this repository on the instance:

```bash
git clone REPOSITORY_URL
cd model-scaffolding-bench
```

Continue with [Running experiments](running-experiments.md), starting at
**Set up the case-study environment**. It covers virtual-environment setup,
the required project extra, GPU verification, benchmark execution, and
environment/result capture.

Use the [Case-study index](index.md) to select a model and its example command.

## Step 7: Stop or terminate

GPU instance charges continue while an instance runs. Stopping preserves the
attached EBS volume but still incurs storage charges. Terminate only after
preserving required artifacts.

```bash
aws ec2 stop-instances --instance-ids "$INSTANCE_ID"

# Destructive: removes the instance. Check the EBS volume settings before
# running this command, then preserve any required artifacts first.
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID"

# Delete the dedicated SSH group after the instance has terminated.
aws ec2 wait instance-terminated --instance-ids "$INSTANCE_ID"
aws ec2 delete-security-group --group-id "$SG_ID"
```

## Next

Read [Running experiments](running-experiments.md) to run a case study and
save the measurements from the prepared instance.
