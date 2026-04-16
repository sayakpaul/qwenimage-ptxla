## Environment Variables

Before starting, ensure you have installed and configured the `gcloud` command.

```bash
export PROJECT_ID=...
export ZONE=us-...
export TPU_NAME=pytorch-xla-sprint
export DISK_NAME=pytorch-xla-data
export ACCELERATOR_TYPE=v6e-8
export RUNTIME_VERSION=v2-alpha-tpuv6e
```

## Create a Hyperdisk

We need it to be fault-tolerant. Because we'll be working with on-spot instances that can be preempted, we will always configure our environments, model cache, etc. on a mounted disk so that our progress isn't lost. v6e doesn't support standard Persistent Disk — use Hyperdisk Balanced.

```bash
gcloud compute disks create ${DISK_NAME} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --type=hyperdisk-balanced \
  --size=1000GB
```

## Create the TPU VM

Before, running the following, ensure you have TPU services and other stuff enabled. Please consult the Google Cloud documentation for that (or have your
favorite AI agent do that for you).

We'll be working with on spot instances. The command below creates one.

```bash
gcloud alpha compute tpus tpu-vm create ${TPU_NAME} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --accelerator-type=${ACCELERATOR_TYPE} \
  --version=${RUNTIME_VERSION} \
  --spot \
  --data-disk source=projects/${PROJECT_ID}/zones/${ZONE}/disks/${DISK_NAME},mode=read-write
```

It's common to see capacity issues and we cannot do much about it other than
being patient.

## SSH into the VM

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --project=${PROJECT_ID} \
  --zone=${ZONE}
```

## Format and Mount the Disk

Find the disk (usually `nvme0n2` for v6e):

```bash
lsblk
```

Locate the volume that shows "1000 GB".

Format (**first time only** — skip if re-mounting an existing disk):

```bash
sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0 /dev/nvme0n2
```

Mount:

```bash
sudo mkdir -p /mnt/data
sudo mount -o discard,defaults /dev/nvme0n2 /mnt/data
sudo chmod a+w /mnt/data
```

Persist across reboots:

```bash
echo UUID=$(sudo blkid -s UUID -o value /dev/nvme0n2) /mnt/data ext4 discard,defaults,nofail 0 2 | sudo tee -a /etc/fstab
```

Even if the machine is preempted, we can still benefit from this disk (particularly everything that's saved in `/mnt/data`).

## Set Up Conda (on the Mounted Disk)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /mnt/data/miniconda3
rm Miniconda3-latest-Linux-x86_64.sh
eval "$(/mnt/data/miniconda3/bin/conda shell.bash hook)"
conda init
source ~/.bashrc

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create -n ptxla python=3.11 -y
conda activate ptxla
```

## Install PyTorch/XLA + Pallas

```bash
pip install torch==2.8.0 'torch_xla[tpu]==2.8.0'

pip install --pre 'torch_xla[pallas]' \
  --index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ \
  --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## Install Diffusers and Friends

```bash
pip install transformers accelerate safetensors sentencepiece protobuf huggingface_hub structlog
```

Install Diffusers from source:

```bash
git clone https://github.com/huggingface/diffusers/
cd diffusers
pip install -e .
```

## Configure HuggingFace Cache

Point everything at the mounted disk so models don't fill the boot disk and we
can benefit from when using another VM instance:

```bash
echo 'export HF_HOME=/mnt/data/huggingface' >> ~/.bashrc
echo 'export HUGGINGFACE_HUB_CACHE=/mnt/data/huggingface' >> ~/.bashrc
echo 'export TMPDIR=/mnt/data/huggingface' >> ~/.bashrc
source ~/.bashrc
mkdir -p /mnt/data/huggingface
```

## Verify

```python
import torch
import torch_xla
import torch_xla.core.xla_model as xm

device = xm.xla_device()
t = torch.randn(3, 3, device=device)
print(t)
print(f"Device: {t.device}")

import torch_xla.experimental.custom_kernel
print("Pallas OK")
```

## Managing the VM

Stop:

```bash
gcloud compute tpus tpu-vm stop ${TPU_NAME} --project=${PROJECT_ID} --zone=${ZONE}
```

Start:

```bash
gcloud compute tpus tpu-vm start ${TPU_NAME} --project=${PROJECT_ID} --zone=${ZONE}
```

Delete VM:

```bash
gcloud alpha compute tpus tpu-vm delete ${TPU_NAME} --project=${PROJECT_ID} --zone=${ZONE}
```

Delete disk (detach or delete VM first):

```bash
gcloud compute disks delete ${DISK_NAME} --project=${PROJECT_ID} --zone=${ZONE}
```

## After a Preemption (Spot VMs)

Spot VMs can't be restarted once preempted — delete and recreate:

```bash
gcloud alpha compute tpus tpu-vm delete ${TPU_NAME} --project=${PROJECT_ID} --zone=${ZONE}

gcloud alpha compute tpus tpu-vm create ${TPU_NAME} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --accelerator-type=${ACCELERATOR_TYPE} \
  --version=${RUNTIME_VERSION} \
  --spot \
  --data-disk source=projects/${PROJECT_ID}/zones/${ZONE}/disks/${DISK_NAME},mode=read-write
```

Then SSH in and re-mount (no formatting needed):

```bash
sudo mkdir -p /mnt/data
sudo mount -o discard,defaults /dev/nvme0n2 /mnt/data

# Re-init conda
eval "$(/mnt/data/miniconda3/bin/conda shell.bash hook)"
conda init && source ~/.bashrc
conda activate ptxla

# Re-add env vars (new VM = fresh ~/.bashrc)
echo 'export HF_HOME=/mnt/data/huggingface' >> ~/.bashrc
echo 'export HUGGINGFACE_HUB_CACHE=/mnt/data/huggingface' >> ~/.bashrc
echo 'export TMPDIR=/mnt/data/huggingface' >> ~/.bashrc
source ~/.bashrc
```
