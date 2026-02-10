# DDP EKS Infrastructure

Production-ready EKS cluster with Karpenter for distributed deep learning workloads.

## Quick Start

```bash
# Initialize Terraform
terraform init

# Review the plan
terraform plan

# Apply infrastructure (takes ~15 minutes)
terraform apply

# Configure kubectl
aws eks update-kubeconfig --region us-east-2 --name ddp --profile self

# Verify cluster access
kubectl get nodes
```

## Infrastructure Overview

```
┌─────────────────────────────────────────────┐
│           EKS Control Plane                 │
│         (Kubernetes 1.34)                   │
└─────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
┌───────▼──────┐       ┌────────▼────────┐
│  Public      │       │  Karpenter      │
│  Subnets     │       │  Controller     │
│              │       │  (1x t3.small)  │
└──────────────┘       └─────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │  Karpenter Manages  │
                    │  Workload Nodes     │
                    └─────────────────────┘
```

## Karpenter Setup

After infrastructure is deployed, install Karpenter:

```bash
# Install Karpenter Helm chart
helm upgrade --install karpenter oci://public.ecr.aws/karpenter/karpenter \
  --version 1.1.1 \
  --namespace kube-system \
  --set settings.clusterName=ddp \
  --set serviceAccount.name=karpenter \
  --set controller.resources.requests.cpu=1 \
  --set controller.resources.requests.memory=1Gi
```

## Karpenter NodePool Examples

### GPU Nodes (for training)

Save as `gpu-nodepool.yaml`:

```yaml
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: gpu
spec:
  disruption:
    consolidationPolicy: WhenEmpty
    consolidateAfter: 30s

  template:
    spec:
      requirements:
      - key: karpenter.sh/capacity-type
        operator: In
        values: ["spot"]
      - key: node.kubernetes.io/instance-type
        operator: In
        values: ["g5.xlarge", "g5.2xlarge"]
      - key: kubernetes.io/arch
        operator: In
        values: ["amd64"]

      nodeClassRef:
        group: karpenter.k8s.aws
        kind: EC2NodeClass
        name: gpu

      taints:
      - key: nvidia.com/gpu
        effect: NoSchedule
        value: "true"

---
apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: gpu
spec:
  amiSelectorTerms:
  - alias: al2@latest

  role: # Will be filled by Terraform output

  subnetSelectorTerms:
  - tags:
      karpenter.sh/discovery: ddp

  securityGroupSelectorTerms:
  - tags:
      karpenter.sh/discovery: ddp

  blockDeviceMappings:
  - deviceName: /dev/xvda
    ebs:
      volumeSize: 100Gi
      volumeType: gp3
      encrypted: true
      deleteOnTermination: true
```

### CPU Nodes (general workloads)

Save as `cpu-nodepool.yaml`:

```yaml
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: cpu
spec:
  disruption:
    consolidationPolicy: WhenEmpty
    consolidateAfter: 30s

  template:
    spec:
      requirements:
      - key: karpenter.sh/capacity-type
        operator: In
        values: ["spot"]
      - key: node.kubernetes.io/instance-type
        operator: In
        values: ["t3.large", "t3.xlarge", "c6i.xlarge"]
      - key: kubernetes.io/arch
        operator: In
        values: ["amd64"]

      nodeClassRef:
        group: karpenter.k8s.aws
        kind: EC2NodeClass
        name: cpu

---
apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: cpu
spec:
  amiSelectorTerms:
  - alias: al2023@latest

  role: # Will be filled by Terraform output

  subnetSelectorTerms:
  - tags:
      karpenter.sh/discovery: ddp

  securityGroupSelectorTerms:
  - tags:
      karpenter.sh/discovery: ddp

  blockDeviceMappings:
  - deviceName: /dev/xvda
    ebs:
      volumeSize: 50Gi
      volumeType: gp3
      encrypted: true
      deleteOnTermination: true
```

### Apply NodePools

After Terraform apply completes:

```bash
# Get the IAM role name from Terraform outputs
ROLE_NAME=$(terraform output -raw karpenter_node_iam_role_name)

# Update the NodePool YAML files with the role name
sed -i "s|role:.*|role: $ROLE_NAME|" gpu-nodepool.yaml
sed -i "s|role:.*|role: $ROLE_NAME|" cpu-nodepool.yaml

# Apply the NodePools
kubectl apply -f cpu-nodepool.yaml
kubectl apply -f gpu-nodepool.yaml
```

## Example Workload

Deploy a test GPU workload:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
spec:
  tolerations:
  - key: nvidia.com/gpu
    operator: Equal
    value: "true"
    effect: NoSchedule

  containers:
  - name: cuda-test
    image: nvidia/cuda:12.3.0-base-ubuntu22.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
```

Karpenter will automatically provision a GPU node when this pod is created!

## Terraform Outputs

After `terraform apply`, get useful information:

```bash
# Configure kubectl command
terraform output -raw configure_kubectl

# Get Karpenter IAM role for NodePools
terraform output -raw karpenter_node_iam_role_name

# View all outputs
terraform output
```

## Troubleshooting

### Nodes not joining cluster
```bash
# Check Karpenter logs
kubectl logs -n kube-system -l app.kubernetes.io/name=karpenter -f

# Verify NodePools
kubectl get nodepools

# Check EC2NodeClass
kubectl get ec2nodeclasses
```

### Pods stuck in Pending
```bash
# Check Karpenter events
kubectl get events -n kube-system --sort-by='.lastTimestamp'

# Verify pod resource requests are within instance capacity
kubectl describe pod <pod-name>
```

## Clean Up

```bash
# Delete all Karpenter-managed nodes first
kubectl delete nodepools --all

# Wait for nodes to terminate (check AWS console)
# Then destroy infrastructure
terraform destroy
```

## Security Notes

- ✅ IMDSv2 enforced on all nodes
- ✅ EBS volumes encrypted
- ✅ Security groups configured for least privilege
- ✅ Public subnets (NAT disabled)
  - **Security**: Nodes have public IPs but:
    - Security groups restrict inbound traffic
    - Only cluster-related ports open
    - Internet access for ECR/S3 pulls

To enable private networking:
1. Set `enable_nat_gateway = true` in `main.tf`
2. `terraform apply`

## Files

- **main.tf** - Infrastructure configuration
- **outputs.tf** - Terraform outputs
- **README.md** - This file

## Support

For issues or questions, see:
- [EKS Best Practices](https://aws.github.io/aws-eks-best-practices/)
- [Karpenter Documentation](https://karpenter.sh/)
- [Terraform AWS Modules](https://github.com/terraform-aws-modules/terraform-aws-eks)
