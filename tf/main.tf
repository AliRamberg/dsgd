terraform {
  required_version = ">= 1.5.7"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 3.1"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 3.0"
    }
  }

}

provider "aws" {
  region  = local.region
  profile = "self"
}

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  region = "us-east-2"
  name   = "ddp"

  vpc_cidr = "10.0.0.0/16"
  azs      = slice(data.aws_availability_zones.available.names, 0, 2)

  enable_nat_gateway = false
  single_az          = false

  tags = {
    Project     = "DDP"
    Environment = "dev"
    ManagedBy   = "Terraform"
    GithubRepo  = "ddp"
  }
}

################################################################################
# VPC Module
################################################################################

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 6.0"

  name = "${local.name}-vpc"
  cidr = local.vpc_cidr

  azs             = local.single_az ? [local.azs[0]] : local.azs
  private_subnets = local.enable_nat_gateway ? [for k, v in(local.single_az ? [local.azs[0]] : local.azs) : cidrsubnet(local.vpc_cidr, 4, k)] : []
  public_subnets  = [for k, v in(local.single_az ? [local.azs[0]] : local.azs) : cidrsubnet(local.vpc_cidr, 4, k)]

  enable_nat_gateway   = local.enable_nat_gateway
  single_nat_gateway   = true
  enable_dns_hostnames = true
  enable_dns_support   = true

  # When NAT is disabled, nodes will use public subnets
  map_public_ip_on_launch = !local.enable_nat_gateway

  # Manage default VPC resources to prevent accidental usage
  manage_default_vpc               = true
  default_vpc_enable_dns_hostnames = false
  default_vpc_enable_dns_support   = true
  default_vpc_name                 = "default-vpc-disabled"
  default_vpc_tags = {
    Name      = "Default VPC (Managed - Do Not Use)"
    ManagedBy = "Terraform"
    Status    = "Disabled"
  }

  # Tags required for EKS
  public_subnet_tags = {
    "kubernetes.io/role/elb"              = "1"
    "kubernetes.io/cluster/${local.name}" = "shared"
    # Allow nodes in public subnet when NAT is disabled
    "kubernetes.io/role/internal-elb" = local.enable_nat_gateway ? "0" : "1"
    # Karpenter subnet discovery
    "karpenter.sh/discovery" = local.name
  }

  private_subnet_tags = local.enable_nat_gateway ? {
    "kubernetes.io/role/internal-elb"     = "1"
    "kubernetes.io/cluster/${local.name}" = "shared"
  } : {}

  tags = local.tags
}

################################################################################
# EKS Module
################################################################################

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 21.0"

  name               = local.name
  kubernetes_version = "1.34"

  # Cluster access configuration
  endpoint_public_access  = true
  endpoint_private_access = true

  # Enable cluster creator admin permissions (v21+ best practice)
  enable_cluster_creator_admin_permissions = true

  # VPC and subnet configuration
  vpc_id     = module.vpc.vpc_id
  subnet_ids = local.enable_nat_gateway ? module.vpc.private_subnets : module.vpc.public_subnets

  # Cluster addons with latest versions
  addons = {
    coredns = {
      most_recent = true
    }
    eks-pod-identity-agent = {
      most_recent    = true
      before_compute = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent    = true
      before_compute = true
      configuration_values = jsonencode({
        env = {
          ENABLE_PREFIX_DELEGATION = "true"
          WARM_PREFIX_TARGET       = "1"
        }
      })
    }
    aws-ebs-csi-driver = {
      most_recent                 = true
      service_account_role_arn    = module.ebs_csi_driver_irsa.arn
      resolve_conflicts_on_create = "OVERWRITE"
      resolve_conflicts_on_update = "OVERWRITE"
    }
  }

  # Minimal managed node group for Karpenter controller
  eks_managed_node_groups = {
    karpenter = {
      name            = "${local.name}-karpenter-ng"
      use_name_prefix = true

      ami_type       = "BOTTLEROCKET_ARM_64"
      instance_types = ["t4g.medium"]

      min_size     = 1
      max_size     = 2
      desired_size = 2

      labels = {
        "karpenter.sh/controller" = "true"
      }

      # Removed taints to allow all workloads to schedule on managed node
      # taints = {
      #   karpenter = {
      #     key    = "CriticalAddonsOnly"
      #     value  = "true"
      #     effect = "NO_SCHEDULE"
      #   }
      # }

      # IMDSv2 configuration for enhanced security
      metadata_options = {
        http_endpoint               = "enabled"
        http_tokens                 = "required"
        http_put_response_hop_limit = 2
        instance_metadata_tags      = "enabled"
      }
    }
  }

  # Cluster security group rules
  security_group_additional_rules = {
    egress_nodes_ephemeral_ports_tcp = {
      description                = "Cluster to node on ephemeral ports"
      protocol                   = "tcp"
      from_port                  = 1025
      to_port                    = 65535
      type                       = "egress"
      source_node_security_group = true
    }
  }

  # Node security group rules
  node_security_group_additional_rules = {
    ingress_self_all = {
      description = "Node to node all ports/protocols"
      protocol    = "-1"
      from_port   = 0
      to_port     = 0
      type        = "ingress"
      self        = true
    }

    ingress_cluster_all = {
      description                   = "Cluster to node all ports/protocols"
      protocol                      = "-1"
      from_port                     = 0
      to_port                       = 0
      type                          = "ingress"
      source_cluster_security_group = true
    }

    egress_all = {
      description      = "Node all egress"
      protocol         = "-1"
      from_port        = 0
      to_port          = 0
      type             = "egress"
      cidr_blocks      = ["0.0.0.0/0"]
      ipv6_cidr_blocks = ["::/0"]
    }
  }

  # Tag security group for Karpenter discovery
  node_security_group_tags = {
    "karpenter.sh/discovery" = local.name
  }

  # Minimal cluster logging
  enabled_log_types = ["api"]

  # CloudWatch log group settings
  create_cloudwatch_log_group            = true
  cloudwatch_log_group_retention_in_days = 1

  tags = local.tags
}

################################################################################
# Karpenter Module
################################################################################

#
# Spot prerequisites
#
# If the account has never used EC2 Spot before, the Spot service-linked role may not exist.
# Karpenter may attempt to create it and fail with:
#   AuthFailure.ServiceLinkedRoleCreationNotPermitted
#
# Best practice: manage the service-linked role as part of account/cluster bootstrap
# rather than granting Karpenter iam:CreateServiceLinkedRole.
#
resource "aws_iam_service_linked_role" "ec2_spot" {
  aws_service_name = "spot.amazonaws.com"
  description      = "Service-linked role for EC2 Spot Instances (required for Spot fleets)"
}

module "karpenter" {
  source  = "terraform-aws-modules/eks/aws//modules/karpenter"
  version = "~> 21.0"

  cluster_name = module.eks.cluster_name

  # Enable Pod Identity for Karpenter (v21+ best practice)
  create_pod_identity_association = true

  # Attach additional IAM policies to the Karpenter node IAM role
  node_iam_role_additional_policies = {
    AmazonSSMManagedInstanceCore = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
  }

  tags = local.tags
}

################################################################################
# EBS CSI Driver IAM Role
################################################################################

module "ebs_csi_driver_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts"
  version = "~> 6.2"

  name            = "ebs-csi-driver"
  use_name_prefix = true

  attach_ebs_csi_policy = true

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:ebs-csi-controller-sa"]
    }
  }

  tags = local.tags
}
