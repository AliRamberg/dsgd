################################################################################
# Kubernetes Provider for Karpenter Configurations
################################################################################

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args = [
      "eks",
      "get-token",
      "--cluster-name",
      module.eks.cluster_name,
      "--region",
      local.region,
      "--profile",
      "self"
    ]
  }
}

################################################################################
# Karpenter Default NodePool (CPU)
################################################################################

resource "kubernetes_manifest" "karpenter_nodepool_default" {
  manifest = {
    apiVersion = "karpenter.sh/v1"
    kind       = "NodePool"
    metadata = {
      name = "default"
    }
    spec = {
      template = {
        spec = {
          nodeClassRef = {
            group = "karpenter.k8s.aws"
            kind  = "EC2NodeClass"
            name  = "default"
          }
          requirements = [
            {
              key      = "karpenter.k8s.aws/instance-generation"
              operator = "Gt"
              values   = ["3"] # Require generation 4+ (Graviton 2+), excludes a1 (Graviton 1)
            },
            {
              key      = "karpenter.k8s.aws/instance-size"
              operator = "NotIn"
              values   = ["nano", "micro", "small"]
            },
            {
              key      = "kubernetes.io/arch"
              operator = "In"
              values   = ["amd64", "arm64"]
            },
            {
              key      = "karpenter.sh/capacity-type"
              operator = "In"
              values   = ["spot", "on-demand"]
            }
          ]
        }
      }
      limits = {
        cpu    = "100"
        memory = "400Gi"
      }
      disruption = {
        consolidationPolicy = "WhenEmptyOrUnderutilized"
        consolidateAfter    = "1m"
      }
      weight = 10
    }
  }

  depends_on = [
    helm_release.karpenter
  ]
}

################################################################################
# Karpenter Default EC2NodeClass (CPU)
################################################################################

resource "kubernetes_manifest" "karpenter_ec2nodeclass_default" {
  manifest = {
    apiVersion = "karpenter.k8s.aws/v1"
    kind       = "EC2NodeClass"
    metadata = {
      name = "default"
    }
    spec = {
      amiSelectorTerms = [
        {
          alias = "bottlerocket@latest"
        }
      ]
      amiFamily = "Bottlerocket"
      # Use dynamic reference to Karpenter node IAM role
      role = module.karpenter.node_iam_role_name
      subnetSelectorTerms = [
        {
          tags = {
            "karpenter.sh/discovery" = local.name
          }
        }
      ]
      securityGroupSelectorTerms = [
        {
          tags = {
            "karpenter.sh/discovery" = local.name
          }
        },
        {
          # Include EKS cluster primary security group for node-to-control-plane communication
          id = module.eks.cluster_primary_security_group_id
        }
      ]
      userData = <<-EOT
        [settings.kubernetes]
        cluster-name = "${module.eks.cluster_name}"
        # Enable high pod density with prefix delegation
        max-pods = 110
      EOT
      blockDeviceMappings = [
        {
          deviceName = "/dev/xvda"
          ebs = {
            volumeSize          = "100Gi"
            volumeType          = "gp3"
            encrypted           = true
            deleteOnTermination = true
          }
        }
      ]
      metadataOptions = {
        httpEndpoint            = "enabled"
        httpProtocolIPv6        = "disabled"
        httpPutResponseHopLimit = 2
        httpTokens              = "required"
      }
      tags = merge(
        local.tags,
        {
          Name                     = "karpenter-node"
          ManagedBy                = "Karpenter"
          "karpenter.sh/discovery" = local.name
        }
      )
    }
  }

  depends_on = [
    helm_release.karpenter
  ]
}

################################################################################
# Karpenter GPU NodePool
################################################################################

resource "kubernetes_manifest" "karpenter_nodepool_gpu" {
  field_manager {
    force_conflicts = true
  }

  manifest = {
    apiVersion = "karpenter.sh/v1"
    kind       = "NodePool"
    metadata = {
      name = "gpu"
    }
    spec = {
      template = {
        spec = {
          nodeClassRef = {
            group = "karpenter.k8s.aws"
            kind  = "EC2NodeClass"
            name  = "gpu"
          }
          requirements = [
            {
              key      = "karpenter.k8s.aws/instance-gpu-manufacturer"
              operator = "In"
              values   = ["nvidia"]
            },
            {
              key      = "kubernetes.io/arch"
              operator = "In"
              values   = ["amd64"]
            },
            {
              key      = "karpenter.sh/capacity-type"
              operator = "In"
              values   = ["on-demand", "spot"]
            }
          ]
          taints = [
            {
              key    = "nvidia.com/gpu"
              effect = "NoSchedule"
            }
          ]
        }
      }
      limits = {
        cpu    = "1000"
        memory = "2000Gi"
      }
      disruption = {
        consolidationPolicy = "WhenEmpty"
        consolidateAfter    = "5m"
      }
      weight = 5
    }
  }

  depends_on = [
    helm_release.karpenter
  ]
}

################################################################################
# Karpenter GPU EC2NodeClass
################################################################################

resource "kubernetes_manifest" "karpenter_ec2nodeclass_gpu" {
  field_manager {
    force_conflicts = true
  }

  manifest = {
    apiVersion = "karpenter.k8s.aws/v1"
    kind       = "EC2NodeClass"
    metadata = {
      name = "gpu"
    }
    spec = {
      amiSelectorTerms = [
        {
          name = "bottlerocket-aws-k8s-1.34-nvidia-aarch64-*"
        },
        {
          name = "bottlerocket-aws-k8s-1.34-nvidia-x86_64-*"
        }
      ]
      amiFamily = "Bottlerocket"
      # Use dynamic reference to Karpenter node IAM role
      role = module.karpenter.node_iam_role_name
      subnetSelectorTerms = [
        {
          tags = {
            "karpenter.sh/discovery" = local.name
          }
        }
      ]
      securityGroupSelectorTerms = [
        {
          tags = {
            "karpenter.sh/discovery" = local.name
          }
        },
        {
          # Include EKS cluster primary security group for node-to-control-plane communication
          id = module.eks.cluster_primary_security_group_id
        }
      ]
      userData = <<-EOT
        [settings.kubernetes]
        cluster-name = "${module.eks.cluster_name}"
        # Enable high pod density with prefix delegation
        max-pods = 110
      EOT
      blockDeviceMappings = [
        {
          deviceName = "/dev/xvda"
          ebs = {
            volumeSize          = "100Gi"
            volumeType          = "gp3"
            iops                = 3000
            throughput          = 125
            encrypted           = true
            deleteOnTermination = true
          }
        }
      ]
      metadataOptions = {
        httpEndpoint            = "enabled"
        httpProtocolIPv6        = "disabled"
        httpPutResponseHopLimit = 2
        httpTokens              = "required"
      }
      tags = merge(
        local.tags,
        {
          Name                     = "karpenter-gpu-node"
          ManagedBy                = "Karpenter"
          NodeType                 = "gpu"
          "karpenter.sh/discovery" = local.name
        }
      )
    }
  }

  depends_on = [
    helm_release.karpenter
  ]
}
