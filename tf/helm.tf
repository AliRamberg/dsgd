################################################################################
# Helm Provider Configuration
################################################################################

provider "helm" {
  kubernetes = {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    exec = {
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
}

################################################################################
# Karpenter Helm Release
################################################################################

resource "helm_release" "karpenter" {
  name       = "karpenter"
  repository = "oci://public.ecr.aws/karpenter"
  chart      = "karpenter"
  version    = "1.8.3"
  namespace  = "kube-system"

  values = [
    templatefile("${path.module}/karpenter-values.yaml", {
      cluster_name     = module.eks.cluster_name
      cluster_endpoint = module.eks.cluster_endpoint
      queue_name       = module.karpenter.queue_name
    })
  ]

  depends_on = [
    module.eks.eks_managed_node_groups,
    module.karpenter
  ]
}

################################################################################
# Prometheus Operator Helm Release
################################################################################

resource "helm_release" "prometheus" {
  name       = "prometheus"
  repository = "oci://ghcr.io/prometheus-community/charts"
  chart      = "kube-prometheus-stack"
  version    = "80.13.3"
  namespace  = "prometheus-system"

  create_namespace = true

  values = [
    file("${path.module}/prometheus-values.yaml")
  ]

  depends_on = [
    helm_release.karpenter,
    kubernetes_manifest.karpenter_nodepool_default,
    kubernetes_manifest.karpenter_ec2nodeclass_default
  ]
}

################################################################################
# KubeRay Operator Helm Release
################################################################################

resource "helm_release" "kuberay_operator" {
  name       = "kuberay-operator"
  repository = "https://ray-project.github.io/kuberay-helm/"
  chart      = "kuberay-operator"
  version    = "1.5.1"
  namespace  = "ray-operator"

  create_namespace = true

  values = [
    file("${path.module}/ray-values.yaml")
  ]

  depends_on = [
    helm_release.karpenter,
    kubernetes_manifest.karpenter_nodepool_default,
    kubernetes_manifest.karpenter_ec2nodeclass_default
  ]
}

################################################################################
# NVIDIA Device Plugin Helm Release
################################################################################

# resource "helm_release" "nvidia_device_plugin" {
#   name       = "nvidia-device-plugin"
#   repository = "https://nvidia.github.io/k8s-device-plugin"
#   chart      = "nvidia-device-plugin"
#   version    = "0.17.1"
#   namespace  = "kube-system"

#   values = [
#     file("${path.module}/nvidia-device-plugin-values.yaml")
#   ]

#   depends_on = [
#     helm_release.karpenter,
#     kubernetes_manifest.karpenter_nodepool_gpu,
#     kubernetes_manifest.karpenter_ec2nodeclass_gpu
#   ]
# }

################################################################################
# NVIDIA DCGM Exporter Helm Release
################################################################################

resource "helm_release" "dcgm_exporter" {
  name         = "dcgm-exporter"
  repository   = "https://nvidia.github.io/dcgm-exporter/helm-charts"
  chart        = "dcgm-exporter"
  version      = "4.7.1"
  namespace    = "gpu-monitoring"
  force_update = true

  create_namespace = true

  values = [
    file("${path.module}/dcgm-exporter-values.yaml")
  ]

  depends_on = [
    helm_release.prometheus,
    kubernetes_manifest.karpenter_nodepool_gpu,
    kubernetes_manifest.karpenter_ec2nodeclass_gpu
  ]
}
