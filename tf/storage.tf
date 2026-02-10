################################################################################
# Storage Classes
################################################################################

# gp3 Storage Class (default) - modern, faster, cheaper than gp2
resource "kubernetes_storage_class_v1" "gp3" {
  metadata {
    name = "gp3"
    annotations = {
      "storageclass.kubernetes.io/is-default-class" = "true"
    }
  }

  storage_provisioner    = "ebs.csi.aws.com"
  reclaim_policy         = "Retain" # Changed from Delete to prevent data loss
  allow_volume_expansion = true
  volume_binding_mode    = "WaitForFirstConsumer"

  parameters = {
    type       = "gp3"
    fsType     = "ext4"
    iops       = "3000" # Default baseline (no extra cost up to 3000)
    throughput = "125"  # Default baseline in MiB/s (no extra cost up to 125)
  }

  depends_on = [
    module.eks.aws_eks_addon
  ]
}

# Remove default annotation from gp2 if it exists
resource "kubernetes_annotations" "gp2_remove_default" {
  api_version = "storage.k8s.io/v1"
  kind        = "StorageClass"
  metadata {
    name = "gp2"
  }

  annotations = {
    "storageclass.kubernetes.io/is-default-class" = "false"
  }

  force = true

  depends_on = [
    kubernetes_storage_class_v1.gp3
  ]
}
