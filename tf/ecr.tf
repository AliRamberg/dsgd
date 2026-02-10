################################################################################
# ECR Repository for Ray Worker Images
################################################################################

module "ecr" {
  source  = "terraform-aws-modules/ecr/aws"
  version = "~> 3.1"

  repository_name = "ddp"

  # Allow image tag mutability for development
  repository_image_tag_mutability = "MUTABLE"

  # Lifecycle policy to clean up old images
  repository_lifecycle_policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 tagged images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v", "latest"]
          countType     = "imageCountMoreThan"
          countNumber   = 10
        }
        action = {
          type = "expire"
        }
      },
      {
        rulePriority = 2
        description  = "Delete untagged images after 7 days"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 7
        }
        action = {
          type = "expire"
        }
      }
    ]
  })

  # Enable scan on push for security
  repository_image_scan_on_push = true

  # Grant read/write access to EKS node IAM roles
  repository_read_write_access_arns = [
    module.eks.eks_managed_node_groups["karpenter"].iam_role_arn,
    module.karpenter.node_iam_role_arn
  ]

  tags = local.tags
}
