################################################################################
# S3 Bucket for Datasets
################################################################################

resource "aws_s3_bucket" "datasets" {
  bucket = "yahli-asap-ddp"

  tags = merge(local.tags, {
    Name        = "yahli-asap-ddp"
    Description = "Storage for distributed training datasets"
  })
}

resource "aws_s3_bucket_versioning" "datasets" {
  bucket = aws_s3_bucket.datasets.id

  versioning_configuration {
    status = "Disabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "datasets" {
  bucket = aws_s3_bucket.datasets.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "datasets" {
  bucket = aws_s3_bucket.datasets.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Allow EKS worker nodes to read from bucket
resource "aws_iam_policy" "s3_dataset_read" {
  name        = "ddp-s3-dataset-read"
  description = "Allow reading datasets from S3 bucket"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.datasets.arn,
          "${aws_s3_bucket.datasets.arn}/*"
        ]
      }
    ]
  })
}

# Attach policy to Karpenter node IAM role so pods can read from S3
resource "aws_iam_role_policy_attachment" "karpenter_s3_read" {
  role       = module.karpenter.node_iam_role_name
  policy_arn = aws_iam_policy.s3_dataset_read.arn
}

################################################################################
# Outputs
################################################################################

output "s3_bucket_name" {
  description = "Name of the S3 bucket for datasets"
  value       = aws_s3_bucket.datasets.id
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket for datasets"
  value       = aws_s3_bucket.datasets.arn
}
