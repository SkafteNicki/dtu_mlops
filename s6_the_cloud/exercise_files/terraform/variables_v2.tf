variable "gcp_project_id" {
  description = "The GCP project ID"
  type        = string
  default     = "dtu-mlops-2026"
}

variable "region" {
  description = "The GCP region for resources"
  type        = string
  default     = "europe-west1"
}

variable "bucket_name" {
  description = "The name of the storage bucket"
  type        = string
  default     = "dtu-mlops-2026-infra-as-code-bucket-<random-numbers>"
}
