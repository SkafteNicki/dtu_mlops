# Variables for Exercise 8: Adding Artifact Registry

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

variable "instance_name" {
  description = "The name of the compute instance"
  type        = string
  default     = "mnist-training-instance"
}

variable "machine_type" {
  description = "The machine type for the compute instance"
  type        = string
  default     = "n1-standard-4"
}

variable "artifact_registry_id" {
  description = "The ID of the artifact registry repository (will be prefixed with project ID)"
  type        = string
  default     = "docker-repo"
}
