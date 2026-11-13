# Exercise 8: Adding Artifact Registry for container storage
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 7.16.0"
    }
  }
  required_version = ">= 1.5.0"
}

provider "google" {
  project = var.gcp_project_id
  region  = var.region
}

# Storage bucket from previous exercises
resource "google_storage_bucket" "my_bucket" {
  name          = var.bucket_name
  location      = "EU"
  force_destroy = true

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }
}

# Compute instance from previous exercises
resource "google_compute_instance" "training_instance" {
  name         = var.instance_name
  machine_type = var.machine_type
  zone         = "${var.region}-a"

  boot_disk {
    initialize_params {
      image = "projects/ml-images/global/images/common-cu128-ubuntu-2404-nvidia-570-v20260129"
      size  = 50  # GB
    }
  }

  network_interface {
    network = "default"
    access_config {
      // Ephemeral public IP
    }
  }

  tags = ["mnist-training", "http-server"]

  metadata = {
    enable-oslogin = "TRUE"
  }

  service_account {
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }
}

# Enable Artifact Registry API
# This API must be enabled before we can create artifact registry repositories
resource "google_project_service" "artifact_registry_api" {
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false  # Keep API enabled even if we destroy this resource
}

# Create Artifact Registry repository for Docker images
# This is where we'll store our containerized ML applications
resource "google_artifact_registry_repository" "docker_repo" {
  # Ensure the API is enabled before creating the repository
  depends_on = [google_project_service.artifact_registry_api]

  location      = var.region
  repository_id = "${var.gcp_project_id}-${var.artifact_registry_id}"
  description   = "Docker repository for ML training images"
  format        = "DOCKER"

  # Cleanup policy to manage costs by limiting stored images
  # Note: This is configured through the cleanup_policy_dry_run flag
  # To add actual cleanup policies, you can use the GCP console or gcloud commands
  cleanup_policy_dry_run = false
}
