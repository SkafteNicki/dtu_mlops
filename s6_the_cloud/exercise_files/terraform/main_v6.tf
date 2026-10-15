# Exercise 9: Adding Cloud Build service account and IAM configuration
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
resource "google_project_service" "artifact_registry_api" {
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

# Create Artifact Registry repository for Docker images
resource "google_artifact_registry_repository" "docker_repo" {
  depends_on = [google_project_service.artifact_registry_api]

  location      = var.region
  repository_id = "${var.gcp_project_id}-${var.artifact_registry_id}"
  description   = "Docker repository for ML training images"
  format        = "DOCKER"

  cleanup_policy_dry_run = false
}

# Enable Cloud Build API
# This API is required for automated container builds in GCP
resource "google_project_service" "cloud_build_api" {
  service            = "cloudbuild.googleapis.com"
  disable_on_destroy = false
}

# Create service account for Cloud Build
# Service accounts are special accounts used by services (not humans) to authenticate
# This follows the principle of least privilege - giving only necessary permissions
resource "google_service_account" "cloud_build_sa" {
  account_id   = "cloud-build-sa"
  display_name = "Service Account for Cloud Build"

  depends_on = [google_project_service.cloud_build_api]
}

# Grant Cloud Build service account permission to push to Artifact Registry
# This IAM binding allows our Cloud Build service account to write Docker images
# to the artifact registry we created in the previous exercise
resource "google_artifact_registry_repository_iam_member" "cloud_build_push" {
  project    = var.gcp_project_id
  location   = google_artifact_registry_repository.docker_repo.location
  repository = google_artifact_registry_repository.docker_repo.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.cloud_build_sa.email}"

  depends_on = [
    google_artifact_registry_repository.docker_repo,
    google_service_account.cloud_build_sa
  ]
}
