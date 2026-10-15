# Exercise 11: Adding Cloud Run Deployment Infrastructure
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
resource "google_project_service" "cloud_build_api" {
  service            = "cloudbuild.googleapis.com"
  disable_on_destroy = false
}

# Create service account for Cloud Build
resource "google_service_account" "cloud_build_sa" {
  account_id   = "cloud-build-sa"
  display_name = "Service Account for Cloud Build"

  depends_on = [google_project_service.cloud_build_api]
}

# Grant Cloud Build service account permission to push to Artifact Registry
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

# ============================================================================
# VERTEX AI INFRASTRUCTURE (FROM EXERCISE 10)
# ============================================================================

# Enable Vertex AI API
# This API is required for running custom training jobs on Vertex AI
resource "google_project_service" "vertex_ai_api" {
  service            = "aiplatform.googleapis.com"
  disable_on_destroy = false
}

# Create service account for Vertex AI training jobs
# This service account will be used by Vertex AI to access various GCP resources
# when running custom training jobs (pulling containers, accessing data, writing logs)
resource "google_service_account" "vertex_ai_sa" {
  account_id   = "vertex-ai-training-sa"
  display_name = "Service Account for Vertex AI Training"

  depends_on = [google_project_service.vertex_ai_api]
}

# Grant Vertex AI service account permission to pull Docker images from Artifact Registry
# This allows Vertex AI to pull the training container images we built in Exercise 8
# Note: We use "reader" role (not "writer") because Vertex AI only needs to pull images
resource "google_artifact_registry_repository_iam_member" "vertex_ai_pull" {
  project    = var.gcp_project_id
  location   = google_artifact_registry_repository.docker_repo.location
  repository = google_artifact_registry_repository.docker_repo.name
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.vertex_ai_sa.email}"

  depends_on = [
    google_artifact_registry_repository.docker_repo,
    google_service_account.vertex_ai_sa
  ]
}

# Grant Vertex AI service account access to Cloud Storage
# This enables two important capabilities:
# 1. Reading training data from the bucket (via mounted /gcs/ filesystem)
# 2. Writing model checkpoints and outputs back to the bucket
# Note: We use "objectAdmin" to allow both read and write on objects
resource "google_storage_bucket_iam_member" "vertex_ai_data_access" {
  bucket = google_storage_bucket.my_bucket.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.vertex_ai_sa.email}"

  depends_on = [
    google_storage_bucket.my_bucket,
    google_service_account.vertex_ai_sa
  ]
}

# Grant Vertex AI service account permission to run custom training jobs
# This is a project-level permission (not resource-specific like the above)
# The "aiplatform.user" role allows the service account to:
# - Create and manage custom training jobs
# - Access Vertex AI resources
# - Submit jobs to the training service
resource "google_project_iam_member" "vertex_ai_user" {
  project = var.gcp_project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.vertex_ai_sa.email}"

  depends_on = [google_service_account.vertex_ai_sa]
}

# Grant Vertex AI service account permission to write logs
# Training jobs generate logs that need to be written to Cloud Logging
# This permission allows the service account to write logs so you can:
# - Debug training issues
# - Monitor training progress
# - View stdout/stderr from your training scripts
resource "google_project_iam_member" "vertex_ai_logs" {
  project = var.gcp_project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.vertex_ai_sa.email}"

  depends_on = [google_service_account.vertex_ai_sa]
}

# Optional: Create a dedicated bucket for training configurations and outputs
# This keeps training artifacts separate from general data storage
# Best practice for organizing ML experiments
resource "google_storage_bucket" "training_configs" {
  name          = "${var.bucket_name}-training-configs"
  location      = "EU"
  force_destroy = true

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }
}

# Grant Vertex AI service account full access to training configs bucket
# Using objectAdmin role to allow reading configs and writing training outputs
resource "google_storage_bucket_iam_member" "vertex_ai_configs_access" {
  bucket = google_storage_bucket.training_configs.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.vertex_ai_sa.email}"

  depends_on = [
    google_storage_bucket.training_configs,
    google_service_account.vertex_ai_sa
  ]
}

# ============================================================================
# CLOUD RUN INFRASTRUCTURE (NEW IN EXERCISE 11)
# ============================================================================

# Enable Cloud Run API
# This API is required for deploying serverless containers to Cloud Run
# Cloud Run is Google's serverless platform that automatically scales your
# containerized applications based on incoming requests
resource "google_project_service" "cloud_run_api" {
  service            = "run.googleapis.com"
  disable_on_destroy = false
}

# Create service account for Cloud Run deployments
# This service account will be used by Cloud Run services to access GCP resources
# when running your deployed containers (pulling images, accessing storage, etc.)
# Important: This is different from:
# - cloud_build_sa: Used during BUILD time to push images to Artifact Registry
# - vertex_ai_sa: Used during TRAINING to run ML jobs
# - cloud_run_sa: Used during DEPLOYMENT/RUNTIME to serve ML inference APIs
resource "google_service_account" "cloud_run_sa" {
  account_id   = "cloud-run-sa"
  display_name = "Service Account for Cloud Run Deployments"

  depends_on = [google_project_service.cloud_run_api]
}

# Grant Cloud Run service account permission to pull Docker images from Artifact Registry
# When you deploy a Cloud Run service with a container image from Artifact Registry,
# Cloud Run needs permission to pull that image during deployment and when scaling up
# Note: We use "reader" role (not "writer") because Cloud Run only pulls images,
# it doesn't push them. This is the same role we use for Vertex AI.
# Comparison:
# - Cloud Build: artifactregistry.writer (pushes newly built images)
# - Vertex AI: artifactregistry.reader (pulls images for training)
# - Cloud Run: artifactregistry.reader (pulls images for deployment)
resource "google_artifact_registry_repository_iam_member" "cloud_run_pull" {
  project    = var.gcp_project_id
  location   = google_artifact_registry_repository.docker_repo.location
  repository = google_artifact_registry_repository.docker_repo.name
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.cloud_run_sa.email}"

  depends_on = [
    google_artifact_registry_repository.docker_repo,
    google_service_account.cloud_run_sa
  ]
}

# Optional: Grant Cloud Run service account access to Cloud Storage
# Uncomment this if your deployed Cloud Run services need to:
# - Read model weights from storage buckets
# - Write prediction results or logs to storage
# - Access any other data stored in GCS
# This is commonly needed for ML inference APIs that load models from storage
# resource "google_storage_bucket_iam_member" "cloud_run_storage_access" {
#   bucket = google_storage_bucket.my_bucket.name
#   role   = "roles/storage.objectViewer"  # Read-only access
#   member = "serviceAccount:${google_service_account.cloud_run_sa.email}"
#
#   depends_on = [
#     google_storage_bucket.my_bucket,
#     google_service_account.cloud_run_sa
#   ]
# }
