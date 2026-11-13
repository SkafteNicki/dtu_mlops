# Outputs for Exercise 11: Adding Cloud Run Deployment Infrastructure

# Storage bucket outputs (from previous exercises)
output "bucket_name" {
  description = "The name of the created storage bucket"
  value       = google_storage_bucket.my_bucket.name
}

output "bucket_url" {
  description = "The GCS URL of the bucket"
  value       = "gs://${google_storage_bucket.my_bucket.name}"
}

output "bucket_location" {
  description = "The location of the bucket"
  value       = google_storage_bucket.my_bucket.location
}

# Compute instance outputs (from previous exercises)
output "instance_name" {
  description = "The name of the created instance"
  value       = google_compute_instance.training_instance.name
}

output "instance_internal_ip" {
  description = "The internal IP of the created instance"
  value       = google_compute_instance.training_instance.network_interface[0].network_ip
}

output "instance_external_ip" {
  description = "The external IP of the created instance"
  value       = google_compute_instance.training_instance.network_interface[0].access_config[0].nat_ip
}

output "ssh_command" {
  description = "Command to SSH into the instance"
  value       = "gcloud compute ssh ${google_compute_instance.training_instance.name} --zone=${google_compute_instance.training_instance.zone}"
}

# Artifact Registry outputs (from Exercise 8)
output "artifact_registry_repository_url" {
  description = "The URL of the Artifact Registry repository for pushing/pulling images"
  value       = "${var.region}-docker.pkg.dev/${var.gcp_project_id}/${google_artifact_registry_repository.docker_repo.repository_id}"
}

output "artifact_registry_repository_id" {
  description = "The ID of the Artifact Registry repository"
  value       = google_artifact_registry_repository.docker_repo.repository_id
}

output "artifact_registry_location" {
  description = "The location of the Artifact Registry repository"
  value       = google_artifact_registry_repository.docker_repo.location
}

# Cloud Build outputs (from Exercise 9)
output "cloud_build_service_account_email" {
  description = "Email of the Cloud Build service account (use this in Cloud Build triggers)"
  value       = google_service_account.cloud_build_sa.email
}

output "cloud_build_service_account_name" {
  description = "Full name of the Cloud Build service account"
  value       = google_service_account.cloud_build_sa.name
}

# Vertex AI outputs (from Exercise 10)
output "vertex_ai_service_account_email" {
  description = "Email of the Vertex AI training service account (use with --service-account flag in gcloud ai custom-jobs create)"
  value       = google_service_account.vertex_ai_sa.email
}

output "vertex_ai_service_account_name" {
  description = "Full name of the Vertex AI training service account"
  value       = google_service_account.vertex_ai_sa.name
}

output "training_configs_bucket_name" {
  description = "Name of the bucket for storing training configurations and outputs"
  value       = google_storage_bucket.training_configs.name
}

output "training_configs_bucket_url" {
  description = "GCS URL of the training configs bucket"
  value       = "gs://${google_storage_bucket.training_configs.name}"
}

# Cloud Run outputs (new in Exercise 11)
output "cloud_run_service_account_email" {
  description = "Email of the Cloud Run service account (use with --service-account flag in gcloud run deploy)"
  value       = google_service_account.cloud_run_sa.email
}

output "cloud_run_service_account_name" {
  description = "Full name of the Cloud Run service account"
  value       = google_service_account.cloud_run_sa.name
}

output "cloud_run_deployment_command_example" {
  description = "Example command for deploying to Cloud Run using this infrastructure (for M25)"
  value       = "gcloud run deploy <service-name> --image=${var.region}-docker.pkg.dev/${var.gcp_project_id}/${google_artifact_registry_repository.docker_repo.repository_id}/<image-name>:latest --service-account=${google_service_account.cloud_run_sa.email} --region=${var.region} --allow-unauthenticated"
}
