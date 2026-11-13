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
