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
