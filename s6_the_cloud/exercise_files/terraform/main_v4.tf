# Exercise 5: Adding compute instance
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

resource "google_storage_bucket" "my_bucket" {
  name          = var.bucket_name
  location      = "EU"
  force_destroy = true

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }
}

resource "google_compute_instance" "training_instance" {
  name         = var.instance_name
  machine_type = var.machine_type
  zone         = "${var.region}-a"

  boot_disk {
    initialize_params {
      image = "projects/debian-cloud/global/images/debian-12-bookworm-v20240110"
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
