# Exercise 2: Basic configuration with hardcoded values
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
  project = "dtu-mlops-2026"
  region  = "europe-west1"
}

resource "google_storage_bucket" "my_bucket" {
  name          = "dtu-mlops-2026-infra-as-code-bucket-<random-numbers>"
  location      = "EU"
  force_destroy = true

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }
}
