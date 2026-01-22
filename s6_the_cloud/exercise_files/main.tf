terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
  }
  }
}

provider "google" {
    project = "<YOUR_GCP_PROJECT_ID>"
    region  = "eu-west1s-central1"
}
