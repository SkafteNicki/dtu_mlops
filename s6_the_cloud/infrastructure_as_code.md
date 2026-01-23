![Logo](../figures/icons/opentofu.png){ align=right width="130"}

# Infrastructure as Code (IaC)

---

If you have worked through the previous two module, [M20 on cloud setup](cloud_setup.md) and
[M21 on using cloud services](using_the_cloud.md), you should now have a basic understanding of how to use different
cloud services to support your machine learning pipeline. You will most likely have noticed that setting up and managing
cloud resources manually either through the GCP web interface or CLI can both be time-consuming and error-prone. This is
where Infrastructure as Code (IaC) comes into play. You can see it as the cloud version of what we did in module
[M6 on coding structure](../s2_organisation_and_version_control/code_structure.md) for organizing and managing your
codebase. In that module, we learned how to use `cookiecutter` to create a projects from a template, which provided a
standardized reusable structure. Similarly, IaC allows you to define and manage your cloud infrastructure using code,
enabling you to automate the provisioning, configuration, and management of cloud resources which can be reused also
in future projects.

In this module, we will explore how to use [OpenTofu](https://opentofu.io/), an open-source IaC tool, to define and
manage our cloud infrastructure. OpenTofu allows us to write declarative configuration files that describe the desired
state of our cloud resources. By using OpenTofu, we can automate the deployment and management of our cloud
infrastructure, making it easier to scale and maintain our machine learning pipelines.

!!! note "OpenTofu vs Terraform"

    If you ever encountered the concept of Infrastructure as Code before, you might have heard the mention of
    [Terraform](https://www.terraform.io/). This has been the defacto standard for IaC for many years. However, due to
    recent licensing changes from HashiCorp (the company behind Terraform), the open-source community has forked
    Terraform and created OpenTofu as a fully open-source alternative. Because OpenTofu is a fork of Terraform, there
    should be a very high degree of compatibility between the two tools, syntax should be the same and the core concepts
    remain unchanged.

## A short introduction

The two core principles of Intfrastructure as Code are **idempotency** and **declarative configuration**.

- **Idempotency**: This means that applying the same configuration multiple times will always result in the same
    infrastructure state. For example, if you define a compute instance in your configuration and apply it, running the
    apply command again will not create a duplicate instance but will ensure that the existing instance matches the
    defined configuration. This is highly related to the previous session on
    [reproducibility](../s3_reproducibility/README.md).

- **Declarative Configuration**: Instead of writing imperative commands to create and manage resources, you define the
    desired state of your infrastructure in configuration files. E.g. instead of writing
    `gcloud compute instances create ...`, you will instead be creating a `.tf` (tf=terraform) file that describes the
    desired state of your compute instance.

Alright, then how does it work. Your job as the MLOps engineer is to write configuration files that describe the desired
state of your cloud infrastructure. The files have the extension `.tf` and in general I would recommend organizing them
in a `infrastructure` subfolder. A `.tf` file in general look something like this:

```hcl
provider "google" {
  ...
}

resource "google_compute_instance" "my_instance" {
  ...
}

data "google_compute_image" "my_image" {
  ...
}

```

In the beginning of the file, you define which cloud provider you want to use (yes, you can define multiple providers).
Then you define different resources that you want to create. Each resource has a type (e.g. `google_compute_instance`)
and a name (e.g. `my_instance`). Inside the resource block, you define different parameters that describe how you want
the resource to be configured. You can see this file as a way to structure all the `gcloud` commands you would have to
run manually to create the same resource. Finally, you can also define data sources which are read-only references to
existing resources.

After writing your configuration files, you can use the OpenTofu CLI to apply the configuration and create the resources

```bash
tofu apply
```

in your cloud account. A side effect of applying the configuration is that OpenTofu creates a **state file** that keeps
track of the current state of your infrastructure. Running `tofu apply` again will compare the desired state (your
configuration files) with the current state (the state file) and apply any necessary changes to reach the desired state.

!!! warning "Keep state file secure"

    The state file contains sensitive information (such as database passwords) and should be kept secure. When working
    in teams, it is recommended to store the state file in a remote backend (like Google Cloud Storage) rather than
    locally on your machine.

## ❔ Exercises

1. Start by installing OpenTofu on your local machine by following the instructions in the
    [official documentation](https://opentofu.org/docs/intro/install/). Verify the installation by running

    ```bash
    tofu --version
    ```

2. Initialize a new OpenTofu configuration directory in the root of your repository:

    ```bash
    tofu init
    ```

    This command creates a `.terraform` directory and downloads the necessary provider plugins.

    ??? success "Solution"

        If the command runs successfully, you should see output similar to:

        ```
        Initializing the backend...
        Initializing provider plugins...
        Terraform has been successfully initialized!
        ```

### Exercise 2: Basic Provider Configuration

For the rest of the exercises, we assume that you are using the corrupt MNIST dataset example. Create the foundational
OpenTofu configuration files to provision cloud resources.

1. In the root of the repository, create a new file called `main.tf`. This file will contain the main OpenTofu
    configuration for provisioning the necessary cloud resources.

2. Add the following code to configure the Google Cloud provider:

    ```hcl
    terraform {
      required_providers {
        google = {
          source  = "hashicorp/google"
          version = "~> 4.0"
        }
      }
    }

    provider "google" {
      project = var.gcp_project_id
      region  = var.region
    }
    ```

    !!! note "Using Variables"

        Notice we're using `var.gcp_project_id` and `var.region` instead of hardcoding values. This is a best practice
        in IaC as it makes configurations reusable across different environments.

    ??? success "Solution"

        The provider block tells OpenTofu to use the Google Cloud provider and specifies which project and region
        to use. The `required_providers` block specifies the minimum version of the provider required.

### Exercise 3: Variables and Configuration

Create a `variables.tf` file to define input variables that will be used throughout your configuration:

1. Create a new file called `variables.tf` in the root of your repository.

2. Add the following variables:

    ```hcl
    variable "gcp_project_id" {
      description = "The GCP project ID"
      type        = string
    }

    variable "region" {
      description = "The GCP region for resources"
      type        = string
      default     = "us-central1"
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
    ```

3. Create a `terraform.tfvars` file to provide values for these variables:

    ```hcl
    gcp_project_id = "YOUR_GCP_PROJECT_ID"
    region         = "us-central1"
    instance_name  = "mnist-training"
    machine_type   = "n1-standard-4"
    ```

    !!! warning "Don't commit terraform.tfvars"

        The `terraform.tfvars` file often contains sensitive information. Add it to your `.gitignore` file to prevent
        accidentally committing it to version control. Instead, you can use `terraform.tfvars.example` as a template
        for team members.

    ??? success "Solution"

        Your directory structure should now look like:

        ```
        .
        ├── main.tf
        ├── variables.tf
        ├── terraform.tfvars
        └── .terraform/
        ```

### Exercise 4: Creating a GCP Compute Instance

Now let's create an actual cloud resource. Add the following code to your `main.tf` file to create a Compute Engine
instance:

1. Add a resource block to `main.tf`:

    ```hcl
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

      tags = ["mnist-training", "http-server"]

      metadata = {
        enable-oslogin = "TRUE"
      }

      service_account {
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
      }
    }
    ```

2. Before applying this configuration, use `tofu plan` to see what changes will be made:

    ```bash
    tofu plan
    ```

    This command shows you exactly what resources will be created without making any actual changes.

3. If the plan looks correct, apply the configuration:

    ```bash
    tofu apply
    ```

    OpenTofu will ask for confirmation before creating the resources. Type `yes` to proceed.

    ??? success "Solution"

        After running `tofu apply`, you should see output showing the created resources. You can verify the instance
        was created by running:

        ```bash
        gcloud compute instances list
        ```

        Or checking the GCP Console directly.

### Exercise 5: Outputs

Create an `outputs.tf` file to extract and display important information about the created resources:

1. Create a new file called `outputs.tf` in the root of your repository.

2. Add the following output blocks:

    ```hcl
    output "instance_name" {
      description = "The name of the created instance"
      value       = google_compute_instance.training_instance.name
    }

    output "instance_id" {
      description = "The ID of the created instance"
      value       = google_compute_instance.training_instance.id
    }

    output "instance_internal_ip" {
      description = "The internal IP of the created instance"
      value       = google_compute_instance.training_instance.network_interface[0].network_ip
    }

    output "instance_external_ip" {
      description = "The external IP of the created instance"
      value       = google_compute_instance.training_instance.network_interface[0].access_config[0].nat_ip
    }
    ```

3. After adding the outputs, run:

    ```bash
    tofu apply
    ```

    The outputs will be displayed at the end of the command, and they are also stored in the state file.

4. To retrieve outputs later without applying changes, use:

    ```bash
    tofu output
    ```

    ??? success "Solution"

        The output command should show something like:

        ```
        instance_external_ip = "34.123.45.67"
        instance_id = "1234567890123456"
        instance_internal_ip = "10.128.0.2"
        instance_name = "mnist-training"
        ```

### Exercise 6: Adding a Storage Bucket

Extend your infrastructure to include a Google Cloud Storage bucket for storing data:

1. Add the following resource to your `main.tf` file:

    ```hcl
    resource "google_storage_bucket" "data_bucket" {
      name          = "${var.gcp_project_id}-mnist-data"
      location      = var.region
      force_destroy = false

      uniform_bucket_level_access = true

      versioning {
        enabled = true
      }

      lifecycle_rule {
        action {
          type = "Delete"
        }
        condition {
          age = 90  # Delete objects older than 90 days
        }
      }
    }
    ```

2. Add corresponding output to `outputs.tf`:

    ```hcl
    output "storage_bucket_name" {
      description = "The name of the created storage bucket"
      value       = google_storage_bucket.data_bucket.name
    }

    output "storage_bucket_url" {
      description = "The URL of the created storage bucket"
      value       = "gs://${google_storage_bucket.data_bucket.name}"
    }
    ```

3. Run `tofu plan` to see what will be created, then `tofu apply` to create the bucket.

    ??? success "Solution"

        Verify the bucket was created by running:

        ```bash
        gsutil ls
        ```

        Or check in the GCP Console under Cloud Storage.

### Exercise 7: State Management and Destruction

Now that you've created resources, let's understand how to manage and clean up:

1. Examine your state file by running:

    ```bash
    tofu state list
    ```

    This shows all the resources currently managed by OpenTofu.

2. To see details of a specific resource:

    ```bash
    tofu state show google_compute_instance.training_instance
    ```

3. When you no longer need the resources, you can destroy them all:

    ```bash
    tofu destroy
    ```

    OpenTofu will ask for confirmation before destroying resources.

    !!! warning "Destroy Warning"

        Be careful with `tofu destroy` in production environments. It will delete all resources managed by your
        configuration!

    ??? success "Solution"

        After running `tofu destroy`, all resources should be removed. Verify by checking the GCP Console or running:

        ```bash
        gcloud compute instances list
        gsutil ls
        ```

### Exercise 8: Module Organization

For larger infrastructure projects, it's best practice to organize your code into modules. Create a module structure:

1. Create a directory called `modules/compute_instance` in the root of your repository:

    ```bash
    mkdir -p modules/compute_instance
    ```

2. Move the compute instance configuration into the module. Create `modules/compute_instance/main.tf`:

    ```hcl
    resource "google_compute_instance" "instance" {
      name         = var.instance_name
      machine_type = var.machine_type
      zone         = var.zone

      boot_disk {
        initialize_params {
          image = var.boot_image
          size  = var.boot_disk_size
        }
      }

      tags = var.tags

      metadata = {
        enable-oslogin = "TRUE"
      }

      service_account {
        scopes = var.service_account_scopes
      }
    }
    ```

3. Create `modules/compute_instance/variables.tf`:

    ```hcl
    variable "instance_name" {
      type = string
    }

    variable "machine_type" {
      type = string
    }

    variable "zone" {
      type = string
    }

    variable "boot_image" {
      type = string
    }

    variable "boot_disk_size" {
      type    = number
      default = 50
    }

    variable "tags" {
      type    = list(string)
      default = []
    }

    variable "service_account_scopes" {
      type    = list(string)
      default = ["https://www.googleapis.com/auth/cloud-platform"]
    }
    ```

4. Create `modules/compute_instance/outputs.tf`:

    ```hcl
    output "instance_id" {
      value = google_compute_instance.instance.id
    }

    output "instance_name" {
      value = google_compute_instance.instance.name
    }

    output "internal_ip" {
      value = google_compute_instance.instance.network_interface[0].network_ip
    }

    output "external_ip" {
      value = try(google_compute_instance.instance.network_interface[0].access_config[0].nat_ip, null)
    }
    ```

5. Update your root `main.tf` to use the module:

    ```hcl
    module "training_instance" {
      source = "./modules/compute_instance"

      instance_name = var.instance_name
      machine_type  = var.machine_type
      zone          = "${var.region}-a"
      boot_image    = "projects/debian-cloud/global/images/debian-12-bookworm-v20240110"
      boot_disk_size = 50
      tags          = ["mnist-training"]
    }
    ```

6. Update `outputs.tf` to reference module outputs:

    ```hcl
    output "instance_external_ip" {
      description = "The external IP of the training instance"
      value       = module.training_instance.external_ip
    }
    ```

    ??? success "Solution"

        Your directory structure should now look like:

        ```
        .
        ├── main.tf
        ├── variables.tf
        ├── outputs.tf
        ├── terraform.tfvars
        ├── modules/
        │   └── compute_instance/
        │       ├── main.tf
        │       ├── variables.tf
        │       └── outputs.tf
        └── .terraform/
        ```

        This modular structure makes it easier to reuse the module for different purposes or share it across teams.

### Exercise 9: Conditional Resources and Local Variables

Add flexibility to your configuration using conditional resources and local variables:

1. Add to your `variables.tf`:

    ```hcl
    variable "enable_storage_bucket" {
      description = "Whether to create a storage bucket"
      type        = bool
      default     = true
    }

    variable "environment" {
      description = "The environment (dev, staging, prod)"
      type        = string
      default     = "dev"
      validation {
        condition     = contains(["dev", "staging", "prod"], var.environment)
        error_message = "Environment must be one of: dev, staging, prod"
      }
    }
    ```

2. Add local variables to your `main.tf`:

    ```hcl
    locals {
      common_tags = {
        environment = var.environment
        managed_by  = "terraform"
        created_at  = timestamp()
      }

      bucket_prefix = "${var.gcp_project_id}-${var.environment}"
    }
    ```

3. Update your storage bucket resource to use conditionals:

    ```hcl
    resource "google_storage_bucket" "data_bucket" {
      count = var.enable_storage_bucket ? 1 : 0

      name          = "${local.bucket_prefix}-mnist-data"
      location      = var.region
      force_destroy = var.environment != "prod"  # Don't auto-destroy in production

      uniform_bucket_level_access = true

      versioning {
        enabled = true
      }

      labels = local.common_tags
    }
    ```

4. Update corresponding outputs to handle the conditional:

    ```hcl
    output "storage_bucket_name" {
      description = "The name of the created storage bucket"
      value       = try(google_storage_bucket.data_bucket[0].name, null)
    }
    ```

    ??? success "Solution"

        You can now control resource creation using variables:

        ```bash
        tofu plan -var="environment=staging" -var="enable_storage_bucket=false"
        ```

        This makes your infrastructure code more flexible and reusable across different environments.

### Exercise 10: Artifact Registry for Container Storage

In this exercise, you'll create an Artifact Registry repository to store Docker container images, similar to what
you learned in [M21 Using the Cloud - Artifact Registry section](using_the_cloud.md#artifact-registry).

1. Add the following code to your `main.tf` to create an Artifact Registry repository:

    ```hcl
    # Enable required APIs
    resource "google_project_service" "artifact_registry_api" {
      service            = "artifactregistry.googleapis.com"
      disable_on_destroy = false
    }

    # Create Artifact Registry repository
    resource "google_artifact_registry_repository" "docker_repo" {
      depends_on = [google_project_service.artifact_registry_api]

      location      = var.region
      repository_id = "${var.gcp_project_id}-docker-repo"
      description   = "Docker repository for MNIST training images"
      format        = "DOCKER"

      labels = local.common_tags
    }

    # Cleanup policy to keep only the most recent 5 images
    resource "google_artifact_registry_repository" "docker_repo_cleanup" {
      depends_on = [google_artifact_registry_repository.docker_repo]

      location      = var.region
      repository_id = google_artifact_registry_repository.docker_repo.repository_id

      cleanup_policies {
        action = "KEEP"
        most_recent_versions {
          keep_count = 5
        }
      }
    }
    ```

2. Add corresponding outputs to your `outputs.tf`:

    ```hcl
    output "artifact_registry_repository_url" {
      description = "The URL of the Artifact Registry repository"
      value       = "${var.region}-docker.pkg.dev/${var.gcp_project_id}/${google_artifact_registry_repository.docker_repo.repository_id}"
    }

    output "artifact_registry_repository_id" {
      description = "The ID of the Artifact Registry repository"
      value       = google_artifact_registry_repository.docker_repo.repository_id
    }
    ```

3. Apply the configuration:

    ```bash
    tofu plan
    tofu apply
    ```

4. Verify the repository was created by listing artifacts:

    ```bash
    gcloud artifacts repositories list --location=us-central1
    ```

    ??? success "Solution"

        You should see output similar to:

        ```
        REPOSITORY            FORMAT     DESCRIPTION
        my-project-docker-repo  DOCKER     Docker repository for MNIST training images
        ```

        You can reference this repository when pushing Docker images as:
        ```
        us-central1-docker.pkg.dev/my-project-id/my-project-docker-repo
        ```

### Exercise 11: Cloud Build Configuration for CI/CD

Create infrastructure for automatically building and pushing Docker images using Cloud Build:

1. Add the Cloud Build API to your `main.tf`:

    ```hcl
    resource "google_project_service" "cloud_build_api" {
      service            = "cloudbuild.googleapis.com"
      disable_on_destroy = false
    }
    ```

2. Create a Cloud Build trigger using a configuration file. First, create a `cloud_build.tf` file:

    ```hcl
    # Create service account for Cloud Build
    resource "google_service_account" "cloud_build_sa" {
      account_id   = "cloud-build-sa"
      display_name = "Service Account for Cloud Build"

      depends_on = [google_project_service.cloud_build_api]
    }

    # Grant Cloud Build service account permission to push to Artifact Registry
    resource "google_artifact_registry_repository_iam_member" "cloud_build_push" {
      depends_on = [google_artifact_registry_repository.docker_repo]

      location   = var.region
      repository = google_artifact_registry_repository.docker_repo.name
      role       = "roles/artifactregistry.writer"
      member     = "serviceAccount:${google_service_account.cloud_build_sa.email}"
    }
    ```

3. Add outputs for the Cloud Build configuration:

    ```hcl
    output "cloud_build_service_account_email" {
      description = "Email of the Cloud Build service account"
      value       = google_service_account.cloud_build_sa.email
    }
    ```

4. Apply and verify:

    ```bash
    tofu apply
    gcloud service-accounts list
    ```

    ??? success "Solution"

        The Cloud Build infrastructure is now set up. You can reference the service account email when
        configuring triggers for automated builds. The service account has permissions to push images to
        your Artifact Registry repository.

### Exercise 12: Vertex AI Training Job Configuration (Optional Advanced)

In this advanced exercise, set up infrastructure for running training jobs on Vertex AI, building on concepts from
[M21 Using the Cloud - Training section](using_the_cloud.md#training).

1. First, enable the Vertex AI API in your `main.tf`:

    ```hcl
    resource "google_project_service" "vertex_ai_api" {
      service            = "aiplatform.googleapis.com"
      disable_on_destroy = false
    }
    ```

2. Create a service account for Vertex AI custom training jobs:

    ```hcl
    # Service account for Vertex AI training
    resource "google_service_account" "vertex_ai_sa" {
      account_id   = "vertex-ai-training-sa"
      display_name = "Service Account for Vertex AI Training"

      depends_on = [google_project_service.vertex_ai_api]
    }

    # Grant Vertex AI access to Artifact Registry
    resource "google_artifact_registry_repository_iam_member" "vertex_ai_pull" {
      depends_on = [google_artifact_registry_repository.docker_repo]

      location   = var.region
      repository = google_artifact_registry_repository.docker_repo.name
      role       = "roles/artifactregistry.reader"
      member     = "serviceAccount:${google_service_account.vertex_ai_sa.email}"
    }

    # Grant Vertex AI access to Cloud Storage for training data
    resource "google_storage_bucket_iam_member" "vertex_ai_data_access" {
      count = var.enable_storage_bucket ? 1 : 0

      bucket = google_storage_bucket.data_bucket[0].name
      role   = "roles/storage.objectViewer"
      member = "serviceAccount:${google_service_account.vertex_ai_sa.email}"
    }

    # Grant Vertex AI AI Training Agent role
    resource "google_project_iam_member" "vertex_ai_trainer" {
      project = var.gcp_project_id
      role    = "roles/aiplatform.customCodeTrainingJobRunner"
      member  = "serviceAccount:${google_service_account.vertex_ai_sa.email}"
    }

    # Grant access to write logs
    resource "google_project_iam_member" "vertex_ai_logs" {
      project = var.gcp_project_id
      role    = "roles/logging.logWriter"
      member  = "serviceAccount:${google_service_account.vertex_ai_sa.email}"
    }
    ```

3. Add a variable for the container image URI in your `variables.tf`:

    ```hcl
    variable "training_container_image" {
      description = "The Docker image URI for training jobs"
      type        = string
      default     = ""  # Leave empty to use a default or provide your own
    }
    ```

4. Create a resource for storing Vertex AI training configurations:

    ```hcl
    # Storage bucket for training configurations and outputs
    resource "google_storage_bucket" "training_configs" {
      count = var.environment == "prod" ? 1 : 0

      name          = "${local.bucket_prefix}-training-configs"
      location      = var.region
      force_destroy = false

      uniform_bucket_level_access = true

      labels = local.common_tags
    }

    # Grant Vertex AI service account access to training configs bucket
    resource "google_storage_bucket_iam_member" "vertex_ai_configs_access" {
      count = var.environment == "prod" ? 1 : 0

      bucket = google_storage_bucket.training_configs[0].name
      role   = "roles/storage.objectAdmin"
      member = "serviceAccount:${google_service_account.vertex_ai_sa.email}"
    }
    ```

5. Add outputs for Vertex AI configuration:

    ```hcl
    output "vertex_ai_service_account_email" {
      description = "Email of the Vertex AI training service account"
      value       = google_service_account.vertex_ai_sa.email
    }

    output "vertex_ai_training_config_bucket" {
      description = "Storage bucket for training configurations"
      value       = try(google_storage_bucket.training_configs[0].name, null)
    }
    ```

6. Apply the configuration:

    ```bash
    tofu apply -var="environment=prod"
    ```

7. Verify the setup by checking the service account permissions:

    ```bash
    gcloud iam service-accounts describe $(tofu output -raw vertex_ai_service_account_email)
    ```

    ??? success "Solution"

        Your infrastructure is now ready for running training jobs on Vertex AI. The service account has:
        - Access to pull images from Artifact Registry
        - Access to read training data from Cloud Storage
        - Permissions to run Vertex AI training jobs
        - Permissions to write logs for monitoring

        To use this setup with actual training jobs, you would reference the service account in your
        `gcloud ai custom-jobs create` commands or create additional Terraform resources that depend on
        these service accounts.

### Exercise 13: Remote State Backend (Optional Advanced)

For team collaboration, configure a remote backend to store the state file in Google Cloud Storage:

1. Create a `backend.tf` file:

    ```hcl
    terraform {
      backend "gcs" {
        bucket = "your-project-id-terraform-state"
        prefix = "mnist/training"
      }
    }
    ```

2. Create the GCS bucket for storing state (this needs to be done manually first or with a separate configuration):

    ```bash
    gsutil mb gs://your-project-id-terraform-state
    ```

3. Reconfigure the backend:

    ```bash
    tofu init
    ```

    OpenTofu will ask if you want to copy the existing state to the remote backend. Answer `yes`.

    ??? success "Solution"

        After configuring the remote backend, your state file will be stored in Google Cloud Storage instead of
        locally. This allows team members to work with the same infrastructure state and prevents conflicts.

        To verify:

        ```bash
        gsutil cat gs://your-project-id-terraform-state/mnist/training/default.tfstate
        ```

### Exercise 14: Cloud Run Deployment (Optional Advanced)

In this advanced exercise, you'll set up infrastructure for deploying containerized applications to Cloud Run,
a serverless container platform. This builds on the containers and artifact registry exercises from earlier.

1. First, enable the Cloud Run API in your `main.tf`:

    ```hcl
    resource "google_project_service" "cloud_run_api" {
      service            = "run.googleapis.com"
      disable_on_destroy = false
    }
    ```

2. Create a Cloud Run service that deploys from your Artifact Registry:

    ```hcl
    # Service account for Cloud Run
    resource "google_service_account" "cloud_run_sa" {
      account_id   = "cloud-run-sa"
      display_name = "Service Account for Cloud Run"

      depends_on = [google_project_service.cloud_run_api]
    }

    # Grant Cloud Run service account permission to pull from Artifact Registry
    resource "google_artifact_registry_repository_iam_member" "cloud_run_pull" {
      depends_on = [google_artifact_registry_repository.docker_repo]

      location   = var.region
      repository = google_artifact_registry_repository.docker_repo.name
      role       = "roles/artifactregistry.reader"
      member     = "serviceAccount:${google_service_account.cloud_run_sa.email}"
    }

    # Cloud Run service for MNIST inference API
    resource "google_cloud_run_service" "mnist_api" {
      name            = "mnist-inference-api"
      location        = var.region
      service_account = google_service_account.cloud_run_sa.email

      template {
        spec {
          service_account_name = google_service_account.cloud_run_sa.email

          containers {
            image = "${var.region}-docker.pkg.dev/${var.gcp_project_id}/${google_artifact_registry_repository.docker_repo.repository_id}/mnist-api:latest"

            ports {
              container_port = 8080
            }

            # Environment variables for your API
            env {
              name  = "BUCKET_NAME"
              value = try(google_storage_bucket.data_bucket[0].name, "")
            }

            env {
              name  = "ENVIRONMENT"
              value = var.environment
            }

            # Resource limits
            resources {
              limits = {
                cpu    = "1"
                memory = "512Mi"
              }
            }
          }

          # Autoscaling configuration
          autoscaling {
            max_instances = var.environment == "prod" ? 10 : 2
            min_instances = var.environment == "prod" ? 1 : 0
          }
        }
      }

      traffic {
        percent         = 100
        latest_revision = true
      }

      depends_on = [google_cloud_run_service_iam_member.cloud_run_public]
    }

    # Make the Cloud Run service publicly accessible
    resource "google_cloud_run_service_iam_member" "cloud_run_public" {
      service  = google_cloud_run_service.mnist_api.name
      location = google_cloud_run_service.mnist_api.location
      role     = "roles/run.invoker"
      member   = "allUsers"
    }
    ```

3. Add variables for Cloud Run configuration in your `variables.tf`:

    ```hcl
    variable "cloud_run_memory" {
      description = "Memory allocation for Cloud Run service (e.g., 512Mi, 1Gi)"
      type        = string
      default     = "512Mi"
    }

    variable "cloud_run_cpu" {
      description = "CPU allocation for Cloud Run service"
      type        = string
      default     = "1"
    }

    variable "cloud_run_max_instances" {
      description = "Maximum number of Cloud Run instances"
      type        = number
      default     = 2
    }
    ```

4. Add outputs for the Cloud Run service:

    ```hcl
    output "cloud_run_service_url" {
      description = "The public URL of the Cloud Run service"
      value       = google_cloud_run_service.mnist_api.status[0].url
    }

    output "cloud_run_service_name" {
      description = "The name of the Cloud Run service"
      value       = google_cloud_run_service.mnist_api.name
    }

    output "cloud_run_service_account_email" {
      description = "Service account email for Cloud Run"
      value       = google_service_account.cloud_run_sa.email
    }
    ```

5. Apply the configuration:

    ```bash
    tofu apply
    ```

6. After deployment, test your Cloud Run service:

    ```bash
    # Get the service URL
    SERVICE_URL=$(tofu output -raw cloud_run_service_url)

    # Test with a simple request (adjust based on your API)
    curl $SERVICE_URL/health

    # Test with data
    curl -X POST $SERVICE_URL/predict \
      -H "Content-Type: application/json" \
      -d '{"data": [...]}'
    ```

7. Monitor the Cloud Run service:

    ```bash
    # View recent logs
    gcloud run services describe mnist-inference-api --region=us-central1

    # View logs in real-time
    gcloud logging read "resource.type=cloud_run_revision" --limit=50
    ```

    ??? success "Solution"

        Your Cloud Run service is now deployed and accessible via a public URL. The service:
        - Automatically scales based on traffic
        - Has separate configurations for dev and prod environments
        - Pulls images from your Artifact Registry
        - Has access to Cloud Storage for data
        - Can be monitored through Cloud Logging

        The service URL can be integrated into your frontend or used as an API endpoint. Since it's deployed from
        Artifact Registry, you can update the image tag in your configuration to deploy new versions.

8. (Optional) Set up Cloud Run with secrets management:

    If your API needs secrets (like database credentials or API keys), you can inject them from Secret Manager:

    ```hcl
    # Grant Cloud Run service account access to secrets
    resource "google_secret_manager_secret_iam_member" "cloud_run_secret_access" {
      secret_id = "your-secret-name"
      role      = "roles/secretmanager.secretAccessor"
      member    = "serviceAccount:${google_service_account.cloud_run_sa.email}"
    }

    # Reference in Cloud Run:
    # env {
    #   name = "DATABASE_PASSWORD"
    #   value_from {
    #     secret_key_ref {
    #       name = "database-password"
    #       key  = "latest"
    #     }
    #   }
    # }
    ```

## Best Practices

When using Infrastructure as Code, follow these best practices:

1. **Version Control**: Always commit your `.tf` files to version control (git), but exclude `terraform.tfvars` and
    `*.tfstate` files.

2. **Code Organization**: Organize your code into logical files (`main.tf`, `variables.tf`, `outputs.tf`) and modules.
    As your infrastructure grows, create separate files for different services (e.g., `artifact_registry.tf`,
    `vertex_ai.tf`).

3. **State Management**: Use remote backends for team collaboration and enable state locking to prevent concurrent
    modifications. The state file contains sensitive information and should be stored securely.

4. **Variable Validation**: Use validation rules to ensure variables have appropriate values, especially for sensitive
    settings like environment and machine types.

5. **Documentation**: Add descriptions to all variables and outputs. This makes it easier for team members to understand
    what each resource does.

6. **Planning Before Applying**: Always run `tofu plan` before `tofu apply` to review changes. This is especially
    important when working with production environments.

7. **Environment Separation**: Use different directories, workspaces, or variable files for different environments
    (dev, staging, prod). This prevents accidental changes to production infrastructure.

8. **Module Reusability**: Design modules to be reusable across different projects and environments. Avoid hardcoding
    values in modules.

9. **Service Accounts**: Create specific service accounts for different purposes (Cloud Build, Vertex AI, etc.) with
    minimal required permissions (principle of least privilege).

10. **API Management**: Use OpenTofu to explicitly enable required APIs. This ensures all dependencies are tracked and
    can be reproduced in other projects or environments.

11. **Linking to M21 Concepts**: Ensure your IaC configuration mirrors what you learned in
    [M21 Using the Cloud](using_the_cloud.md). For example:
    - Cloud Storage bucket configuration aligns with the data storage exercises
    - Artifact Registry setup matches the container registry exercises
    - Vertex AI configuration supports the training exercises
