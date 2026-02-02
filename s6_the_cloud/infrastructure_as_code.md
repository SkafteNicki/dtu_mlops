![Logo](../figures/icons/opentofu.png){ align=right width="130"}

# Infrastructure as Code (IaC)

---

If you have worked through the previous two module, [M20 on cloud setup](cloud_setup.md) and
[M21 on using cloud services](using_the_cloud.md), you should now have a basic understanding of how to use different
cloud services to support your machine learning pipeline. You will most likely have noticed that setting up and managing
cloud resources manually either through the GCP web interface or CLI can both be time-consuming and error-prone. This is
where Infrastructure as Code (IaC) comes into play. You can see it as the cloud version of what we did in module
[M6 on coding structure](../s2_organisation_and_version_control/code_structure.md) for organizing and managing your
codebase. In that module, we learned how to use `cookiecutter` to create a project from a template, which provided a
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
terraform  {
  ...
}

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

In the beginning of the file, you define the `terraform` block which contains general settings for OpenTofu/Terraform.
It often includes which plugins (providers) you want to use and their versions. After that, you define one or more
`provider` blocks where you define which cloud provider you want to use (yes, you can define multiple providers).
Then you define different resources that you want to create. Each resource has a type (e.g. `google_compute_instance`)
and a name (e.g. `my_instance`). Inside the resource block, you define different parameters that describe how you want
the resource to be configured. You can see this file as a way to structure all the `gcloud` commands you would have to
run manually to create the same resource. Finally, you can also define data sources which are read-only references to
existing resources.

After writing your configuration files, you can use the OpenTofu CLI to apply the configuration and create the resources

```bash
tofu init  # first time setup
tofu plan  # see what changes will be made
tofu apply  # apply the configuration
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

2. Then lets add a `main.tf` file in the root of your repository with the following content:

    ```hcl
    terraform {
      required_providers {
        google = {
          source  = "hashicorp/google"
          version = "~> 7.16.0"  # code
        }
      }
      required_version = ">= 1.5.0"
    }

    provider "google" {
      project = "dtu-mlops-2026"
      region  = "europe-west1"
    }
    ```

    and then initialize the OpenTofu configuration by running

    ```bash
    tofu init
    ```

    you should see something like this:

    ```bash
    ❯ tofu init

    Initializing the backend...

    Initializing provider plugins...
    - Finding hashicorp/google versions matching "~> 7.16.0"...
    - Installing hashicorp/google v7.16.0...
    - Installed hashicorp/google v7.16.0 (signed, key ID 0C0AF313E5FD9F80)
    ...
    ```

    !!! note "Add .terraform to .gitignore"

        Running `tofu init` creates a `.terraform` directory that contains provider plugins and modules. This directory
        can be large and should not be committed to version control. Add `.terraform/` to your `.gitignore` file. The
        `.terraform.lock.hcl` file, however, **should** be committed as it locks provider versions for reproducibility.

        Make sure your `.gitignore` includes:

        ```
        .terraform/
        *.tfstate
        *.tfstate.backup
        terraform.tfvars
        ```

    look at the code in the `main.tf` file, can you find the relevant information/documentation on the internet to
    understand what it does? Additionally, running `tofu init` has created a file in your folder, what is the purpose of
    this file?

    ??? success "Solution"

        Even though we are using OpenTofu, most information we need is still found in the Terraform documentation. For
        example to read about the requirements block, you can go to this link:

        <https://developer.hashicorp.com/terraform/language/providers/requirements>

        The relevant documentation for google cloud provider can be found here:

        <https://registry.terraform.io/providers/hashicorp/google/latest>

        And the specific resources that is available for Google Cloud can be found here:

        <https://registry.terraform.io/providers/hashicorp/google/latest/docs>

        Finally, the `tofu init` command has created a `.terraform.lock.hcl` file that locks the *exact* version of the
        provider plugins being used. This ensures that the same versions are used across different machines and this
        file should therefore be committed to version control.

3. Let's begin the process of creating resources. To begin with lets create a simple GCP bucket.
    Add the following code to your `main.tf` file:

    ```hcl
    resource "google_storage_bucket" "my_bucket" {
      name          = "dtu-mlops-2026-infra-as-code-bucket-<random-numbers>"
      location      = "EU"
      force_destroy = true

      uniform_bucket_level_access = true

      versioning {
        enabled = true
      }
    }
    ```

    and then run `tofu plan` to see what changes will be made. If everything looks good e.g. your plan should return
    `Plan: 1 to add, 0 to change, 0 to destroy.` then run `tofu apply` to create the bucket. Run `gsutil ls` to verify
    that the bucket was created. Also checkout the `*.tfstate` file that was created, what information does it contain?

    ??? success "Solution"

        The generated state file will look something like this:

        ```json
        {
          "terraform_version": "1.11.4",
          "serial": 2,

          "resources": [
            {
              "type": "google_storage_bucket",
              "name": "my_bucket",

              "instances": [
                {
                  "id": "dtu-mlops-2026-infra-as-code-bucket-123940141",

                  "attributes": {
                    "name": "dtu-mlops-2026-infra-as-code-bucket-123940141",
                    "project": "dtu-mlops-2026",
                    "location": "EU",
                    "storage_class": "STANDARD",
                    "versioning": {
                      "enabled": true
                    },
                    "force_destroy": true,
                    "self_link": "https://www.googleapis.com/storage/v1/b/..."
                  }
                }
              ]
            }
          ]
        }
        ```

        in very simple terms it answers the question: "what did I create, where is it, and what does it look like right
        now?". Importantly, you will see that when you run `tofu apply` again, a `terraform.tfstate.backup` file will be
        created as a backup of the previous state before any changes are applied.

4. Next, let's make our file a little configurable. Instead of hardcoding everything in the `main.tf` file, we can use
    variables to make it more flexible. Create a new file called `variables.tf` in the root of the repository and add
    the following content:

    ```hcl
    variable "gcp_project_id" {
      description = "The GCP project ID"
      type        = string
      default     = "dtu-mlops-2026"
    }

    variable "region" {
      description = "The GCP region for resources"
      type        = string
      default     = "europe-west1"
    }

    variable "bucket_name" {
      description = "The name of the storage bucket"
      type        = string
      default     = "dtu-mlops-2026-infra-as-code-bucket-<random-numbers>"
    }
    ```

    Then update your `main.tf` file to use these variables:

    ```hcl
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
    ```

    Assuming you in `variables.tf` have set the default values to the same as before, you should see that no changes
    are needed when running `tofu plan`. Let's try to change the bucket name to something else. This can either be done
    using the command line:

    ```bash
    tofu plan -var="bucket_name=dtu-mlops-2026-infra-as-code-bucket-987654321"
    ```

    or by creating a `terraform.tfvars` file with the following content:

    ```hcl
    bucket_name = "dtu-mlops-2026-infra-as-code-bucket-987654321"
    ```

    !!! note "Do not commit terraform.tfvars"

        The `terraform.tfvars` file often contains sensitive information. Add it to your `.gitignore` file to prevent
        accidentally committing it to version control. Instead, you can use `terraform.tfvars.example` as a template
        for team members.

    Try it out and run `tofu apply` to create the new bucket. Try changing the other variables as well. Can you explain
    why only changing the bucket name results in a change when running `tofu plan`?

    ??? success "Solution"

        This part

        ```hcl
        provider "google" {
          project = var.gcp_project_id
          region  = var.region
        }
        ```

        only affects new resources created by the provider, not the provider itself. Changing the provider configuration
        does not automatically change existing resources.

5. Now that we have created a bucket, let's learn how OpenTofu can output information from our infrastructure.
    Outputs are useful for extracting values from your created resources, such as URLs, IPs, or resource names.
    Create a new file called `outputs.tf` in the root of your repository and add the following content:

    ```hcl
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
    ```

    Run `tofu apply` again to see the outputs. Even though no infrastructure changes are needed, OpenTofu will display
    the output values. You can also retrieve outputs later without applying changes by running:

    ```bash
    tofu output
    ```

    or to get a specific output value:

    ```bash
    tofu output bucket_url
    ```

    Try accessing a specific output value and verify that it matches the bucket you created.

    ??? success "Solution"

        After running `tofu apply`, you should see output similar to:

        ```
        Apply complete! Resources: 0 added, 0 changed, 0 destroyed.

        Outputs:

        bucket_location = "EU"
        bucket_name = "dtu-mlops-2026-infra-as-code-bucket-123940141"
        bucket_url = "gs://dtu-mlops-2026-infra-as-code-bucket-123940141"
        ```

        Running `tofu output bucket_url` will return just the URL value, which is useful for scripting and automation.
        You can use these outputs in other Terraform configurations, scripts, or CI/CD pipelines to reference the
        created resources without hardcoding values.

6. Next, try to figure out how to provision a virtual machine. The
    [precise configuration](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/compute_instance)
    you can determine but you need to add it to your `main.tf` file, use variables where appropriate and create outputs to extract important information about the created instance (e.g. instance name, internal and external IP address).

    ??? success "Solution"

        Here is an example of how you can create a GCP Compute Instance:

        In `variables.tf`:

        ```hcl
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

        In `main.tf`:

        ```hcl
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
        ```

        In `outputs.tf`:

        ```hcl
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
        ```

        Run `tofu plan` to see the changes and then `tofu apply` to create the instance. Note that we are using the
        ML-optimized image ``projects/ml-images/global/images/common-cu128-ubuntu-2404-nvidia-570-v20260129` which comes
        pre-installed with NVIDIA drivers and CUDA 12.8.

    1. After creating the instance, verify that it was created correctly by SSH-ing into it. Use the SSH command from
        the outputs to connect to the instance:

        ```bash
        # Get the SSH command from outputs
        tofu output ssh_command

        # Or directly SSH using the command
        $(tofu output -raw ssh_command)
        ```

        Try to connect to the instance to verify that it is running and accessible.

7. At this point it is probably a good idea to get to know also how you can manage and destroy resources using OpenTofu.
    Start by examining your state file by running:

    ```bash
    tofu state list
    ```

    This shows all the resources currently managed by OpenTofu.
    To see details of a specific resource:

    ```bash
    tofu state show google_compute_instance.training_instance
    ```

    What command would you use to destroy all resources created by your configuration?

    ??? success "Solution"

        To destroy all resources managed by your configuration, you can run:

        ```bash
        tofu destroy
        ```

        OpenTofu will ask for confirmation before destroying resources. It goes without saying that you should be very
        careful with this command, especially in production environments. After running `tofu destroy`, all resources
        should be removed. You can verify this by checking the GCP Console or running:

        ```bash
        gcloud compute instances list
        gsutil ls
        ```

8. In [M21 Using the Cloud - Artifact Registry section](using_the_cloud.md#artifact-registry), you manually created
    an Artifact Registry repository through the UI and gcloud commands for storing Docker container images. Now you'll
    automate this entire process using OpenTofu, making it reproducible and version-controlled.

    1. First, you need to enable the Artifact Registry API. Just like with compute instances and storage buckets, GCP
        requires APIs to be explicitly enabled before you can create resources. Add the following to your `main.tf`:

        ```hcl
        resource "google_project_service" "artifact_registry_api" {
          service            = "artifactregistry.googleapis.com"
          disable_on_destroy = false
        }
        ```

        The `disable_on_destroy = false` setting means that even if you destroy this resource with `tofu destroy`,
        the API will remain enabled in your project. This prevents accidentally breaking other services that might
        depend on it.

    2. Next, add the Artifact Registry repository resource. This will create a Docker repository in your specified
        region:

        ```hcl
        resource "google_artifact_registry_repository" "docker_repo" {
          depends_on = [google_project_service.artifact_registry_api]

          location      = var.region
          repository_id = "${var.gcp_project_id}-docker-repo"
          description   = "Docker repository for ML training images"
          format        = "DOCKER"
        }
        ```

        Notice the `depends_on` argument. This explicitly tells OpenTofu that the API must be enabled before creating
        the repository.

    3. Add a variable for the artifact registry ID to your `variables.tf`:

        ```hcl
        variable "artifact_registry_id" {
          description = "The ID of the artifact registry repository (will be prefixed with project ID)"
          type        = string
          default     = "docker-repo"
        }
        ```

        Then update the repository resource to use this variable:

        ```hcl
        resource "google_artifact_registry_repository" "docker_repo" {
          depends_on = [google_project_service.artifact_registry_api]

          location      = var.region
          repository_id = "${var.gcp_project_id}-${var.artifact_registry_id}"
          description   = "Docker repository for ML training images"
          format        = "DOCKER"
        }
        ```

    4. Add outputs to your `outputs.tf` to easily reference the repository:

        ```hcl
        output "artifact_registry_repository_url" {
          description = "The URL of the Artifact Registry repository for pushing/pulling images"
          value       = "${var.region}-docker.pkg.dev/${var.gcp_project_id}/${google_artifact_registry_repository.docker_repo.repository_id}"
        }

        output "artifact_registry_repository_id" {
          description = "The ID of the Artifact Registry repository"
          value       = google_artifact_registry_repository.docker_repo.repository_id
        }
        ```

        This output will give you the full URL you need for pushing and pulling Docker images, such as:
        `europe-west1-docker.pkg.dev/dtu-mlops-2026/dtu-mlops-2026-docker-repo`

    5. Run `tofu plan` to preview the changes. You should see that OpenTofu plans to create 2 new resources:

        ```bash
        Plan: 2 to add, 0 to change, 0 to destroy.
        ```

        then apply the changes with `tofu apply` and verify that the repository was created successfully using `gcloud`:

        ```bash
        gcloud artifacts repositories list --location=europe-west1
        ```

        Confirm also that the repository URL is the same as the one generated by OpenTofu outputs.

        ```bash
        tofu output artifact_registry_repository_url
        ```

    6. (Optional) Test the registry by pushing an image. First, configure Docker to authenticate with GCP:

        ```bash
        gcloud auth configure-docker europe-west1-docker.pkg.dev
        ```

        Then tag and push an image (using the busybox image from earlier exercises):

        ```bash
        # Get the repository URL
        REPO_URL=$(tofu output -raw artifact_registry_repository_url)

        # Tag the image
        docker tag busybox ${REPO_URL}/busybox:latest

        # Push the image
        docker push ${REPO_URL}/busybox:latest
        ```

        You can verify the image was pushed by checking the Artifact Registry in the GCP console or running:

        ```bash
        gcloud artifacts docker images list ${REPO_URL}
        ```

    7. (Optional) Can you figure out how to configure your OpenTofu setup to automatically clean up old images in the
        Artifact Registry e.g. setting a cleanup policy?

        ??? success "Solution"

            You can add a cleanup policy resource to your `main.tf`:

            ```hcl
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

            This configuration will keep only the 5 most recent versions of each image in the repository, automatically
            cleaning up older images. A good number of examples can be found
            [in the documentation](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/artifact_registry_repository)

9. (Optional) One thing you may have probably encountered when working in GCP is service accounts and IAM permissions.
    Let's see how we can setup a service account for Cloud Build using OpenTofu.

    1. Since we have gone through this a couple of times now, can you figure out how to enable the Cloud Build API using
        OpenTofu and then create a service account for Cloud Build?

        ??? success "Solution"

            First, enable the Cloud Build API by adding to your `main.tf`:

            ```hcl
            resource "google_project_service" "cloud_build_api" {
              service            = "cloudbuild.googleapis.com"
              disable_on_destroy = false
            }
            ```

            Create a service account for Cloud Build:

            ```hcl
            resource "google_service_account" "cloud_build_sa" {
              account_id   = "cloud-build-sa"
              display_name = "Service Account for Cloud Build"

              depends_on = [google_project_service.cloud_build_api]
            }
            ```

            The `account_id` must be unique within your project and between 6-30 characters. The full email of this
            service account will be `cloud-build-sa@<your-project-id>.iam.gserviceaccount.com`.

    2. Grant the service account permission to push images to the Artifact Registry. This is done using an **IAM
        (Identity and Access Management) binding**:

        ```hcl
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
        ```

        This resource grants the `roles/artifactregistry.writer` role to our Cloud Build service account on the
        Artifact Registry repository. The writer role allows the service account to push (write) images to the
        repository but not delete the repository itself.

    3. Add an output to expose the service account email:

        ```hcl
        output "cloud_build_service_account_email" {
          description = "Email of the Cloud Build service account (use this in Cloud Build triggers)"
          value       = google_service_account.cloud_build_sa.email
        }
        ```

    4. Run `tofu plan` to see what will be created:

        ```bash
        tofu plan
        ```

        You should see 3 new resources: the API enablement, the service account, and the IAM binding. Afterwareds, run
        `tofu apply` to create the resources and then verify everything was created correctly

        ```bash
        gcloud iam service-accounts list
        ```

        and that IAM permissions were set up correctly

        ```bash
        gcloud artifacts repositories get-iam-policy \
            $(tofu output -raw artifact_registry_repository_id) \
            --location=$(tofu output -raw artifact_registry_location)
        ```

        You should see your service account listed with the `roles/artifactregistry.writer` role.

    9. (Optional) If you already have Cloud Build triggers set up from M21, you can update them to use this service
        account. In your Cloud Build trigger configuration or `cloudbuild.yaml`, you can reference the service account:

        ```yaml
        # In cloudbuild.yaml (optional configuration)
        serviceAccount: 'projects/<project-id>/serviceAccounts/cloud-build-sa@<project-id>.iam.gserviceaccount.com'
        ```

        Or when creating a trigger via `gcloud`:

        ```bash
        gcloud builds triggers create github \
            --name="my-trigger" \
            --repo-name="my-repo" \
            --repo-owner="my-username" \
            --branch-pattern="^main$" \
            --build-config="cloudbuild.yaml" \
            --service-account="projects/<project-id>/serviceAccounts/cloud-build-sa@<project-id>.iam.gserviceaccount.com"
        ```

10.

## Further Exploration

Congratulations! You've now automated the creation of storage buckets, compute instances, artifact registries, and
Cloud Build infrastructure using OpenTofu. You've learned how to translate manual cloud operations into reproducible,
version-controlled Infrastructure as Code.

If you want to explore more advanced topics, the exercises below (currently commented out in the documentation)
provide additional learning opportunities:

- **Remote State Backend**: Configure OpenTofu to store state files in Google Cloud Storage for team collaboration
    and state locking (see commented Exercise 13 in the source)
- **Vertex AI Training Infrastructure**: Set up service accounts and permissions for running ML training jobs on
    Vertex AI (see commented Exercise 12 in the source)
- **Cloud Run Deployment**: Provision serverless container deployment infrastructure (see commented Exercise 14 in
    the source)

You can find the code for these advanced exercises in the commented sections of this file (lines 783-1026).

<!---
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
    minimal required permissions (principle of least privilege). See Exercise 9 for an example of creating a dedicated
    service account with specific IAM permissions.

10. **API Management**: Use OpenTofu to explicitly enable required APIs. This ensures all dependencies are tracked and
    can be reproduced in other projects or environments. See Exercises 8 and 9 for examples of enabling APIs before
    creating dependent resources.

11. **Linking to M21 Concepts**: Ensure your IaC configuration mirrors what you learned in
    [M21 Using the Cloud](using_the_cloud.md). For example:
    - Cloud Storage bucket configuration aligns with the data storage exercises
    - Artifact Registry setup matches the container registry exercises
    - Vertex AI configuration supports the training exercises -->
