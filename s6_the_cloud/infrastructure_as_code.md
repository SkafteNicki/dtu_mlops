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

    !!! note "Looking Ahead"

        The Artifact Registry you just created will be used in Exercise 10 when we set up Vertex AI training
        infrastructure. Training jobs will pull Docker images from this registry, so it's important to keep it
        organized with cleanup policies!

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

10. We have now set up infrastructure for building and storing containers. The next critical piece is setting up
    infrastructure for **training** our ML models in the cloud. In [M21 Using the Cloud - Training section](using_the_cloud.md#training),
    you learned how to run custom training jobs on Vertex AI by manually executing `gcloud ai custom-jobs create`
    commands with config files specifying machine types and container images.

    Behind the scenes, those training jobs needed several permissions to work:
    - Access to pull Docker images from Artifact Registry
    - Access to read training data from Cloud Storage
    - Access to write model checkpoints and logs
    - Permission to create and run training jobs

    In this exercise, you'll automate the provisioning of all these permissions using Infrastructure as Code, creating
    a dedicated service account for Vertex AI training with precisely the access it needs.

    !!! info "What is Vertex AI?"

        [Vertex AI](https://cloud.google.com/vertex-ai) is Google Cloud's unified ML platform that handles the entire
        machine learning workflow. In this course, we focus specifically on **custom training jobs**, which allow you to:

        - Run your own Docker containers with custom training code
        - Automatically provision VMs with specified hardware (CPU/GPU)
        - Scale experiments horizontally (run many jobs in parallel)
        - Access data via mounted Cloud Storage filesystem
        - Automatically clean up resources when jobs complete

        This is more scalable than manually creating VMs (Compute Engine approach from M21) because Vertex AI handles
        the infrastructure lifecycle automatically.

    1. First, let's enable the Vertex AI API. Add the following to your `main.tf`:

        ```hcl
        resource "google_project_service" "vertex_ai_api" {
          service            = "aiplatform.googleapis.com"
          disable_on_destroy = false
        }
        ```

        This enables the AI Platform API, which is the backend service for Vertex AI custom training jobs.

    2. Create a dedicated service account for Vertex AI training. This service account will be used by all your
        training jobs to access GCP resources:

        ```hcl
        resource "google_service_account" "vertex_ai_sa" {
          account_id   = "vertex-ai-training-sa"
          display_name = "Service Account for Vertex AI Training"

          depends_on = [google_project_service.vertex_ai_api]
        }
        ```

        The full email will be `vertex-ai-training-sa@<your-project-id>.iam.gserviceaccount.com`. You'll reference
        this when submitting training jobs.

    3. Now we need to grant this service account the necessary permissions. Let's start with **Artifact Registry access**.
        In M21, your config files specified an `imageUri` pointing to a container in Artifact Registry. Vertex AI
        needs permission to pull that container:

        ```hcl
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
        ```

        Notice we use the **reader** role (not writer like Cloud Build). Vertex AI only needs to *pull* images during
        training, not push them. This follows the principle of least privilege.

    4. Next, grant access to **Cloud Storage** for training data and outputs. In M21, you learned about the mounted
        filesystem (`/gcs/<bucket-name>/`) that Vertex AI provides. This requires storage permissions:

        ```hcl
        resource "google_storage_bucket_iam_member" "vertex_ai_data_access" {
          bucket = google_storage_bucket.my_bucket.name
          role   = "roles/storage.objectAdmin"
          member = "serviceAccount:${google_service_account.vertex_ai_sa.email}"

          depends_on = [
            google_storage_bucket.my_bucket,
            google_service_account.vertex_ai_sa
          ]
        }
        ```

        The **objectAdmin** role allows both reading data and writing outputs (model checkpoints, logs, etc.).

        !!! tip "Storage Roles Explained"

            - `storage.objectViewer` - Read-only access to objects
            - `storage.objectAdmin` - Read/write access to objects (recommended for training)
            - `storage.admin` - Full bucket control including deletion (too permissive!)

            For training jobs, `objectAdmin` is the right balance: jobs can read data and write results, but can't
            delete the bucket itself or change bucket-level settings.

    5. Grant the **Vertex AI User** role, which allows the service account to actually run training jobs. This is a
        project-level permission:

        ```hcl
        resource "google_project_iam_member" "vertex_ai_user" {
          project = var.gcp_project_id
          role    = "roles/aiplatform.user"
          member  = "serviceAccount:${google_service_account.vertex_ai_sa.email}"

          depends_on = [google_service_account.vertex_ai_sa]
        }
        ```

        This is different from the previous IAM bindings because it's a **project-level** permission (using
        `google_project_iam_member`) rather than a resource-specific permission. The `aiplatform.user` role allows
        creating and managing custom training jobs.

        !!! info "Project-Level vs Resource-Level Permissions"

            **Resource-level permissions** (`google_artifact_registry_repository_iam_member`, `google_storage_bucket_iam_member`):
            - Apply to a specific resource (e.g., one bucket, one repository)
            - More granular and secure
            - Preferred when possible

            **Project-level permissions** (`google_project_iam_member`):
            - Apply to all resources in the project
            - Necessary for some roles like `aiplatform.user`
            - Use when resource-level isn't available

            Always prefer resource-level permissions when you have the choice!

    6. Finally, grant permission to write logs. Training jobs produce stdout/stderr output and metrics that get written
        to Cloud Logging:

        ```hcl
        resource "google_project_iam_member" "vertex_ai_logs" {
          project = var.gcp_project_id
          role    = "roles/logging.logWriter"
          member  = "serviceAccount:${google_service_account.vertex_ai_sa.email}"

          depends_on = [google_service_account.vertex_ai_sa]
        }
        ```

        Without this permission, you wouldn't be able to view logs from your training jobs in the GCP Console or via
        `gcloud logging read` commands.

    7. Create a dedicated bucket for training configurations and outputs. This is a best practice for organizing
        your ML experiments:

        ```hcl
        resource "google_storage_bucket" "training_configs" {
          name          = "${var.bucket_name}-training-configs"
          location      = "EU"
          force_destroy = true

          uniform_bucket_level_access = true

          versioning {
            enabled = true
          }
        }
        ```

        This bucket will store:
        - Training configuration files (config.yaml)
        - Model checkpoints during training
        - Final trained models
        - Training metrics and logs

    8. Grant the service account access to this new bucket:

        ```hcl
        resource "google_storage_bucket_iam_member" "vertex_ai_configs_access" {
          bucket = google_storage_bucket.training_configs.name
          role   = "roles/storage.objectAdmin"
          member = "serviceAccount:${google_service_account.vertex_ai_sa.email}"

          depends_on = [
            google_storage_bucket.training_configs,
            google_service_account.vertex_ai_sa
          ]
        }
        ```

    9. Add outputs to expose the service account information:

        ```hcl
        output "vertex_ai_service_account_email" {
          description = "Email of the Vertex AI training service account (use with --service-account flag)"
          value       = google_service_account.vertex_ai_sa.email
        }

        output "training_configs_bucket_name" {
          description = "Name of the bucket for storing training configurations and outputs"
          value       = google_storage_bucket.training_configs.name
        }

        output "training_configs_bucket_url" {
          description = "GCS URL of the training configs bucket"
          value       = "gs://${google_storage_bucket.training_configs.name}"
        }
        ```

    10. Run `tofu plan` to preview all the resources that will be created:

        ```bash
        tofu plan
        ```

        You should see approximately 7 new resources:
        - 1 API enablement (Vertex AI)
        - 1 service account
        - 4 IAM bindings (Artifact Registry, Storage, Vertex AI User, Logging)
        - 1 training configs bucket
        - 1 IAM binding for training configs bucket

    11. Apply the configuration:

        ```bash
        tofu apply
        ```

    12. Verify the service account was created:

        ```bash
        gcloud iam service-accounts list | grep vertex-ai
        ```

        You should see `vertex-ai-training-sa@<your-project-id>.iam.gserviceaccount.com`

    13. Verify the Artifact Registry permissions:

        ```bash
        gcloud artifacts repositories get-iam-policy \
            $(tofu output -raw artifact_registry_repository_id) \
            --location=$(tofu output -raw artifact_registry_location)
        ```

        You should see your Vertex AI service account listed with the `roles/artifactregistry.reader` role.

    14. Verify the Cloud Storage permissions:

        ```bash
        gsutil iam get gs://$(tofu output -raw bucket_name)
        ```

        You should see the Vertex AI service account with `roles/storage.objectAdmin`.

    15. Now comes the important test: **running an actual training job** using the infrastructure you just provisioned!
        If you have a `config.yaml` file from your M21 exercises, you can use it. Here's an example config:

        === "CPU"

            ```yaml
            # config_cpu.yaml
            workerPoolSpecs:
                machineSpec:
                    machineType: n1-highmem-2
                replicaCount: 1
                containerSpec:
                    imageUri: europe-west1-docker.pkg.dev/dtu-mlops-2026/dtu-mlops-2026-docker-repo/trainer:latest
            ```

        === "GPU"

            ```yaml
            # config_gpu.yaml
            workerPoolSpecs:
                machineSpec:
                    machineType: n1-standard-8
                    acceleratorType: NVIDIA_TESLA_T4
                    acceleratorCount: 1
                replicaCount: 1
                containerSpec:
                    imageUri: europe-west1-docker.pkg.dev/dtu-mlops-2026/dtu-mlops-2026-docker-repo/trainer:latest
            ```

        Update the `imageUri` to match your Artifact Registry repository (you can get this from `tofu output artifact_registry_repository_url`).

    16. Submit a training job using your new service account:

        ```bash
        gcloud ai custom-jobs create \
            --region=europe-west1 \
            --display-name=iac-test-job \
            --service-account=$(tofu output -raw vertex_ai_service_account_email) \
            --config=config_cpu.yaml
        ```

        The key addition here is `--service-account=$(tofu output -raw vertex_ai_service_account_email)`, which tells
        Vertex AI to use the service account you just created with OpenTofu.

    17. Monitor the job:

        ```bash
        gcloud ai custom-jobs list --region=europe-west1
        ```

        You can also view the job in the GCP Console under Vertex AI → Training → Custom Jobs.

    18. If the job runs successfully, congratulations! You've successfully:
        - Provisioned all necessary infrastructure for Vertex AI training using IaC
        - Created a properly-scoped service account with minimal required permissions
        - Tested the infrastructure with an actual training job
        - Automated what you previously did manually in M21

    ??? success "Solution"

        The complete solution files are available in the `exercise_files/terraform/` directory. 
        You can expand the sections below to see the full file contents and how they build upon previous exercises:

        ??? example "main_v7.tf - Complete infrastructure with Vertex AI"

            This file includes all resources from previous exercises plus the new Vertex AI infrastructure:

            ```hcl linenums="1" title="terraform/main_v7.tf"
            --8<-- "s6_the_cloud/exercise_files/terraform/main_v7.tf"
            ```

        ??? example "variables_v7.tf - Variables for all exercises"

            The variables file includes all the variables used across exercises 1-10:

            ```hcl linenums="1" title="terraform/variables_v7.tf"
            --8<-- "s6_the_cloud/exercise_files/terraform/variables_v7.tf"
            ```

        ??? example "outputs_v7.tf - All outputs including Vertex AI"

            This outputs file exposes all the important values from your infrastructure:

            ```hcl linenums="1" title="terraform/outputs_v7.tf"
            --8<-- "s6_the_cloud/exercise_files/terraform/outputs_v7.tf"
            ```

        Key concepts demonstrated:
        - **Multiple IAM binding types**: Resource-level and project-level permissions
        - **Principle of least privilege**: Reader for images, objectAdmin for data, specific training role
        - **Service account lifecycle**: Create SA, grant permissions, use in jobs
        - **Infrastructure testing**: Verify with actual training job
        - **Best practices**: Separate bucket for training artifacts, versioning enabled

    !!! tip "Summary of Permissions"

        Here's a table summarizing all the permissions granted to the Vertex AI service account:

        | Permission | Type | Role | Purpose |
        |------------|------|------|---------|
        | Artifact Registry | Resource | `artifactregistry.reader` | Pull training container images |
        | Cloud Storage (data) | Resource | `storage.objectAdmin` | Read training data, write outputs |
        | Cloud Storage (configs) | Resource | `storage.objectAdmin` | Read/write training configs |
        | Vertex AI | Project | `aiplatform.user` | Create and manage training jobs |
        | Cloud Logging | Project | `logging.logWriter` | Write training logs |

        Each permission serves a specific purpose and follows the principle of least privilege!

    !!! info "Connecting Back to M21"

        In [M21 Using the Cloud](using_the_cloud.md#training), you:
        - Manually enabled the Vertex AI API through the console
        - Used default service accounts or your own credentials
        - Ran `gcloud ai custom-jobs create` commands
        - Specified containers from Artifact Registry
        - Accessed data via `/gcs/` mounted filesystem

        All of those manual setup steps and implicit permissions are now:
        - ✅ Explicitly defined in code
        - ✅ Version controlled and reproducible
        - ✅ Properly scoped with minimal permissions
        - ✅ Easy to replicate across projects or teams

        This is the power of Infrastructure as Code for ML workflows!

11. (Optional Advanced) In the previous exercises, you've automated infrastructure for **training** ML models (Vertex AI), 
    **building** containers (Cloud Build), and **storing** images (Artifact Registry). The final piece of a complete MLOps 
    pipeline is **deployment**—making your trained models accessible to users via APIs. 
    
    In [M25 Cloud Deployment](../../s7_deployment/cloud_deployment.md#cloud-run), you'll learn how to manually deploy 
    applications to Cloud Run using `gcloud run deploy`. But just like with training and building, we can automate the 
    underlying infrastructure setup using Infrastructure as Code. In this optional exercise, you'll provision the service 
    accounts and permissions needed to support Cloud Run deployments.

    !!! info "What is Cloud Run?"

        [Cloud Run](https://cloud.google.com/run/docs) is Google Cloud's serverless container platform that lets you 
        deploy containerized applications without managing servers. Key features include:

        - **Serverless**: No infrastructure to manage—just deploy your container
        - **Auto-scaling**: Automatically scales from zero to handle traffic spikes
        - **Pay-per-use**: Only pay for actual usage (CPU, memory, requests)
        - **Perfect for ML APIs**: Ideal for inference endpoints that serve predictions

        Cloud Run automatically:
        - Creates and manages compute infrastructure
        - Scales containers based on incoming requests
        - Handles load balancing and traffic routing
        - Provides HTTPS endpoints
        - Cleans up idle containers to save costs

        This makes it perfect for deploying ML inference APIs that have variable traffic patterns!

    1. First, let's enable the Cloud Run API. Add the following to your `main.tf`:

        ```hcl
        resource "google_project_service" "cloud_run_api" {
          service            = "run.googleapis.com"
          disable_on_destroy = false
        }
        ```

        The Cloud Run API is required for deploying and managing serverless containers in Google Cloud.

    2. Create a dedicated service account for Cloud Run deployments. This service account will be used by your deployed 
        Cloud Run services to access other GCP resources:

        ```hcl
        resource "google_service_account" "cloud_run_sa" {
          account_id   = "cloud-run-sa"
          display_name = "Service Account for Cloud Run Deployments"

          depends_on = [google_project_service.cloud_run_api]
        }
        ```

        The full email will be `cloud-run-sa@<your-project-id>.iam.gserviceaccount.com`. You'll use this when deploying 
        services to Cloud Run in M25.

        !!! tip "Service Account Comparison"

            You've now created three different service accounts, each with a specific purpose:

            | Service Account | Used During | Purpose | Artifact Registry Permission |
            |----------------|-------------|---------|------------------------------|
            | `cloud-build-sa` | **Build time** | Push newly built Docker images to Artifact Registry | `artifactregistry.writer` (push) |
            | `vertex-ai-training-sa` | **Training time** | Pull containers and run ML training jobs | `artifactregistry.reader` (pull) |
            | `cloud-run-sa` | **Deployment/Runtime** | Pull containers and serve ML inference APIs | `artifactregistry.reader` (pull) |

            This separation follows the **principle of least privilege**: each service account has only the permissions 
            it needs for its specific task. Build processes need to push images, while training and serving only need 
            to pull them.

    3. Grant the Cloud Run service account permission to pull Docker images from Artifact Registry. When you deploy a 
        Cloud Run service, it needs to pull the container image from your registry:

        ```hcl
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
        ```

        Notice we use the **reader** role (not writer), just like with Vertex AI. Cloud Run only needs to pull images 
        during deployment, not push them. This is a key security principle—only grant write permissions where absolutely 
        necessary.

    4. (Optional) If your deployed Cloud Run services need to access Cloud Storage (e.g., to load model weights or write 
        prediction results), you can grant storage permissions. **Skip this step for now** unless you know your 
        deployment will need storage access:

        ```hcl
        # Uncomment if your Cloud Run services need to read/write from Cloud Storage
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
        ```

        If you need read-only access, use `storage.objectViewer`. For read-write access (e.g., writing prediction logs), 
        use `storage.objectAdmin`.

    5. Add outputs to expose the Cloud Run service account information. This will make it easy to reference when deploying 
        in M25:

        ```hcl
        output "cloud_run_service_account_email" {
          description = "Email of the Cloud Run service account (use with --service-account flag in gcloud run deploy)"
          value       = google_service_account.cloud_run_sa.email
        }

        output "cloud_run_deployment_command_example" {
          description = "Example command for deploying to Cloud Run using this infrastructure (for M25)"
          value       = "gcloud run deploy <service-name> --image=<image-url> --service-account=${google_service_account.cloud_run_sa.email} --region=${var.region} --allow-unauthenticated"
        }
        ```

        The example output provides a template command you can use in M25, with the service account already filled in.

    6. Run `tofu plan` to preview the changes:

        ```bash
        tofu plan
        ```

        You should see approximately 3 new resources:
        - 1 API enablement (Cloud Run)
        - 1 service account (cloud-run-sa)
        - 1 IAM binding (Artifact Registry reader)

    7. Apply the configuration:

        ```bash
        tofu apply
        ```

    8. Verify the service account was created:

        ```bash
        gcloud iam service-accounts list | grep cloud-run-sa
        ```

        You should see `cloud-run-sa@<your-project-id>.iam.gserviceaccount.com`

    9. Verify the Artifact Registry permissions:

        ```bash
        gcloud artifacts repositories get-iam-policy \
            $(tofu output -raw artifact_registry_repository_id) \
            --location=$(tofu output -raw artifact_registry_location)
        ```

        You should now see **three** service accounts with permissions:
        - `cloud-build-sa` with `roles/artifactregistry.writer` (from Exercise 9)
        - `vertex-ai-training-sa` with `roles/artifactregistry.reader` (from Exercise 10)
        - `cloud-run-sa` with `roles/artifactregistry.reader` (new!)

    10. Get the deployment command example for M25:

        ```bash
        tofu output cloud_run_deployment_command_example
        ```

        This will show you the exact command format you'll use in M25, with your service account and region already filled 
        in. Save this for later!

    !!! success "Infrastructure Complete!"

        Congratulations! You've now provisioned the complete infrastructure needed for Cloud Run deployments. The 
        service account you created has the minimal permissions required to:
        
        - ✅ Pull container images from Artifact Registry during deployment
        - ✅ Run as the identity of deployed Cloud Run services
        - ✅ (Optional) Access Cloud Storage if you uncommented that section

    !!! info "Looking Ahead to M25"

        In [M25 Cloud Deployment](../../s7_deployment/cloud_deployment.md#cloud-run), you'll use this infrastructure to:

        - **Deploy ML inference APIs** as serverless containers
        - **Automatically scale** based on incoming prediction requests
        - **Pay only for actual usage** when serving predictions
        - **Get HTTPS endpoints** automatically for your APIs
        - **Monitor deployments** with Cloud Logging

        Example deployment command you'll run in M25:

        ```bash
        gcloud run deploy mnist-inference-api \
            --source . \
            --region europe-west1 \
            --allow-unauthenticated \
            --service-account=$(tofu output -raw cloud_run_service_account_email)
        ```

        The `--service-account` flag uses the infrastructure you just created!

    ??? success "Solution"

        The complete solution files are available in the `exercise_files/terraform/` directory.
        You can expand the sections below to see the full file contents:

        ??? example "main_v8.tf - Complete infrastructure with Cloud Run"

            This file includes all resources from exercises 1-10 plus the new Cloud Run infrastructure:

            ```hcl linenums="1" title="terraform/main_v8.tf"
            --8<-- "s6_the_cloud/exercise_files/terraform/main_v8.tf"
            ```

        ??? example "variables_v8.tf - Variables configuration"

            The variables file (same as v7, no new variables needed for Cloud Run):

            ```hcl linenums="1" title="terraform/variables_v8.tf"
            --8<-- "s6_the_cloud/exercise_files/terraform/variables_v8.tf"
            ```

        ??? example "outputs_v8.tf - All outputs including Cloud Run"

            This outputs file includes all previous outputs plus Cloud Run service account information:

            ```hcl linenums="1" title="terraform/outputs_v8.tf"
            --8<-- "s6_the_cloud/exercise_files/terraform/outputs_v8.tf"
            ```

        Key concepts demonstrated:
        - **Minimal infrastructure approach**: Only service account + permissions, no actual Cloud Run service deployment
        - **Separation of concerns**: Infrastructure (IaC in M22) vs. Deployment (runtime in M25)
        - **Consistent patterns**: Same approach as Vertex AI (enable API, create SA, grant permissions)
        - **Principle of least privilege**: Reader access to Artifact Registry (not writer)
        - **Service account segregation**: Separate SAs for build, training, and deployment

    !!! tip "Summary of Service Accounts"

        You've now created a complete set of service accounts for your MLOps pipeline:

        | Service Account | Purpose | When Used | Key Permissions |
        |----------------|---------|-----------|-----------------|
        | `cloud-build-sa` | Build Docker images | CI/CD pipeline (M21) | `artifactregistry.writer` |
        | `vertex-ai-training-sa` | Run ML training jobs | Training (M21, M22) | `artifactregistry.reader`, `storage.objectAdmin`, `aiplatform.user` |
        | `cloud-run-sa` | Serve ML inference APIs | Deployment (M25) | `artifactregistry.reader` |

        This separation ensures that a compromised deployment can't push malicious images, and a compromised build 
        process can't access production training data!

## 🧠 Knowledge check

1. OpenTofu operates on two core principles: idempotency and declarative configuration. 
   Explain what idempotency means in the context of Infrastructure as Code, and why it's 
   important for managing cloud resources.

    ??? success "Solution"

        **Idempotency** means that applying the same configuration multiple times will always 
        result in the same infrastructure state. For example, if you define a compute instance 
        in your configuration and apply it, running `tofu apply` again will not create a duplicate 
        instance but will ensure that the existing instance matches the defined configuration.

        This is important because:
        - It prevents accidental creation of duplicate resources
        - It makes infrastructure changes predictable and safe
        - It allows you to re-apply configurations without worrying about side effects
        - It's essential for automation and CI/CD pipelines

2. After running `tofu apply`, OpenTofu creates a state file. What information does this 
   file contain, and why is it important to keep it secure? What command would you use 
   to list all resources currently tracked in the state?

    ??? success "Solution"

        The **state file** (`terraform.tfstate`) contains:
        - A mapping of resource names to their actual cloud resource IDs
        - Current configuration values for all managed resources
        - Metadata about dependencies between resources
        - Sensitive information like database passwords or API keys (if stored in outputs)

        It's important to keep it secure because:
        - It may contain sensitive credentials
        - It's the source of truth for what OpenTofu manages
        - Loss or corruption can lead to orphaned resources or duplicate creations

        To list all resources tracked in the state:
        ```bash
        tofu state list
        ```

3. In this module, you created three different service accounts: one for Cloud Build, 
   one for Vertex AI training, and one for Cloud Run. Why is it better to have separate 
   service accounts rather than using a single service account for all three purposes?

    ??? success "Solution"

        Using separate service accounts is better for several reasons:

        - **Principle of Least Privilege**: Each service account only has the permissions 
          it needs for its specific task. Cloud Build needs to push images (writer), while 
          Vertex AI and Cloud Run only need to pull images (reader).

        - **Security Isolation**: If one service is compromised, the damage is limited. 
          For example, if Cloud Run is compromised, the attacker can't push malicious images 
          because the Cloud Run SA doesn't have write access to Artifact Registry.

        - **Auditability**: It's easier to track which service performed which action when 
          they have separate identities.

        - **Compliance**: Many security standards require separation of duties between 
          build, training, and deployment processes.

4. When granting IAM permissions, you used both `google_project_iam_member` (for Vertex AI User role) 
   and `google_storage_bucket_iam_member` (for storage access). What's the difference between 
   these two approaches, and when should you prefer one over the other?

    ??? success "Solution"

        **Resource-level permissions** (`google_storage_bucket_iam_member`, `google_artifact_registry_repository_iam_member`):
        - Apply to a specific resource (e.g., one bucket, one repository)
        - More granular and secure
        - Preferred when possible
        - Example: Granting access to only the training data bucket, not all buckets

        **Project-level permissions** (`google_project_iam_member`):
        - Apply to all resources of a certain type in the project
        - Less granular but necessary for some roles
        - Use when resource-level isn't available
        - Example: The `aiplatform.user` role must be granted at project level

        **Best practice**: Always prefer resource-level permissions when available! Only use 
        project-level permissions when the role doesn't support resource-level binding.

5. In your Terraform configuration, you used the `depends_on` argument in several places, 
   such as when creating the Artifact Registry repository after enabling the API. What would 
   happen if you removed these `depends_on` declarations and tried to apply the configuration?

    ??? success "Solution"

        Without `depends_on`, OpenTofu would try to create resources in parallel without 
        respecting the dependency order. This could lead to:

        - **API not enabled errors**: If the Artifact Registry repository was created before 
          the API was enabled, the API call would fail with an error like "API not enabled".

        - **Race conditions**: Resources might be created in the wrong order, causing failures.

        - **Inconsistent state**: Some resources might be created while others fail, leaving 
          the infrastructure in a partially working state.

        The `depends_on` argument ensures that:
        - The Cloud Build API is enabled before creating the Cloud Build service account
        - The Artifact Registry API is enabled before creating the repository
        - Service accounts are created before IAM permissions are granted to them

        This explicit ordering prevents these race conditions and ensures reliable deployments.

## Further Exploration

Congratulations! You've now automated the creation of a complete MLOps infrastructure stack using OpenTofu:

- ✅ **Storage buckets** for data and training artifacts
- ✅ **Compute instances** for development and experimentation
- ✅ **Artifact Registry** for storing Docker images
- ✅ **Cloud Build** service accounts for CI/CD
- ✅ **Vertex AI** infrastructure for scalable training
- ✅ **Cloud Run** service accounts for serverless deployment (if you completed Exercise 11)

You've learned how to translate manual cloud operations from M21 into reproducible, version-controlled Infrastructure
as Code. This is the foundation of production-ready MLOps!

If you want to explore more advanced topics, consider these optional exercises:

- **Remote State Backend**: Configure OpenTofu to store state files in Google Cloud Storage for team collaboration
    and state locking. This is essential for working in teams where multiple people manage the same infrastructure.
    [Learn more](https://opentofu.org/docs/language/settings/backends/gcs/)

- **Modular Terraform Organization**: As your infrastructure grows, organize it into reusable modules instead of
    keeping everything in one file. This makes it easier to share infrastructure patterns across projects.
    [Learn more](https://opentofu.org/docs/language/modules/)

- **Environment Separation**: Create separate configurations for dev, staging, and production environments using
    workspaces or separate directories. This prevents accidental changes to production infrastructure.
    [Learn more](https://opentofu.org/docs/cli/workspaces/)

