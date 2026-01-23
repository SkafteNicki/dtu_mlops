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

2. For the rest of the exercises, we assume that you are just going to be using the running example on the corrupt
    mnist dataset. In the root of the repository, create a new file called `main.tf`. This file will contain the
    OpenTofu configuration for provisioning the necessary cloud resources.

    1. Add the following code to the `main.tf` file in the root of your repository to configure the Google Cloud
      provider:

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
          project = "<YOUR_GCP_PROJECT_ID>"
          region  = "us-central1"
        }
        ```

        Look through the [opentofu documentation](https://opentofu.org/docs/providers/google/) to understand what this code does.
