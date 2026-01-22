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
    Terraform and created OpenTofu as a fully open-source alternative. For this reason the two tools are very similar
    in terms of syntax and usage, and the core concepts remain the same.

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
