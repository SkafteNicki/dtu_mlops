![Logo](../figures/icons/kubernetes.png){ align=right width="130"}

!!! danger
    Module is still under development

# Kubernetes

Kubernetes, also known as K8s, is an open-source platform designed to automate the deployment, scaling, and operation of
application containers across clusters of hosts. It provides the framework to run distributed systems resiliently,
handling scaling and failover for your applications, providing deployment patterns, and more.

## What is Kubernetes?

### Brief History
Kubernetes was originally developed by Google and is now maintained by the Cloud Native Computing Foundation.

### Core Functions
Kubernetes makes it easier to deploy and manage containerized applications at scale.

### Key Concepts

- **Pods**
- **Nodes**
- **Clusters**
- ...

---

## Kubernetes Architecture

Kubernetes follows a client-server architecture. At a high level, it consists of a Control Plane (master) and Nodes
(workers).

<figure markdown>
![Kubernetes Architecture](../figures/components_of_kubernetes.png){ width="800" }
<figcaption>
Overview of Kubernetes Architecture.
Image Credit: <a href="https://kubernetes.io/docs/concepts/overview/components/">Kubernetes Official Documentation</a>
</figcaption>
</figure>

### Control Plane Components

- **API Server**: The frontend for Kubernetes.
- **etcd**: Consistent and highly-available key value store.
- ...

### Node Components

- **Kubelet**: An agent that runs on each node.
- **Container Runtime**: The software responsible for running containers.
- ...

---

## Minikube: Local Kubernetes Environment

Minikube is a tool that allows you to run Kubernetes locally. It runs a single-node Kubernetes cluster inside a VM on
your laptop for users looking to try out Kubernetes or develop with it day-to-day.

### Installing Minikube

1. **System Requirements**: Ensure your system meets the minimum requirements.
2. **Download and Install**: Visit [Minikube's official installation guide](https://minikube.sigs.k8s.io/docs/start/).
3. **Start Minikube**: Run `minikube start`.

### ❔ Exercises

1. Install Minikube following the steps above.
2. Validate the installation by typing `minikube` in a terminal.
3. Ensure that [kubectl](https://kubernetes.io/docs/reference/kubectl/kubectl/), the command-line tool for Kubernetes,
is correctly installed by typing `kubectl` in a terminal.

---

## Yatai: Model Serving Platform for Kubernetes

[Yatai](https://github.com/bentoml/Yatai) is a model serving platform, making it easier to deploy machine learning
models in Kubernetes environments.

### What is Yatai?

Yatai simplifies the deployment, management, and scaling of machine learning models in Kubernetes.

### Getting Started with Yatai

1. **Installation**: Steps to install Yatai in your Kubernetes cluster.
2. **Basic Usage**: How to deploy your first model using Yatai.

---

## Additional Resources

- [Official Kubernetes Documentation](https://kubernetes.io/docs/)
- [Interactive Tutorials](https://kubernetes.io/docs/tutorials/)
- [Community Forums](https://discuss.kubernetes.io/)
- ...
