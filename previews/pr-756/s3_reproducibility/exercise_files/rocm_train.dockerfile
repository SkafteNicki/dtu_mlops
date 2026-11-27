# Base image with ROCm and a ROCm-enabled build of PyTorch.
# Replace the tag with one that matches your GPU, see https://hub.docker.com/r/rocm/pytorch/tags
FROM rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.9.1

# Install uv by copying it from the official uv image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Use the Python environment that is already present in the ROCm image.
# You can check the location with `which python` inside the base image.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_LINK_MODE=copy

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src/ src/
COPY data/ data/

# --inexact keeps packages that are not part of the project, such as other libraries from the ROCm image.
# --no-install-package prevents uv from replacing torch and torchvision with the PyPI wheels.
RUN uv sync --locked --no-cache --inexact --no-install-project \
    --no-install-package torch --no-install-package torchvision

# --no-sync prevents uv from synchronizing the environment again when the container starts.
# Replace <project-name> with the name of your package.
ENTRYPOINT ["uv", "run", "--no-sync", "src/<project-name>/train.py"]
