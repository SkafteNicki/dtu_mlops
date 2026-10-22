"""Smoke tests for the Python files that ship with the course exercises.

The course has a lot of small exercise scripts spread over the ``s*_*/exercise_files``
folders. Most of them are never executed by the docs build or by any other CI job,
so library upgrades silently break them. This module imports every exercise file in a
subprocess and reports the ones that fail, without running their ``__main__`` blocks.

The import check is deliberately shallow: it only catches import-time breakage
(syntax errors, renamed kwargs, removed functions, missing modules). Actually running
the scripts (tier 2 smoke tests) is out of scope here.

Run with::

    uv run --group exercises pytest tests/test_exercise_files.py -v

Files that cannot be imported safely (because importing them trains a model,
downloads data or talks to a cloud service) are listed in ``SKIP_IMPORT`` with a
reason. Keep that list as short as honestly possible: every entry is a file whose
imports are no longer covered by CI.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Per-file wall clock budget for the import subprocess. Importing torch alone takes a
# couple of seconds, and some files build small models at import time.
IMPORT_TIMEOUT_SECONDS = 120

# Exercise files that have module level side effects that make a plain import
# unsuitable for CI: training loops, dataset downloads, cloud calls, servers.
# Maps relative path -> short reason. Everything here is a hole in the coverage.
#
# Membership was determined by actually importing every file, both with and without
# these entries, and checking which ones hung, downloaded data, wrote files into the
# repository or needed credentials. Do not add entries "just in case".
SKIP_IMPORT: dict[str, str] = {
    # Placeholder modules that the exercise asks the student to write themselves.
    "s2_organisation_and_version_control/exercise_files/evaluate_solution.py": (
        "imports data.py/model.py, which students create in the exercise"
    ),
    "s2_organisation_and_version_control/exercise_files/train_solution.py": (
        "imports data.py/model.py, which students create in the exercise"
    ),
    "s2_organisation_and_version_control/exercise_files/visualize_solution.py": (
        "imports the cookiecutter package my_project_name, which students create in the exercise"
    ),
    "s4_debugging_and_logging/exercise_files/weights_and_bias_solution.py": (
        "imports the cookiecutter package my_project, which students create in the exercise"
    ),
    "s4_debugging_and_logging/exercise_files/weights_and_bias_solution2.py": (
        "imports the cookiecutter package my_project, which students create in the exercise"
    ),
    "s4_debugging_and_logging/exercise_files/weights_and_bias_solution3.py": (
        "imports the cookiecutter package my_project, which students create in the exercise"
    ),
    "s5_continuous_integration/exercise_files/dataset_statistics.py": (
        "imports mnist_dataset.py/utils.py, which students create in the exercise"
    ),
    # Trains, downloads or writes files at module level.
    "s3_reproducibility/exercise_files/vae_mnist.py": (
        "trains a VAE for 20 epochs on MNIST (downloaded) at import time and writes checkpoints/figures"
    ),
    "s4_debugging_and_logging/exercise_files/vae_mnist_working.py": (
        "trains a VAE on MNIST (downloaded) at import time and writes reconstruction figures"
    ),
    "s4_debugging_and_logging/exercise_files/vae_mnist_pytorch_profiler_solution.py": (
        "trains a VAE on MNIST and writes a pytorch profiler trace at import time"
    ),
    "s4_debugging_and_logging/exercise_files/vae_mnist_bugs.py": (
        "intentional bugs: trains at import time and downloads MNIST"
    ),
    "s7_deployment/exercise_files/sklearn_cloud_functions.py": (
        "fits a KNN model on iris and pickles it to model.pkl at import time"
    ),
    "s8_monitoring/exercise_files/sentiment_classifier.py": (
        "reads reviews.csv and fine-tunes BERT for 10 epochs at module level"
    ),
    "s10_extra/exercise_files/knn_iris.py": "fits a KNN pipeline on iris and writes model.joblib at import time",
    "s10_extra/exercise_files/cross_validate.py": "runs a 200-combination random forest grid search at import time",
    # Network: downloads datasets, model weights or nltk corpora at import time.
    "s8_monitoring/exercise_files/clip_features.py": "downloads CLIP weights from HuggingFace at import time",
    "s8_monitoring/exercise_files/image_drift.py": (
        "downloads MNIST and FashionMNIST at import time and writes an evidently report"
    ),
    "s8_monitoring/exercise_files/data_drift.py": "runs an evidently report at import time and writes HTML reports",
    "s8_monitoring/exercise_files/sentiment_monitoring.py": "calls nltk.download at import time",
    "s8_monitoring/exercise_files/sentiment_monitoring_starter.py": "calls nltk.download at import time",
    # Packages that are not part of the exercises dependency group.
    "s4_debugging_and_logging/exercise_files/my_logger.py": "needs loguru, which the exercise tells students to add",
    "s4_debugging_and_logging/exercise_files/my_logger_hydra.py": (
        "needs loguru and a hydra config, which the exercise tells students to add"
    ),
    "s7_deployment/exercise_files/bentoml_client.py": "needs bentoml, which is not in the exercises group",
    "s7_deployment/exercise_files/bentoml_service.py": "needs bentoml, which is not in the exercises group",
    "s7_deployment/exercise_files/bentoml_service_adaptive_batching.py": (
        "needs bentoml, which is not in the exercises group"
    ),
    "s7_deployment/exercise_files/bentoml_service_composition_concurrent.py": (
        "needs bentoml, which is not in the exercises group"
    ),
    "s7_deployment/exercise_files/bentoml_service_composition_sequential.py": (
        "needs bentoml, which is not in the exercises group"
    ),
    "s7_deployment/exercise_files/onnx_benchmark.py": (
        "exports resnet18 to ONNX and opens an onnxruntime session at import time (needs onnx/onnxscript)"
    ),
    "s7_deployment/exercise_files/onnx_simple.py": "needs onnx, which the exercise tells students to add",
    "s7_deployment/exercise_files/sklearn_main_function.py": (
        "needs functions_framework (Google Cloud Functions), which is not in the exercises group"
    ),
}


def discover_exercise_files() -> list[str]:
    """Return repo relative paths of every Python file in ``s*_*/exercise_files``."""
    pattern = "s*_*/exercise_files/**/*.py"
    return sorted(path.relative_to(REPO_ROOT).as_posix() for path in REPO_ROOT.glob(pattern))


EXERCISE_FILES = discover_exercise_files()

# Runs one file with a module name that is not ``__main__``, so ``if __name__ ==
# "__main__":`` blocks are skipped. The file's own folder goes first on sys.path so
# that ``from model import MyModel`` style sibling imports resolve.
_RUNNER = """
import runpy
import sys

path, directory = sys.argv[1], sys.argv[2]
sys.path[0] = directory
runpy.run_path(path, run_name="exercise_import_check")
"""


@pytest.fixture(scope="session")
def import_env() -> dict[str, str]:
    """Environment for the import subprocesses, with external services disabled."""
    env = os.environ.copy()
    env.update(
        {
            # Never try to reach wandb, it blocks and then fails without credentials.
            "WANDB_MODE": "disabled",
            # Keep huggingface/transformers from downloading model weights.
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            # Matplotlib must not need a display.
            "MPLBACKEND": "Agg",
            # Do not leave __pycache__ folders behind in the exercise folders.
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    return env


def test_exercise_files_are_discovered() -> None:
    """Guard against the glob silently matching nothing (e.g. after a folder rename)."""
    assert len(EXERCISE_FILES) > 50, f"only found {len(EXERCISE_FILES)} exercise files, glob is probably broken"


def test_skip_list_refers_to_existing_files() -> None:
    """A stale entry in SKIP_IMPORT hides a real file from the check, so keep it honest."""
    unknown = sorted(set(SKIP_IMPORT) - set(EXERCISE_FILES))
    assert not unknown, f"SKIP_IMPORT lists files that do not exist (anymore): {unknown}"


@pytest.mark.parametrize("relative_path", EXERCISE_FILES, ids=EXERCISE_FILES)
def test_exercise_file_imports(relative_path: str, import_env: dict[str, str]) -> None:
    """Import one exercise file in a fresh subprocess."""
    if relative_path in SKIP_IMPORT:
        pytest.skip(SKIP_IMPORT[relative_path])

    path = REPO_ROOT / relative_path
    try:
        result = subprocess.run(
            [sys.executable, "-c", _RUNNER, str(path), str(path.parent)],
            cwd=path.parent,
            env=import_env,
            capture_output=True,
            text=True,
            timeout=IMPORT_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"importing {relative_path} did not finish within {IMPORT_TIMEOUT_SECONDS}s. "
            "If it trains, downloads or waits on a server, add it to SKIP_IMPORT.",
        )

    if result.returncode != 0:
        pytest.fail(
            f"importing {relative_path} failed with exit code {result.returncode}\n"
            f"--- stdout ---\n{result.stdout[-4000:]}\n"
            f"--- stderr ---\n{result.stderr[-4000:]}",
        )
