import logging
import operator
import os

import typer
import wandb
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()


def stage_best_model_to_registry(model_name: str, metric_name: str = "accuracy", higher_is_better: bool = True) -> None:
    """
    Stage the best model to the model registry.

    Args:
        model_name: Name of the model to be registered.
        metric_name: Name of the metric to choose the best model from.
        higher_is_better: Whether higher metric values are better.

    """
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )
    artifact_collection = api.artifact_collection(type_name="model", name=model_name)

    best_metric = float("-inf") if higher_is_better else float("inf")
    compare_op = operator.gt if higher_is_better else operator.lt
    best_artifact = None
    for artifact in list(artifact_collection.artifacts()):
        if metric_name in artifact.metadata and compare_op(artifact.metadata[metric_name], best_metric):
            best_metric = artifact.metadata[metric_name]
            best_artifact = artifact

    if best_artifact is None:
        logging.error("No model found in registry.")
        return

    logger.info(f"Best model found in registry: {best_artifact.name} with {metric_name}={best_metric}")
    best_artifact.link(
        target_path=f"{os.getenv('WANDB_ENTITY')}/model-registry/{model_name}",
        aliases=["best", "staging"],
    )
    best_artifact.save()
    logger.info("Model staged to registry.")


if __name__ == "__main__":
    typer.run(stage_best_model_to_registry)
