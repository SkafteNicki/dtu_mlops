import click
import matplotlib.pyplot as plt
import torch
from my_project_name.model import MyAwesomeModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


@click.command()
@click.option("model_checkpoint", default="model.pth", help="Path to model checkpoint")
@click.option("processed_dir", default="data/processed", help="Path to processed data directory")
@click.option("figure_dir", default="reports/figures", help="Path to save figures")
@click.option("figure_name", default="embeddings.png", help="Name of the figure")
def visualize(model_checkpoint: str, processed_dir: str, figure_dir: str, figure_name: str) -> None:
    """Visualize model predictions."""
    model = MyAwesomeModel().load_state_dict(torch.load(model_checkpoint))
    model.eval()
    model.fc = torch.nn.Identity()

    test_images = torch.load(f"{processed_dir}/test_images.pt")
    test_target = torch.load(f"{processed_dir}/test_target.pt")
    test_dataset = torch.utils.data.TensorDataset(test_images, test_target)

    embeddings, targets = [], []
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(test_dataset, batch_size=32):
            images, target = batch
            predictions = model(images)
            embeddings.append(predictions)
            targets.append(target)
        embeddings = torch.cat(embeddings).numpy()
        targets = torch.cat(targets).numpy()

    if embeddings.shape[1] > 500:  # Reduce dimensionality for large embeddings
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i in range(10):
        mask = targets == i
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=str(i))
    plt.legend()
    plt.savefig(f"{figure_dir}/{figure_name}")
