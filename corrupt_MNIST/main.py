import torch
import typer
from data import corrupt_mnist
from model import MyAwesomeModel
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.mps.is_available() else "cpu")
app = typer.Typer()


@app.command()
def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 5) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel().to(device)
    train_set, _ = corrupt_mnist()

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Criterions
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, train_accuracy = [], []

    for epoch in range(epochs):
        model.train()

        # Loop to also get batch index
        for i, (image, label) in enumerate(trainloader):
            image, label = image.to(device), label.to(device)

            optimizer.zero_grad()

            output = model(image)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            accuracy = (output.argmax(dim=1) == label).float().mean().item()
            train_accuracy.append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")
    torch.save(model.state_dict(), "model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(train_losses)
    axs[0].set_title("Train loss")
    axs[1].plot(train_accuracy)
    axs[1].set_title("Train accuracy")
    fig.savefig("training_statistics.png")

@app.command()
def evaluate(model_checkpoint: str, batch_size: int = 32) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel().to(device)
    state_dict = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    _, test_set = corrupt_mnist()
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Accuracy of the model on the test set: {100 * correct / total} %")

if __name__ == "__main__":
    app()