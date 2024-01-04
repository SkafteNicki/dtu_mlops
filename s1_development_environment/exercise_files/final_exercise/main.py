import click
import torch
from torch import nn
from model import myawesomemodel

from data import mnist
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--batch_size", default=256, help="batch size to use for training")
@click.option("--num_epochs", default=20, help="number of epochs to train for")
def train(lr, batch_size, num_epochs):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)
    print(batch_size)

    # TODO: Implement training loop here
    model = myawesomemodel.to(device)
    train_set, _ = mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} Loss {loss}")

    torch.save(model, "model.pt")


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = mnist()
    test_dataloader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=False
    )
    model.eval()

    test_preds = [ ]
    test_labels = [ ]
    with torch.no_grad():
        for batch in test_dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            test_preds.append(y_pred.argmax(dim=1).cpu())
            test_labels.append(y.cpu())

    test_preds = torch.cat(test_preds, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    print((test_preds == test_labels).float().mean())


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()