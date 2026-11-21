from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


class Network(nn.Module):
    """Builds a feedforward network with arbitrary hidden layers.

    Arguments:
        input_size: integer, size of the input layer
        output_size: integer, size of the output layer
        hidden_layers: list of integers, the sizes of the hidden layers

    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: list[int],
        drop_p: float = 0.5,
    ) -> None:
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network, returns the output logits."""
        for each in self.hidden_layers:
            x = nn.functional.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)

        return nn.functional.log_softmax(x, dim=1)


def validation(
    model: nn.Module,
    testloader: torch.utils.data.DataLoader,
    criterion: Callable | nn.Module,
) -> tuple[float, float]:
    """Validation pass through the dataset."""
    accuracy = 0
    test_loss = 0
    for images, labels in testloader:
        images = images.resize_(images.size()[0], 784)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = labels.data == ps.max(1)[1]
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean().item()

    return test_loss, accuracy


def train(
    model: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    criterion: Callable | nn.Module,
    optimizer: None | torch.optim.Optimizer = None,
    epochs: int = 5,
    print_every: int = 40,
) -> None:
    """Train a PyTorch Model."""
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    steps = 0
    running_loss = 0
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for images, labels in trainloader:
            steps += 1

            # Flatten images into a 784 long vector
            images.resize_(images.size()[0], 784)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()

                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion)

                print(
                    f"Epoch: {e + 1}/{epochs}.. ",
                    f"Training Loss: {running_loss / print_every:.3f}.. ",
                    f"Test Loss: {test_loss / len(testloader):.3f}.. ",
                    f"Test Accuracy: {accuracy / len(testloader):.3f}",
                )

                running_loss = 0

                # Make sure dropout and grads are on for training
                model.train()
