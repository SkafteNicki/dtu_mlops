import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

mnist_train = datasets.MNIST("", download=True, train=True)
X_train, y_train = mnist_train.data, mnist_train.targets
mnist_test = datasets.MNIST("", download=True, train=False)
X_test, y_test = mnist_test.data, mnist_test.targets

T = transforms.RandomRotation(50)
for i in range(10):
    torch.save(T(X_train[5000 * i : 5000 * (i + 1)]) / 255.0, f"train_images_{i}.pt")
    torch.save(y_train[5000 * i : 5000 * (i + 1)], f"train_target_{i}.pt")
torch.save(T(X_test[:5000]) / 255.0, "test_images.pt")
torch.save(y_test[:5000], "test_target.pt")

fig, axes = plt.subplots(nrows=2, ncols=10)

for i in range(10):
    axes[0][i].imshow(X_train[5000 * i])
    axes[1][i].imshow(T(X_train[5000 * i].unsqueeze(0).unsqueeze(0)).squeeze())

plt.show()
