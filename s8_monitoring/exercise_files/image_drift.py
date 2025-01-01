import numpy as np
import pandas as pd
from evidently.metrics import DataDriftTable
from evidently.report import Report
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])

mnist = datasets.MNIST(root="data", train=True, download=True, transform=transform)
fashion_mnist = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)

mnist_images = mnist.data.numpy()
fashion_images = fashion_mnist.data.numpy()


def extract_features(images):
    """Extract basic image features from a set of images."""
    features = []
    for img in images:
        avg_brightness = np.mean(img)
        contrast = np.std(img)
        sharpness = np.mean(np.abs(np.gradient(img)))
        features.append([avg_brightness, contrast, sharpness])
    return np.array(features)


mnist_features = extract_features(mnist_images)
fashion_features = extract_features(fashion_images)

feature_columns = ["Average Brightness", "Contrast", "Sharpness"]

mnist_df = np.column_stack((mnist_features, ["MNIST"] * mnist_features.shape[0]))
fashion_df = np.column_stack((fashion_features, ["FashionMNIST"] * fashion_features.shape[0]))

combined_features = np.vstack((mnist_df, fashion_df))

feature_df = pd.DataFrame(combined_features, columns=feature_columns + ["Dataset"])
feature_df[feature_columns] = feature_df[feature_columns].astype(float)

reference_data = feature_df[feature_df["Dataset"] == "MNIST"].drop(columns=["Dataset"])
current_data = feature_df[feature_df["Dataset"] == "FashionMNIST"].drop(columns=["Dataset"])

report = Report(metrics=[DataDriftTable()])
report.run(reference_data=reference_data, current_data=current_data)
report.save_html("data_drift.html")
