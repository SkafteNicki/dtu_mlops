# Load data
import pickle

import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris_x, iris_y = datasets.load_iris(return_X_y=True)

# Split iris data in train and test data
# A random permutation, to split the data randomly
np.random.seed(0)
indices = np.random.permutation(len(iris_x))
iris_x_train = iris_x[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_x_test = iris_x[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

# Create and fit a nearest-neighbor classifier

knn = KNeighborsClassifier()
knn.fit(iris_x_train, iris_y_train)
knn.predict(iris_x_test)

# save model

with open("model.pkl", "wb") as file:
    pickle.dump(knn, file)
