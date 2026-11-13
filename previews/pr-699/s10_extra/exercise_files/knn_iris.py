# Script assumes you have sklearn v1.1.3 installed
# You can update to the newest version using
# pip install -U scikit-learn
# Also you need to have matplotlib installed
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn import datasets
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

n_neighbors = 1

dataset = datasets.load_iris()
X, y = dataset.data, dataset.target
X = X[:, [0, 2]]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state=42)

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=n_neighbors)),
    ],
)

# fit model
pipeline.fit(X_train, y_train)

# predict on new data
predictions = pipeline.predict(X_test)

# save model
dump(pipeline, "model.joblib")

# load model
pipeline = load("model.joblib")

_, ax = plt.subplots()
DecisionBoundaryDisplay.from_estimator(
    pipeline,
    X,
    alpha=0.8,
    ax=ax,
    response_method="predict",
    plot_method="pcolormesh",
    shading="auto",
)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", s=20)

# hyper parameter optimization
parameters = {"knn__n_neighbors": list(range(1, 10))}
hyper_pipeline = GridSearchCV(pipeline, parameters)
hyper_pipeline.fit(X_train, y_train)
print(hyper_pipeline.best_params_)
