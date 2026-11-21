import json

from evidently import DataDefinition, Dataset, Report
from evidently.metrics import DatasetMissingValueCount, RowCount, ValueDrift
from evidently.presets import DataDriftPreset, DataSummaryPreset
from evidently.tests import eq, gte
from sklearn import datasets

iris_frame = datasets.load_iris(as_frame=True).frame
iris_frame = iris_frame.rename(
    columns={
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
    }
)

# Tell evidently how to interpret each column
feature_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
data_definition = DataDefinition(numerical_columns=feature_columns, categorical_columns=["target"])
reference_data = Dataset.from_pandas(iris_frame.iloc[60:], data_definition=data_definition)
current_data = Dataset.from_pandas(iris_frame.iloc[:60], data_definition=data_definition)

# Report with data drift, data summary (quality) and target drift
report = Report(
    metrics=[
        DataDriftPreset(columns=feature_columns),  # drift of the input features
        DataSummaryPreset(),  # data quality
        ValueDrift(column="target"),  # drift of the target
    ],
)
snapshot = report.run(current_data=current_data, reference_data=reference_data)
snapshot.save_html("report.html")

# Tests: conditions attached to metrics, results extracted programmatically
test_report = Report(
    metrics=[
        DatasetMissingValueCount(tests=[eq(0)]),
        RowCount(tests=[gte(50)]),
        DataDriftPreset(),
    ],
    include_tests=True,  # also add the default tests of the preset
)
test_snapshot = test_report.run(current_data=current_data, reference_data=reference_data)
test_snapshot.save_html("tests.html")

test_results = json.loads(test_snapshot.json())["tests"]
for test in test_results:
    print(f"{test['status']:>8}: {test['name']}")
print("All tests passed:", all(test["status"] == "SUCCESS" for test in test_results))
