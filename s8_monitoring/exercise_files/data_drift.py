from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.report import Report
from evidently.test_preset import DataStabilityTestPreset
from evidently.test_suite import TestSuite
from sklearn import datasets

iris_frame = datasets.load_iris(as_frame=True).frame

data_drift_report = Report(
    metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        TargetDriftPreset(),
    ],
)

data_drift_report.run(
    current_data=iris_frame.iloc[:60],
    reference_data=iris_frame.iloc[60:],
    column_mapping=None,
)
data_drift_report.save_html("test.html")

data_stability = TestSuite(
    tests=[
        DataStabilityTestPreset(),
    ],
)
data_stability.run(
    current_data=iris_frame.iloc[:60],
    reference_data=iris_frame.iloc[60:],
    column_mapping=None,
)
data_stability.save_html("test2.html")
