# DTU MLOps course

Course material for the Machine Learning Operations course at DTU. These terms are used across the course modules,
starting with S8 Monitoring, and should be used with exactly these meanings.

## Monitoring

**Monitoring layers**:
The four levels a deployed model is watched at: **system** (is the machine and process healthy), **service** (are
requests answered, how fast and how often they fail), **data** (does the input still look like what the model was
trained on) and **model** (are the predictions still good).
_Avoid_: monitoring levels, metric types

**Production window**:
The requests the deployed service answered in a time span, kept as one stored image and one JSON record per request
under `production/` in GCS.
_Avoid_: production data, request log, live data, inference log

**Reference data**:
The data the live model was trained on, found through the model's lineage; drift is measured against it.
_Avoid_: baseline, training distribution, reference dataset

**Holdout set**:
A fixed set of real photos used only to compare a candidate model with the live model; it never enters `raw/`.
_Avoid_: test set, validation set, evaluation set
