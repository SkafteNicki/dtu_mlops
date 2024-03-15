![Logo](../figures/icons/prefect.png){ align=right width="130"}

!!! danger
    Module is still under development

# Workflow orchestration

## Prefect

If you give an MLOps engineer a job

* Could you just set up this pipeline to train this model?
* Could you set up logging?
* Could you do it every day?
* Could you make it retry if it fails?
* Could you send me a message when it succeeds?
* Could you visualize the dependencies?
* Could you add caching?
* Could you add add collaborators to run ad hoc - who don't code e.g could you add a UI?

```bash
pip install prefect
```

```python
from prefect import task, Flow
```

### ❔ Exercises

1. Start by installing `prefect`:

    ```bash
    pip install prefect
    ```

2. Start a local Prefect server instance in your virtual environment.

    ```bash
    prefect server start
    ```

3. The great thing about Prefect is that the orchestration tasks and flows are written in pure Python.
