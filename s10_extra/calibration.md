
# Calibration of ML models

!!! danger
    Module is still under development

---

## Methods

## ❔ Exercises

1. Implement a script

2. Implement temperature scaling

3. Implement label smoothing

    ```python
    alpha = 0.1
    for i in range(len(y_true)):
        y_true[i] = (1 - alpha) * y_true[i] + alpha / num_classes
    ```

    Note that the formula above assumes that `y_true` contains one-hot encoded labels.

4. Implement mixup

5. Implement cutmix

6. Implement the focal loss

7. Implement it in a continuous integration setup

## 🧠 Knowledge check
