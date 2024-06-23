
# Calibration of ML models

---




## Methods



## ❔ Exercises

1. 

2. Implement temperature scaling

3. Implement label smoothing

    ```python
    alpha = 0.1
    for i in range(len(y_true)):
        y_true[i] = (1 - alpha) * y_true[i] + alpha / num_classes
    ```

4. Implement mixup

5. Implement cutmix

6. Implement the Focal Loss

7. Implement it in a continues integration setup


## 🧠 Knowledge check

