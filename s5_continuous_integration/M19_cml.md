---
layout: default
title: M19 - Continuous Machine Learning
parent: S5 - Continuous Integration
nav_order: 5
---

# Continuous Machine Learning
{: .no_toc }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

The continuous X we have looked at until now is what we can consider "classical" continuous integration.
We are now gonna change gear and look at **continuous machine learning**. As the name may suggest we
are now focusing on automatizing actual machine learning processes (compared to automatizing unit testing). 
The automatization we are going to look at here is reporting of model performance whenever we push 
changes to our github repository.

We are going to use `cml` by [iterative.ai](https://iterative.ai/) for this session. Strictly speaking, 
then `cml` is not a necessary component for CML but it offers tools to easily get a report about how 
a specific run performed. If we where just interested in trigging model training every time we do 
a `git push` we essentially just need to include
```yaml
run: python train.py
```
to any of our workflow files. 

### Exercises

1. We are first going to revisit our `train.py` script. If we want `cml` to automatically be able 
   to report the performance of our trained model to us after it is trained, we need to give it some 
   statistics to work with. Below is some psedo-code that computes the accuracy and the confusion 
   matrix of our trained model. Create an copy of your training script (call it `train_cml.py`) and 
   make sure your script is also producing an classification report and confusion matrix as in the 
   pseudo-code.
   ```python
   # assume we have a trained model
   import matplotlib.pyplot as plt
   from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
   preds, target = [], []
   for batch in train_dataloader:
       x, y = batch
       probs = model(x)
       preds.append(probs.argmax(dim=-1))
       target.append(y.detach())

   target = torch.cat(target, dim=0)
   preds = torch.cat(preds, dim=0)

   report = classification_report(target, preds)
   with open("classification_report.txt", 'w') as outfile:
       outfile.write(report)
   confmat = confusion_matrix(target, preds)
   disp = ConfusionMatrixDisplay(cm = confmat, )
   plt.savefig('confusion_matrix.png')
   ```

2. Similar to what we have looked at until now, automation happens using *github workflow* files. 
   The main difference from continuous integration we have looked on until now, is that we are actually
   going to *train* our model whenever we do a `git push`. Copy the following code into a new workflow 
   (called `cml.yaml`) and add that file to the folder were you keep your workflow files.

    ```yaml
    name: train-my-model
    on: [push]
    jobs:
      run:
        runs-on: [ubuntu-latest]
        container: docker://iterativeai/cml:0-dvc2-base1  # continuous machine learning tools
        steps:
            - uses: actions/checkout@v2
            - name: cml_run
              env:
                  REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
              run: |
                  pip install -r requirements.txt  # install dependencies
                  python train.py  # run training

                  # send all information to report.md that will be reported to us when the workflow finish
                  cat classification_report.txt >> report.md
                  cml-publish confusion_matrix.png --md >> report.md
                  cml-send-comment report.md

    ```

3. Try pushing the workflow file to your github repository and make sure that it completes. 
   If it does not, you may need to adjust the workflow file slightly.

3. Send yourself a pull-request. I recommend seeing [this](https://www.youtube.com/watch?v=xwyJexAnt9k) 
   very short video on how to send yourself a pull-request with a small change. If you workflow file is 
   executed correctly you should see `github-actions` commenting with a performance report on your PR.

4. (Optional) `cml` is offered by the same people behind `dvc` and it should therefore come as no surprise 
   that these features can interact with each other. If you want to deep dive into this, 
   [here](https://cml.dev/doc/cml-with-dvc) is a great starting point.


