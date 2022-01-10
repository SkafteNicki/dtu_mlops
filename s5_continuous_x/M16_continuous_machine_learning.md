---
layout: default
title: M16 - Continuous Machine Learning
parent: S5 - Continuous X
nav_order: 2
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

> Note 2021 version: This exercise is not mandatory and you should only do it if you really feel like it.
  Consider checking out M27 (under the extra modules) on Pre-commit instead.

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


# Continuous docker building

`cml` integrates well with github and can give a taste of what a continuous machine learning
pipeline feels like. However, to take our applications to the next level we are going to look 
at how we can automatize docker building. As you have already seen docker building can take 
a couple of minutes to build each time we do changes to our codebase. For this reason we 
really just want to build a new image every time we do a commit of our code. Thus,
it should come as no surprise that we can also automatize the building process.

### Exercises

We are for now going to focus on automatizing out training and prediction docker images 
from session 3. You will most likely need to change the docker image for your training 
application if it contains your data e.g. you have an `COPY data/ data/` statement in it. 
Since the github repository does not store our data, we cannot copy it during the build 
process. We are later going to see how we can pull data during the build process, but 
this requires us to configure a cloud API storage account which will be part of session 6.

1. Below is an example workflow file for building a docker image. Create one of building 
   your training and one for building your prediction docker files. You will need to 
   adjust the file slightly.
   ```yaml
   name: Create Docker Container

   on: [push]

   jobs:
     mlops-container:
       runs-on: ubuntu-latest
       defaults:
         run:
           working-directory: ./week_6_github_actions
       steps:
         - name: Checkout
           uses: actions/checkout@v2
           with:
             ref: ${{ github.ref }}
         - name: Build container
           run: |
             docker build --tag inference:latest .
   ```
   Explain why it is better to have the two builds in separated workflow files instead of 1.

2. Push the files to do your github repository and make sure the workflow succeeds.

3. (Optional) To test that the container works directly in github you can also try to include an additional
   step that actually runs the container.
   ```yaml
     - name: Run container
       run: |
         docker run ...
   ```

Thats ends the session on Continuous X. We are going to revisit this topic when we get to deployment, which
is the other common factor in classical continuous X e.g. CI/CD=continuous integration and continuous deployment.
