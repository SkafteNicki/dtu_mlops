![Logo](../figures/icons/cml.png){ align=right width="130"}

# Continuous Machine Learning

---

The continuous integration we have looked at until now is what we can consider "classical" continuous integration, that
have its roots in DevOps and not MLOps. While the test that we have written and the containers ww have developed in the
previous session have be around machine learning, everything we have done translate to completely to how it would be
done if we had developed any other application did not include machine learning.

In this session, we are now gonna change gear and look at **continuous machine learning** (CML). As the name may suggest
we are now focusing on automatizing actual machine learning processes. You may ask why we need continuous integration
principals baked into machine learning pipelines? The reason is the same as with any continuous integration, namely that
we have a bunch of checks that we want our newly trained model to pass before we trust it. Writing `unittests` secures
that our code is not broken, but other failure modes of a machine learning pipeline should be checked before the model 
is ready for deployment:

* Did I train on the correct data?
* Did my model converge at all?
* Did it reach a certain threshold at all?

## MLOps maturity model

Before getting started with the exercises, let's first take a look at the MLOps maturity model that will help us clarify
what we are aiming for. The maturity model is a way of understanding how mature an organization is in terms of their
machine learning operations. The model is divided into five stages:

<figure markdown>
![Image](../figures/mlops_maturity_model.png){ width="1000" }
<figcaption>
<a href="https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model"> Image credit </a> 
</figcaption>
</figure>



## ❔ Exercises

In the following exercises, we are going to look at two different cases where we can use continuous machine learning. The
first one is a simple case where we are automatically going to trigger the training of a model whenever we make changes 
to our data. This is a very common use case in machine learning where we have a data pipeline that is continuously
updating our data. The second case is connected to staging and deploying models. In this case, we are going to look at
how we can automatically do further processing of our model whenever we push a new model to our repository.

1. For the first set of exercises, we are going to rely on the `cml` framework by [iterative.ai](https://iterative.ai/),
    which is a framework that is built on top of GitHub actions. The figure below describes the overall process using 
    the `cml` framework. It should be clear that it is the very same process that we go through as in the other 
    continuous integration sessions: `push code` -> `trigger GitHub actions` -> `do stuff`. The new part in this session 
    that we are only going to trigger whenever data changes.

    <figure markdown>
    ![Image](../figures/cml.jpeg){ width="1000" }
    <figcaption>
    <a href="https://towardsdatascience.com/continuous-machine-learning-e1ffb847b8da"> Image credit </a>
    </figcaption>
    </figure>

    1. If you have not already created a dataset class for the corrupted Mnist data, start by doing that. Essentially,
        it is a class that should inherit from `torch.utils.data.Dataset` and should have a `__getitem__` and `__len__`

        ??? success "Solution"

            ```python linenums="1" title="dataset.py"
            --8<-- "s5_continuous_integration/exercise_files/dataset.py"
            ```

    2. Then lets create a function that can report basic statistics such as the number of training samples, number of
        test samples, a distribution of the classes in the dataset. This function should be called `dataset_statistics`
        and should take the dataset as input.

        ??? success "Solution"

            ```python linenums="1" title="dataset.py"
            --8<-- "s5_continuous_integration/exercise_files/dataset.py"
            ```	

    3. Next, we are going to implement a Github actions workflow that only activates when we make changes to our data.
        Create a new workflow file (call it `cml_data.yaml`) and make sure it only activates on push/pull-request events
        when `data/` changes. Relevant 
        [documentation](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#onpushpull_requestpull_request_targetpathspaths-ignore)

        ??? success "Solution"

            The secret is to use the `paths` keyword in the workflow file. We here specify that the workflow should only
            trigger when the `.dvc` folder or any file with the `.dvc` extension changes, which is the case when we
            update our data and call `dvc add data/`.

            ```yaml
            
            name: DVC Workflow

            on:
            push:
                branches:
                - main
                paths:
                - '**/*.dvc'
                - '.dvc/**'
            pull_request:
                branches:
                - main
                paths:
                - '**/*.dvc'
                - '.dvc/**'
            ```

    4. The next step is to implement steps in our workflow that does something when data changes. This is the reason
        why we created the `dataset_statistics` function. Implement a workflow that:

        




    4. Now lets try to activate the workflow. 





## ❔ Exercises

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
    disp = ConfusionMatrixDisplay(confusion_matrix = confmat)
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
        steps:
          - uses: actions/checkout@v2
          - uses: iterative/setup-cml@v1
          - name: Train model
            run: |
              pip install -r requirements.txt  # install dependencies
              python train.py  # run training
          - name: Write report
            env:
              # this authenticates that the right permissions are in place
              REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
            run: |
              # send all information to report.md that will be reported to us when the workflow finish
              cat classification_report.txt >> report.md
              cml-publish confusion_matrix.png --md >> report.md
              cml-send-comment report.md
    ```

    Nearly everything in the workflow file should look familiar, except the last two lines.

3. Try pushing the workflow file to your GitHub repository and make sure that it completes.
    If it does not, you may need to adjust the workflow file slightly.

4. Send yourself a pull-request. I recommend seeing [this](https://www.youtube.com/watch?v=xwyJexAnt9k)
    very short video on how to send yourself a pull-request with a small change. If you workflow file is
    executed correctly you should see `github-actions` commenting with a performance report on your PR.

5. (Optional) `cml` is offered by the same people behind `dvc` and it should therefore come as no surprise
    that these features can interact with each other. If you want to deep dive into this,
    [here](https://cml.dev/doc/cml-with-dvc) is a great starting point.

The ends the session on continuous machine learning. If you have not already noticed, one limitation of using github
actions is that their default runners e.g. `runs-on: [ubuntu-latest]` are only CPU machines (see
[hardware config](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners#supported-runners-and-hardware-resources)
. As we all know, modern machine learning more or less requires hardware acceleration (=GPUs) to train within
reasonable time. Luckily for us `cml` also integrated with large cloud provides and I therefore recommend that
after doing through the modules on [cloud computing](../s6_the_cloud/README.md) that you return to this exercise and
experiment with setting up [self-hosted runners](https://github.com/iterative/cml#advanced-setup).
