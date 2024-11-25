![Logo](../figures/icons/lightning.png){ align=right width="130"}

# Minimizing boilerplate

---

Boilerplate is a general term that describes any *standardized* text, copy, documents, methods, or procedures that may
be used over again without making major changes to the original. But how does this relate to doing machine learning
projects? If you have already tried doing a couple of projects within machine learning you will probably have seen a
pattern: every project usually consist of these three aspects of code:

* a model implementation
* some training code
* a collection of utilities for saving models, logging images etc.

While the latter two certainly seems important, in most cases the actual *development* or *research* often revolves
around defining the model. In this sense, both the training code and the utilities becomes boilerplate that should just
carry over from one project to another. But the problem usually is that we have not generalized our training code to
take care of the small adjusted that may be required in future projects and we therefore end up implementing it over
and over again every time that we start a new project. This is of course a waste of our time that we should try to
find a solution to.

This is where high-level frameworks comes into play. High-level frameworks are build on top of another framework
(PyTorch in this case) and tries to abstract/standardize how to do particular tasks such as training. At first it may
seem irritating that you need to comply to someone else code structure, however there is a good reason for that. The
idea is that you can focus on what really matters (your task, model architecture etc.) and do not have to worry about
the actual boilerplate that comes with it.

The most popular high-level (training) frameworks within the `PyTorch` ecosystem are:

* [fast.ai](https://github.com/fastai/fastai)
* [Ignite](https://github.com/pytorch/ignite)
* [skorch](https://github.com/skorch-dev/skorch)
* [Catalyst](https://github.com/catalyst-team/catalyst)
* [Composer](https://github.com/mosaicml/composer)
* [PyTorch Lightning](https://github.com/Lightning-AI/lightning)

They all offer many of the same features, so choosing one over the other for most projects should not matter that much.
We are here going to use `PyTorch Lightning`, as it offers all the functionality that we are going to need later in the
course.

## PyTorch Lightning

In general we refer to the [documentation](https://lightning.ai/docs/pytorch/stable/) from PyTorch lightning
if in doubt about how to format your code for doing specific tasks. We are here going to explain the key concepts of
the API that you need to understand to use the framework, starting with the `LightningModule` and the `Trainer`.

### LightningModule

The `LightningModule` is a subclass of a standard `nn.Module` that basically adds additional structure. In addition to
the standard `__init__` and `forward` methods that need to be implemented in a `nn.Module`, a `LightningModule` further
requires two more methods implemented:

* `training_step`: should contain your actual training code e.g. given a batch of data this should return the loss
    that you want to optimize

* `configure_optimizers`: should return the optimizer that you want to use

Below is shown these two methods added to standard MNIST classifier

<figure markdown>
![Image](../figures/lightning.png){width="700" }
</figure>

Compared to a standard `nn.Module`, the additional methods in the `LightningModule` basically specifies exactly how you
want to optimize your model.

### Trainer

The second component to lightning is the `Trainer` object. As the name suggest, the `Trainer` object takes care of the
actual training, automizing everything that you do not want to worry about.

```python
from pytorch_lightning import Trainer
model = MyAwesomeModel()  # this is our LightningModule
trainer = Trainer()
traier.fit(model)
```

That's is essentially all that you need to specify in lightning to have a working model. The trainer object does not
have methods that you need to implement yourself, but it have a bunch of arguments that can be used to control how many
epochs that you want to train, if you want to run on gpu etc. To get the training of our model to work we just need to
specify how our data should be feed into the lighning framework.

### Data

For organizing our code that has to do with data in `Lightning` we essentially have three different options. However,
all three assume that we are using `torch.utils.data.DataLoader` for the dataloading.

1. If we already have a `train_dataloader` and possible also a `val_dataloader` and `test_dataloader` defined we can
    simply add them to our `LightningModule` using the similar named methods:

    ```python
    def train_dataloader(self):
        return DataLoader(...)

    def val_dataloader(self):
        return DataLoader(...)

    def test_dataloader(self):
        return DataLoader(...)
    ```

2. Maybe even simpler, we can directly feed such dataloaders in the `fit` method of the `Trainer` object:

    ```python
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)
    ```

3. Finally, `Lightning` also have the `LightningDataModule` that organizes data loading into a single structure, see
    this [page](https://lightning.ai/docs/pytorch/latest/data/datamodule.html) for more info. Putting
    data loading into a `DataModule` makes sense as it is then can be reused between projects.

### Callbacks

Callbacks is one way to add additional functionality to your model, that strictly speaking is not already part of your
model. Callbacks should therefore be seen as self-contained feature that can be reused between projects. You have the
option to implement callbacks yourself (by inheriting from the `pytorch_lightning.callbacks.Callback` base class) or
use one of the
[build in callbacks](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html#built-in-callbacks).
Of particular interest are `ModelCheckpoint` and `EarlyStopping` callbacks:

* The `ModelCheckpoint` makes sure to save checkpoints of you model. This is in principal not hard to do yourself, but
    the `ModelCheckpoint` callback offers additional functionality by saving checkpoints only when some metric improves,
    or only save the best `K` performing models etc.

    ```python
    model = MyModel()
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_loss", mode="min"
    )
    trainer = Trainer(callbacks=[checkpoint_callbacks])
    trainer.fit(model)
    ```

* The `EarlyStopping` callback can help you prevent overfitting by automatically stopping the training if a certain
    value is not improving anymore:

    ```python
    model = MyModel()
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )
    trainer = Trainer(callbacks=[early_stopping_callback])
    trainer.fit(model)
    ```

Multiple callbacks can be used by passing them all in a list e.g.

```python
trainer = Trainer(callbacks=[checkpoint_callbacks, early_stopping_callback])
```

## ❔ Exercises

Please note that the in following exercise we will basically ask you to reformat all your MNIST code to follow the
lightning standard, such that we can take advantage of all the tricks the framework has to offer. The reason we did not
implement our model in `lightning` to begin with, is that to truly understand why it is beneficially to use a high-level
framework to do some of the heavy lifting you need to have gone through some of implementation troubles yourself.

1. Install pytorch lightning:

    ```bash
    pip install pytorch-lightning # (1)!
    ```

    1. :man_raising_hand: You may also install it as `pip install lightning` which includes more than just the
        `PyTorch Lightning` package. This also includes `Lightning Fabric` and `Lightning Apps` which you can read more
        about [here](https://lightning.ai/docs/fabric/stable/) and [here](https://lightning.ai/docs/app/stable/).

2. Convert your corrupted MNIST model into a `LightningModule`. You can either choose to completely override your old
    model or implement it in a new file. The bare minimum that you need to add while converting to get it working with
    the rest of lightning:

    * The `training_step` method. This function should contain essentially what goes into a single
        training step and should return the loss at the end

    * The `configure_optimizers` method

    Please read the [documentation](https://lightning.ai/docs/pytorch/latest/common/lightning_module.html)
    for more info.

    ??? success "Solution"

        ```python linenums="1" hl_lines="23" title="lightning.py"
        --8<-- "s4_debugging_and_logging/exercise_files/lightning_solution.py"
        ```

3. Make sure your data is formatted such that it can be loaded using the `torch.utils.data.DataLoader` object.

4. Instantiate a `Trainer` object. It is recommended to take a look at the
    [trainer arguments](https://lightning.ai/docs/pytorch/latest/common/trainer.html#trainer-flags) (there
    are many of them) and maybe adjust some of them:

    1. Investigate what the `default_root_dir` flag does

    2. As default lightning will run for 1000 epochs. This may be too much (for now). Change this by
        changing the appropriate flag. Additionally, there also exist a flag to set the maximum number of steps that we
        should train for.

        ??? success "Solution"

            Setting the `max_epochs` will accomplish this.

            ```python
            trainer = Trainer(max_epochs=10)
            ```

            Additionally, you may consider instead setting the `max_steps` flag to limit based on the number of steps or
            `max_time` to limit based on time. Similarly, the flags `min_epochs`, `min_steps` and `min_time` can be used
            to set the minimum number of epochs, steps or time.

    3. To start with we also want to limit the amount of training data to 20% of its original size. which
        trainer flag do you need to set for this to work?

        ??? success "Solution"

            Setting the `limit_train_batches` flag will accomplish this.

            ```python
            trainer = Trainer(limit_train_batches=0.2)
            ```

            Similarly, you can also set the `limit_val_batches` and `limit_test_batches` flags to limit the validation
            and test data.

5. Try fitting your model: `trainer.fit(model)`

6. Now try adding some `callbacks` to your trainer.

    ??? success "Solution"

        ```python
        early_stopping_callback = EarlyStopping(
            monitor="val_loss", patience=3, verbose=True, mode="min"
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath="./models", monitor="val_loss", mode="min"
        )
        trainer = Trainer(callbacks=[early_stopping_callback, checkpoint_callback])
        ```

7. The privous module was all about logging in `wandb`, so the question is naturally how does `lightning` support this.
    Lightning does not only support `wandb`, but also many
    [others](https://lightning.ai/docs/pytorch/latest/extensions/logging.html). Common for all of them, is that
    logging just need to happen through the `self.log` method in your `LightningModule`:

    1. Add `self.log` to your `LightningModule. Should look something like this:

        ```python
        def training_step(self, batch, batch_idx):
            data, target = batch
            preds = self(data)
            loss = self.criterion(preds, target)
            acc = (target == preds.argmax(dim=-1)).float().mean()
            self.log('train_loss', loss)
            self.log('train_acc', acc)
            return loss
        ```

    2. Add the `wandb` logger to your trainer

        ```python
        trainer = Trainer(logger=pl.loggers.WandbLogger(project="dtu_mlops"))
        ```

        and try to train the model. Confirm that you are seeing the scalars appearing in your `wandb` portal.

    3. `self.log` does sadly only support logging scalar tensors. Luckily, for logging other quantities we
        can still access the standard `wandb.log` through our model

        ```python
        def training_step(self, batch, batch_idx):
            ...
            # self.logger.experiment is the same as wandb.log
            self.logger.experiment.log({'logits': wandb.Histrogram(preds)})
        ```

        try doing this, by logging something else than scalar tensors.

8. Finally, we maybe also want to do some validation or testing. In lightning we just need to add the `validation_step`
    and `test_step` to our lightning module and supply the respective data in form of a separate dataloader. Try to at
    least implement one of them.

    ??? success "Solution"

        Both validation and test steps can be implemented in the same way as the training step:

        ```python
        def validation_step(self, batch) -> None:
            data, target = batch
            preds = self(data)
            loss = self.criterion(preds, target)
            acc = (target == preds.argmax(dim=-1)).float().mean()
            self.log('val_loss', loss, on_epoch=True)
            self.log('val_acc', acc, on_epoch=True)
        ```

        two things to take note of here is that we are setting the `on_epoch` flag to `True` in the `self.log` method.
        This is because we want to log the validation loss and accuracy only once per epoch. Additionally, we are not
        returning anything from the `validation_step` method, because we do not optimize over the loss.

9. (Optional, requires GPU) One of the big advantages of using `lightning` is that you no more need to deal with device
    placement e.g. called `.to('cuda')` everywhere. If you have a GPU, try to set the `gpus` flag in the trainer. If you
    do not have one, do not worry, we are going to return to this when we are going to run training in the cloud.

    ??? success "Solution"

        The two arguments `accelerator` and `devices` can be used to specify which devices to run on and how many to run
        on. For example, to run on a single GPU you can do

        ```python
        trainer = Trainer(accelerator="gpu", devices=1)
        ```

        as an alternative the accelerator can just be set to `#!python accelerator="auto"` to automatically detect the
        best available device.

10. (Optional) As default PyTorch uses `float32` for representing floating point numbers. However, research have shown
    that neural network training is very robust towards a decrease in precision. The great benefit going from `float32`
    to `float16` is that we get approximately half the
    [memory consumption](https://en.wikipedia.org/wiki/Half-precision_floating-point_format). Try out half-precision
    training in PyTorch lightning. You can enable this by setting the
    [precision](https://lightning.ai/docs/pytorch/latest/common/trainer.html#precision) flag in the `Trainer`.

    ??? success "Solution"

        Lightning supports four different types of mixed precision training (16-bit and 16-bit bfloat) and two types of:

        ```python
        # 16-bit mixed precision (model weights remain in torch.float32)
        trainer = Trainer(precision="16-mixed", devices=1)

        # 16-bit bfloat mixed precision (model weights remain in torch.float32)
        trainer = Trainer(precision="bf16-mixed", devices=1)

        # 16-bit precision (model weights get cast to torch.float16)
        trainer = Trainer(precision="16-true", devices=1)

        # 16-bit bfloat precision (model weights get cast to torch.bfloat16)
        trainer = Trainer(precision="bf16-true", devices=1)
        ```

11. (Optional) Lightning also have built-in support for profiling. Checkout how to do this using the
    [profiler](https://lightning.ai/docs/pytorch/latest/tuning/profiler_basic.html) argument in
    the `Trainer` object.

12. (Optional) Another great feature of Lightning is that the allow for easily defining command line interfaces through
    the [Lightning CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) feature. The
    Lightning CLI is essentially a drop in replacement for defining command line interfaces (covered in
    [this module](../s2_organisation_and_version_control/cli.md)) and can also replace the need for config files
    (covered in [this module](../s3_reproducibility/config_files.md)) for securing reproducibility when working inside
    the Lightning framework. We highly recommend checking out the feature and that you try to refactor your code such
    that you do not need to call `trainer.fit` anymore but it is instead directly controlled from the Lightning CLI.

13. Free exercise: Experiment with what the lightning framework is capable of. Either try out more of the trainer flags,
    some of the other callbacks, or maybe look into some of the other methods that can be implemented in your lightning
    module. Only your imagination is the limit!

That covers everything for today. It has been a mix of topics that all should help you write "better" code (by some
objective measure). If you want to deep dive more into the PyTorch lightning framework, we highly recommend looking at
the different tutorials in the documentation that covers more advanced models and training cases. Additionally, we also
want to highlight other frameworks in the lightning ecosystem:

* [Torchmetrics](https://torchmetrics.readthedocs.io/en/latest/): collection of machine learning metrics written
    in PyTorch
* [lightning flash](https://lightning-flash.readthedocs.io/en/latest/): High-level framework for fast prototyping,
    baselining, finetuning with a even simpler interface than lightning
* [lightning-bolts](https://lightning-bolts.readthedocs.io/en/latest/): Collection of SOTA pretrained models, model
    components, callbacks, losses and datasets for testing out ideas as fast a possible
