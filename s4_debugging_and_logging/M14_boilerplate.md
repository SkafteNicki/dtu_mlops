---
layout: default
title: M14 - Minimizing boilerplate
parent: S4 - Debugging, Profiling and Logging
nav_order: 4
---

# Minimizing boilerplate
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






Boilerplate is a general term that describes any *standardized* text, copy, documents, methods, or procedures that may be used over again without making major changes to the original. But how 


This is very training frameworks comes into play. Training frameworks are build on top of another framework (Pytorch in this case) and tries to standardize how particular training setups should look. At first it may seem irretating that you need to comply to someone elses code structure, however there is a good reason for that. The idea is that you can focus on what really matters (your task, model architechture) and do not have to worry about the actual boilerplate that comes with 

The most popular training frameworks within the `Pytorch` ecosystem:
* [fast.ai](https://docs.fast.ai/)
* [Ignite](https://github.com/pytorch/ignite)
* [skorch](https://github.com/skorch-dev/skorch)
* [Catalyst](https://github.com/catalyst-team/catalyst)
* [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)

They all offer many of the same features, so choosing one over the other for most projects should not matter that much. We are here going to use `Pytorch Lightning`, as it offers all the functionality that we are going to need later in the course.




## Pytorch Lightning

In general we refer to the [documentation](https://pytorch-lightning.readthedocs.io/en/latest/) from pytorch lightning if in doubt about how to format your code for doing specific tasks.

The two core API in lightning that we need to understand is the `LightningModule` and the `Trainer`.

### LightningModule

The `LightningModule` is a subclass of a standard `nn.Module` that basically adds additional structure. In addition to the standard `__init__` and `forward` methods that need to be implemented in a `nn.Module`, a `LightningModule` further requires two more methods implemented:

* `training_step`: should contain your actual training code e.g. given a batch of data this should return the loss
that you want to optimize

* `configure_optimizers`: should return the optimizer that you want to use

<p align="center">
  <img src="../figures/lightning.png" width="700" title="hover text">
</p>


### Trainer

### Data

For organizing our code that has to do with data in `Lightning` we essentially have three different options. However, all three assume that we are using `torch.utils.data.DataLoader` for the dataloading.

* If we already have a `train_dataloader` and possible also a `val_dataloader` and `test_dataloader` defined we can simply add them to our `LightningModule` using the similar named methods:
  ```python
  def train_dataloader(self):
      return DataLoader(...)

  def val_dataloader(self):
      return DataLoader(...)

  def test_dataloader(self):
      return DataLoader(...)

* Maybe even simplier, we can directly feed such dataloaders in the `fit` method of the `Trainer` object:
  ```python
  trainer.fit(model, train_dataloader, val_dataloader)
  trainer.test(model, test_dataloader)

* Finally, `Lightning` also have the `LightningDataModule` that organizes data loading into a single structure, see this [page](https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html) for more info. Putting data loading into a `DataModule` makes sense as it is then can be reused between projects. 

### Callbacks

Callbacks is one way to add additional functionality to your model, that strictly speaking is not already part of your model. Callbacks should therefore be seen as self-contained feature that can be reused between projects. You have the option to implement callbacks yourself (by inheriting from the `pytorch_lightning.callbacks.Callback` base class) or use one of the [build in callbacks](https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html#built-in-callbacks). Of particular interest are `ModelCheckpoint` and `EarlyStopping` callbacks:

* The `ModelCheckpoint` makes sure to save checkpoints of you model. This is in pricipal not hard to do yourself, but the `ModelCheckpoint` callback offers
  additional functionality by saving checkpoints only when some metric improves, or only save the best `K` performing models ect.
  ```python
  model = MyModel()
  checkpoint_callback = ModelCheckpoint(
    dirpath="./models", monitor="val_loss", mode="min"
  )
  trainer = Trainer(callbacks=[checkpoint_callbacks])
  trainer.fit(model)
  ```

* The `EarlyStopping` callback can help you prevent overfitting by automatically stopping the training if a certain value is not improving.
  ```python
  model = MyModel()
  early_stopping_callback = EarlyStopping(
    monitor="val_loss", patience=3, verbose=True, mode="min"
  )
  trainer = Trainer(callbacks=[early_stopping_callback])
  trainer.fit(model)
  ```

Multiple callbacks can be used by passing them all in a list e.g. `Trainer(callbacks=[checkpoint_callbacks, early_stopping_callback])`

### Exercises

Please note that the in following exercise we will basically ask you to reformat all your Mnist code to follow the lightning standard, such
that we can take advantage of all the tricks the framework has to offer. The reason we did not implement lightning in the beginning was that
to truely understand why it is beneficially to use someones else framework to do some of the heavylifting you need to have gone through the
steps 



8. (Optional) Lightning of cause also have build in support for 


Thats everything for today, covering a verity of topics. If you want to deep dive more into the topics of today we highly recommend:
* 
