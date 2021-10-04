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

Boilerplate is a general term that describes any *standardized* text, copy, documents, methods, or procedures that may be used over again without making major changes to the original. But how does this relate to doing machine learning projects? If you have already tried doing a couple of projects within machine learning you will probably have seen a pattern: every project usually consist of these three aspects of code:

* a model implementation
* some training code
* a collection of utilities for saving models, logging images ect.

While the latter two certainly seems important, in most cases the actual *development* or *research* often revolves around defining the model. In this sense, both the training code and the utilities becomes boilerplate that should just carry over from one project to another. But the problem usually is that we have not generalized our training code to take care of the small adjusted that may be required in future projects and we therefore end up implementing it over and over again every time that we start a new project. This is of cause a waste of our time that we should try to find a solution to.

This is where high-level frameworks comes into play. High-level frameworks are build on top of another framework (Pytorch in this case) and tries to abstract/standardize how to do particular tasks such as training. At first it may seem irritating that you need to comply to someone else code structure, however there is a good reason for that. The idea is that you can focus on what really matters (your task, model architecture ect.) and do not have to worry about the actual boilerplate that comes with 

The most popular high-level (training) frameworks within the `Pytorch` ecosystem are:

* [fast.ai](https://docs.fast.ai/)
* [Ignite](https://github.com/pytorch/ignite)
* [skorch](https://github.com/skorch-dev/skorch)
* [Catalyst](https://github.com/catalyst-team/catalyst)
* [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)

They all offer many of the same features, so choosing one over the other for most projects should not matter that much. We are here going to use `Pytorch Lightning`, as it offers all the functionality that we are going to need later in the course.

## Pytorch Lightning

In general we refer to the [documentation](https://pytorch-lightning.readthedocs.io/en/latest/) from pytorch lightning if in doubt about how to format your code for doing specific tasks. We are here going to explain the key concepts of the API that you need to understand to use the framework, starting with the `LightningModule` and the `Trainer`.

### LightningModule

The `LightningModule` is a subclass of a standard `nn.Module` that basically adds additional structure. In addition to the standard `__init__` and `forward` methods that need to be implemented in a `nn.Module`, a `LightningModule` further requires two more methods implemented:

* `training_step`: should contain your actual training code e.g. given a batch of data this should return the loss
that you want to optimize

* `configure_optimizers`: should return the optimizer that you want to use

Below is shown these two methods added to standard MNIST classifier 

<p align="center">
  <img src="../figures/lightning.png" width="700" title="hover text">
</p>

Compared to a standard `nn.Module`, the additional methods in the `LightningModule` basically specifies exactly how you
want to optimize your model. 

### Trainer

The second component to lightning is the `Trainer` object. As the name suggest, the `Trainer object takes care of the actual training, automizing everything that you do not want to worry about.

```python
from pytorch_lightning import Trainer
model = MyAwesomeModel()  # this is our LightningModule
trainer = Trainer()
traier.fit(model)
```
Thats is essentially all that you need to specify in lightning to have a working model. The trainer object does not have methods that you need to implement yourself, but it have a bunch of arguments that can be used to control how many epochs that you want to train, if you want to run on gpu ect. To get the training of our model to work we just need to specify how our data should be feed into the lighning framework.

### Data

For organizing our code that has to do with data in `Lightning` we essentially have three different options. However, all three assume that we are using `torch.utils.data.DataLoader` for the dataloading.

1. If we already have a `train_dataloader` and possible also a `val_dataloader` and `test_dataloader` defined we can simply add them to our `LightningModule` using the similar named methods:
  ```python
  def train_dataloader(self):
      return DataLoader(...)

  def val_dataloader(self):
      return DataLoader(...)

  def test_dataloader(self):
      return DataLoader(...)
  ```

2. Maybe even simplier, we can directly feed such dataloaders in the `fit` method of the `Trainer` object:
  ```python
  trainer.fit(model, train_dataloader, val_dataloader)
  trainer.test(model, test_dataloader)
  ```
 
3. Finally, `Lightning` also have the `LightningDataModule` that organizes data loading into a single structure, see this [page](https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html) for more info. Putting data loading into a `DataModule` makes sense as it is then can be reused between projects. 

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

* The `EarlyStopping` callback can help you prevent overfitting by automatically stopping the training if a certain value is not improving anymore:
  ```python
  model = MyModel()
  early_stopping_callback = EarlyStopping(
    monitor="val_loss", patience=3, verbose=True, mode="min"
  )
  trainer = Trainer(callbacks=[early_stopping_callback])
  trainer.fit(model)
  ```

Multiple callbacks can be used by passing them all in a list e.g. `Trainer(callbacks=[checkpoint_callbacks, early_stopping_callback])`

## Exercises

Please note that the in following exercise we will basically ask you to reformat all your MNIST code to follow the lightning standard, such that we can take advantage of all the tricks the framework has to offer. The reason we did not implement our model in `lightning` to begin with, is that to truly understand why it is beneficially to use a high-level framework to do some of the heavy lifting you need to have gone through some of implementation troubles yourself.

1. Convert your corrupted MNIST model into a `LightningModule`. You can either choose to completly override your old model or implement it in a new file. The bare minimum that you need to add while converting to get it working with the rest of lightning:
   
   * The `training_step` method. This function should contain essentially what goes into a single
   training step and should return the loss at the end
   
   * The `configure_optimizers` method
   
   Please read the [documentation](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html)
   for more info.
   
2. Make sure your data is formatted such that it can be loaded using the `torch.utils.data.DataLoader` object.

3. Instantiate a `Trainer` object. It is recommended to take a look at the [trainer arguments](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags) (there are many of them) and maybe adjust some of them:
         
   1. Investigate what the `default_root_dir` flag does

   2. As default lightning will run for 1000 epochs. This may be too much (for now). Change this by
      changing the appropriate flag. Additionally, there also exist a flag to set the maximum number of steps that we should train for.
       
   3. To start with we also want to limit the amount of training data to 20% of its original size. which
      trainer flag do you need to set for this to work?

4. Try fitting your model: `trainer.fit(model)`

5. Now try adding some `callbacks` to your trainer.

6. The privous module was all about logging in `wandb`, so the question is naturally how does `lightning` support this. Lightning does not only support `wandb`, but also many [others](https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html). Common for all of them, is that logging just need to happen through the `self.log` method in your `LightningModule`:

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

7. Finally, we maybe also want to do some validation or testing. In lightning we just need to add the `validation_step` and `test_step` to our lightning module and supply the respective data in form of a separate dataloader. Try to at least implement one of them.

8. (Optional, requires GPU) One of the big advantages of using `lightning` is that you no more need to deal with device placement e.g. called `.to('cuda')` everywhere. If you have a GPU, try to set the `gpus` flag in the trainer. If you do not have one, do not worry, we are going to return to this when we are going to run training in the cloud.

9. (Optional) As default Pytorch uses `float32` for representing floating point numbers. However, research have shown that neural network training is very robust towards a decrease in precision. The great benefit going from `float32` to `float16` is that we get approximately half the [memory consumption](https://www.khronos.org/opengl/wiki/Small_Float_Formats). Try out half-precision training in Pytorch lightning. You can enable this by setting the [precision](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#precision) flag in the `Trainer`.

10. (Optional) Lightning also have build-in support for profiling. Checkout how to do this using the [profiler](https://pytorch-lightning.readthedocs.io/en/latest/advanced/profiler.html#built-in-checks) argument in the `Trainer` object.

11. Free exercise: Experiment with what the lightning framework is capable of. Either try out more of the trainer flags, some of the other callbacks, or maybe look into some of the other methods that can be implemented in your lightning module. Only your imagination is the limit!

That covers everything for today. It has been a mix of topics that all should help you write "better" code (by some objective measure). If you want to deep dive more into the lightning framework, we highly recommend looking at the different tutorials in the documentation that covers more advanced models and training cases. Additionally, we also want to highlight other frameworks in the lightning ecosystem:

* [Torchmetrics](https://torchmetrics.readthedocs.io/en/latest/): collection of machine learning metrics written in Pytorch
* [lightning flash](https://lightning-flash.readthedocs.io/en/latest/): High-level framework for fast prototyping, baselining, finetuning with a even simpler interface than lightning
* [lightning-bolts](https://lightning-bolts.readthedocs.io/en/latest/): Collection of SOTA pretrained models, model components, callbacks, losses and datasets for testing out ideas as fast a possible

