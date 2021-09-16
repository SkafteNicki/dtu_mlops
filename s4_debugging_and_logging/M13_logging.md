---
layout: default
title: M13 - Experiment logging
parent: S4 - Debugging, Profiling and Logging
nav_order: 3
---

# Experiment logging
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

Experiment logging or model monitoring is an important part of understanding what is going on with your model. It can help you debug your model and help tweak your models to perfection.

The most basic logging we can do is writing the metric that our model is producing to the terminal or a file for later inspection. We can then also use tools such as [matplotlib](https://matplotlib.org/) for plotting the training curve. This kind of workflow may be enough when doing smaller experiments or working alone on a project, but there is no way around using a proper experiment tracker and visualizer when doing large scale experiments in collaboration with others. This is the topic of this module.

There exist many tools for logging your experiments, with some of them being:
* [Tensorboard](https://www.tensorflow.org/tensorboard)
* [Comet](https://www.comet.ml/site/)
* [MLFlow](https://mlflow.org/)
* [Neptune](https://neptune.ai/)
* [Weights and Bias](https://wandb.ai/site)

We are going to use Weights and Bias (wandb) as it is an excellent tool for collaboration.

### Wandb

Wandb is also called weights and biases


Please note that you will begin to see some overlap between the different frameworks that we are using. While `hydra` can be used to only configure your python scripts it can also be used to save metrics and hyperparameters. Similar arguments holds for `dvc` which can also be used to log metrics. We are therefore not saying that the combination of tools presented in this course are the best, but each one do provide specific features that the others does not.


#### Exercises




While tensorboard is a great logger for many things, more advanced loggers may be more suitable. For the remaining 
of the exercises we will try to look at the [wandb](https://wandb.ai/site) logger. The great benefit of using wandb
over tensorboard is that it was build with colllaboration in mind (whereas tensorboard somewhat got it along the
way).

1. Start by creating an account at [wandb](https://wandb.ai/site). I recommend using your github account but feel
   free to choose what you want. When you are logged in you should get an API key of length 40. Copy this for later
   use (HINT: if you forgot to copy the API key, you can find it under settings).

2. Next install wandb on your laptop
   ```
   pip install wandb
   ```

3. Now connect to your wandb account
   ```
   wandb login
   ```
   you will be asked to provide the 40 length API key. The connection should be remain open to the wandb server
   even when you close the terminal, such that you do not have to login each time. If using `wandb` in a notebook 
   you need to manually close the connection using `wandb.finish()`.

4. With it all setup we are now ready to incorporate `wandb` into our code. The interface is fairly simple, and
   this [guide](https://docs.wandb.ai/guides/integrations/pytorch) should give enough hints to get you through
   the exercise. (HINT: the two methods you need to call are `wandb.init` and `wandb.log`). To start with, logging
   the training loss of your model will be enough.

5. After running your model, checkout the webpage. Hopefully you should be able to see at least 

6. Now log something else than scalar values. This could be a image, a histogram or a matplotlib figure. In all
   cases the logging is still going to use `wandb.log` but you need extra calls to `wandb.Image` ect. depending
   on what you choose to log.

7. Finally, lets create a report that you can share. Click the **Create report** button where you choose the *blank*
   option. Then choose to include everything in the report 

8. To make sure that you have completed todays exercises, make the report shareable by clicking the *Share* button
   and create *view-only-link*. Send the link to my email `nsde@dtu.dk`, so I can checkout your awesome work.

9. Feel free to experiment more with `wandb` as it is a great tool for logging, organizing and sharing experiments.
