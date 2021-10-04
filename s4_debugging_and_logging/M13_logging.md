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

The most basic logging we can do is writing the metric that our model is producing to the terminal or a file for later inspection. We can then also use tools such as [matplotlib](https://matplotlib.org/) for plotting the training curve. This kind of workflow may be enough when doing smaller experiments or working alone on a project, but there is no way around using a proper experiment tracker and visualizer when doing large scale experiments in collaboration with others. It especially becomes important when you want to compare performance between different runs. Organizing monitoring is the topic of this module.

There exist many tools for logging your experiments, with some of them being:
* [Tensorboard](https://www.tensorflow.org/tensorboard)
* [Comet](https://www.comet.ml/site/)
* [MLFlow](https://mlflow.org/)
* [Neptune](https://neptune.ai/)
* [Weights and Bias](https://wandb.ai/site)

All of the frameworks offers many of the same functionalities. We are going to use Weights and Bias (wandb), as it support everything we need in this course. Additionally, it is an excellent tool for collaporation and sharing of results.

### Exercises

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

5. After running your model, checkout the webpage. Hopefully you should be able to see at least one run with something
   logged.

6. Now log something else than scalar values. This could be a image, a histogram or a matplotlib figure. In all
   cases the logging is still going to use `wandb.log` but you need extra calls to `wandb.Image` ect. depending
   on what you choose to log.

7. Finally, lets create a report that you can share. Click the **Create report** button where you choose the *blank*
   option. Then choose to include everything in the report.

8. To make sure that you have completed todays exercises, make the report shareable by clicking the *Share* button
   and create *view-only-link*. Send the link to my email `nsde@dtu.dk`, so I can checkout your awesome work 😃

9. When calling `wandb.init` you have two arguments called `project` and `entity`. Make sure that you understand these
   and try them out. It will come in handy for your group work as they essentially allows multiple users to upload their
   own runs to the same project in `wandb`.

9. Wandb also comes with build in feature for doing [hyperparameter sweeping](https://docs.wandb.ai/guides/sweeps)
   which can be beneficial to get a better working model. Look through the documentation on how to do a hyperparameter
   sweep in Wandb. You at least need to create a new file called `sweep.yaml` and make sure that you call `wandb.log`
   in your code on an appropriate value.

10. Feel free to experiment more with `wandb` as it is a great tool for logging, organizing and sharing experiments.

That is the module on logging. Please note that at this point in the course you will begin to see some overlap between the different frameworks. While we mainly used `hydra` for configuring our python scripts it can also be used to save metrics and hyperparameters similar to how `wandb` can. Similar arguments holds for `dvc` which can also be used to log metrics. In our opinion `wandb` just offers a better experience when interacting with the results after logging. We want to stress that the combination of tools presented in this course may not be the best for all your future projects, and we recommend finding a setup that fits you. That said, each framework provide specific features that the others does not.