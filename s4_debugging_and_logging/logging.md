![Logo](../figures/icons/w&b.png){ align=right width="130"}

# Logging

---

!!! info "Core Module"

[Logging](https://en.wikipedia.org/wiki/Logging_(computing)) in general refers to the practise of recording events
activities over time. Having proper logging in your applications can be extremely beneficial for a few reasons:

* *Debugging* becomes easier because we in a more structure way can output information about the state of our program,
    variables, values etc. to help identify and fix bugs or unexpected behavior.

* When we move into a more production environment, proper logging is essential for *monitoring* the health and
    performance of our application.

* It can help in *auditing* as logging info about specific activities etc. can help keeping a record of who did what
    and when.

* Having proper logging means that info is saved for later, that can be *analysed* to gain insight into the behavior of
    our application, such as trends.

We are in this course going to divide the kind of logging we can do into categories: application logging and experiment
logging. In general application logging is important regardless of the kind of application you are developing, whereas
experiment logging is important machine learning based projects where we are doing experiments.

## Application logging

The most basic form of logging in Python applications is the good old `print` statement:

```python
for batch_idx, batch in enumerate(dataloader):
    print(f"Processing batch {batch_idx} out of {len(dataloader)}")
    ...
```

This will keep a "record" of the events happening in our script, in this case how far we have progressed. We could even
change the print to include something like `batch.shape` to also have information about the current data being
processed.

Using `print` statements is fine for small applications, but to have proper logging we need a bit more functionality
than what `print` can offer. Python actually comes with a great
[logging](https://docs.python.org/3/library/logging.html) module, that defines functions for flexible
logging. It is exactly this we are going to look at in this module.

The four main components to the Python logging module are:

1. *Logger*: The main entry point for using the logging system. You create instances of the Logger class to emit log
    messages.

2. *Handler*: Defines where the log messages go. Handlers send the log messages to specific destinations, such as the
    console or a file.

3. *Formatter*: Specifies the layout of the log messages. Formatters determine the structure of the log records,
    including details like timestamps and log message content.

4. *Level*: Specifies the severity of a log message.

Especially, the last point is important to understand. Levels essentially allows of to get rid of statements like this:

```python
if debug:
    print(x.shape)
```

where the logging is conditional on the variable `debug` which we can set a runtime. Thus, it is something we can
disable for users of our application (`debug=False`) but have enabled when we develop the application (`debug=True`).
And it makes sense that not all things logged, should be available to all stakeholders of a codebase. We as developers
probably always wants the highest level of logging, whereas users of the our code need less info and we may want to
differentiate this based on users.

<figure markdown>
![Image](../figures/logging_level.drawio.png){ width="800" }
</figure>

It is also important to understand the different between logging and error handling. Error handling
Python is done using `raise` statements and `try/catch` like:

```python
def f(x: int):
    if not isinstance(x, int):
        raise ValueError("Expected an integer")
    return 2 * x

try:
    f(5):
except ValueError:
    print("I failed to do a thing, but continuing.")
```

Why would we evere need log `warning`, `error`, `critical` levels of information, if we are just going to handle it?
The reason is that raising exceptions are meant to change the *program flow at runtime* e.g. things we do not want the
user to do, but we can deal with in some way. Logging is always for *after* a program have run, to inspect what went
wrong. Sometimes you need one, sometimes the other, sometimes both.

### ❔ Exercises

Exercises are inspired by this [made with ml](https://madewithml.com/courses/mlops/logging/) module on the same topic.
If you need help for the exercises you can find a simple solution script
[here](https://github.com/SkafteNicki/dtu_mlops/tree/main/s4_debugging_and_logging/exercise_files).

1. As logging is a built-in module in Python, nothing needs to be installed. Instead start a new file called
    `my_logger.py` and start out with the following code:

    ```python
    import logging
    import sys

    # Create super basic logger
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logget.getLogger(__name__) # (1)

    # Logging levels (from lowest to highest priority)
    logger.debug("Used for debugging your code.")
    logger.info("Informative messages from your code.")
    logger.warning("Everything works but there is something to be aware of.")
    logger.error("There's been a mistake with the process.")
    logger.critical("There is something terribly wrong and process may terminate.")
    ```

    1. :man_raising_hand: The built-in variable `__name__` always contains the record of the script or module that is
        currently being run. Therefore if we initialize our logger base using this variable, it will always be unique
        to our application and not conflict with logger setup by any third-party package.

    Try running the code. Than try changing the argument `level` when creating the logger. What happens when you do
    that?

2. Instead of sending logs to the terminal, we may also want to send them to a file. This can be beneficial, such that
    only `warning` level logs and higher are available to the user, but `debug` and `info` is still saved when the
    application is running.

    1. Try adding the following dict to your `logger.py` file:

        ```python
        logging_config = {
            "version": 1,
            "formatters": { # (1)
                "minimal": {"format": "%(message)s"},
                "detailed": {
                    "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
                },
            },
            "handlers": { # (2)
                "console": {
                    "class": "logging.StreamHandler",
                    "stream": sys.stdout,
                    "formatter": "minimal",
                    "level": logging.DEBUG,
                },
                "info": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": Path(LOGS_DIR, "info.log"),
                    "maxBytes": 10485760,  # 1 MB
                    "backupCount": 10,
                    "formatter": "detailed",
                    "level": logging.INFO,
                },
                "error": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": Path(LOGS_DIR, "error.log"),
                    "maxBytes": 10485760,  # 1 MB
                    "backupCount": 10,
                    "formatter": "detailed",
                    "level": logging.ERROR,
                },
            },
            "root": {
                "handlers": ["console", "info", "error"],
                "level": logging.INFO,
                "propagate": True,
            },
        }
        ```

        1. :man_raising_hand: The [formatter](https://docs.python.org/3/library/logging.html#formatter-objects) section
            determines how logs should be formatted. Here we define two separate formatters, called `minimal` and
            `detailed` which we can use in the next part of the code.

        2. :man_raising_hand: The [handlers](https://docs.python.org/3/library/logging.html#handler-objects) is in
            charge of what should happen to different level of logging. `console` uses the `minimal` format we defined
            and sens logs to the `stdout` stream for messages of level `DEBUG` and higher. The `info` handler uses
            the `detailed` format and sends messages of level `INFO` and higher to a separate `info.log` file. The
            `error` handler does the same for messages of level `ERROR` and higher to a file called `error.log`.

        you will need to set the `LOGS_DIR` variable and also figure out how to add this `logging_config` using the
        [logging config](https://docs.python.org/3/library/logging.config.html#module-logging.config) submodule to your
        logger.

    2. When the code successfully runs, check the `LOGS_DIR` folder and make sure that a `info.log` and `error.log` file
        was created with the appropriate content.

3. Finally, lets try to add a little bit of style and color to our logging. For this we can use the great package
    [rich](https://github.com/Textualize/rich) which is a great package for *rich* text and beautiful formatting in
    terminals. Install `rich` and add the following line to your `my_logger.py`
    script:

    ```python
    logger.handlers[0] = RichHandler(markup=True)  # set rich handler
    ```

    and try re-running the script. Hopefully you should see something buitiful in your terminal like this:

    <figure markdown>
    ![Image](../figures/rich_terminal_logging.png){ width="700" }
    </figure>

4. (Optional) We already briefly touched on logging during the
    [module on config files using hydra](../s3_reproducibility/config_files.md). If you want to configure hydra to use
    custom logging scheme as the one we setup in the last two exercises, you can take a look at this
    [page](https://hydra.cc/docs/configure_hydra/logging/). In hydra you will need to provide the configuration of the
    logger as config file. You can find examples of such config file
    [here](https://github.com/SkafteNicki/dtu_mlops/tree/main/s4_debugging_and_logging/exercise_files).

## Experiment logging

When most people think machine learning, we think about the training phase. Being able to track and log experiments
is an important part of understanding what is going on with your model while you are training. It can help you debug
your model and help tweak your models to perfection. Without proper logging of experiments, it can be really hard to
iterate on the model because you do not know what changes lead to increase or decrease in performance.

The most basic logging we can do when running experiments is writing the metrics that our model is producing e.g. the
loss or the accuracy to the terminal or a file for later inspection. We can then also use tools such as
[matplotlib](https://matplotlib.org/) for plotting the progression of our metrics over time. This kind of workflow may
be enough when doing smaller experiments or working alone on a project, but there is no way around using a proper
experiment tracker and visualizer when doing large scale experiments in collaboration with others. It especially becomes
important when you want to compare performance between different runs.

There exist many tools for logging your experiments, with some of them being:

* [Tensorboard](https://www.tensorflow.org/tensorboard)
* [Comet](https://www.comet.ml/site/)
* [MLFlow](https://mlflow.org/)
* [Neptune](https://neptune.ai/)
* [Weights and Bias](https://wandb.ai/site)

All of the frameworks offers many of the same functionalities, you can see a (bias) review
[here](https://dagshub.com/blog/best-8-experiment-tracking-tools-for-machine-learning-2023/). We are going to use
Weights and Bias (wandb), as it support everything we need in this course. Additionally, it is an excellent tool for
collaboration and sharing of results.

<figure markdown>
![Image](../figures/wandb_example.png){ width="700" }
<figcaption>
Using the Weights and Bias (wandb) dashboard we can quickly get an overview and compare many runs over different
metrics. This allows for better iteration of models and training procedure.
</figure>

### ❔ Exercises

1. Start by creating an account at [wandb](https://wandb.ai/site). I recommend using your github account but feel
    free to choose what you want. When you are logged in you should get an API key of length 40. Copy this for later
    use (HINT: if you forgot to copy the API key, you can find it under settings).

2. Next install wandb on your laptop

    ```bash
    pip install wandb
    ```

3. Now connect to your wandb account

    ```bash
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
    cases the logging is still going to use `wandb.log` but you need extra calls to `wandb.Image` etc. depending
    on what you choose to log.

7. Finally, lets create a report that you can share. Click the **Create report** button and include some of the
    graphs/plots/images that you have generated in the report.

8. To make sure that you have completed todays exercises, make the report shareable by clicking the *Share* button
    and create *view-only-link*. Send the link to my email `nsde@dtu.dk`, so I can checkout your awesome work 😃

9. When calling `wandb.init` you have two arguments called `project` and `entity`. Make sure that you understand these
    and try them out. It will come in handy for your group work as they essentially allows multiple users to upload their
    own runs to the same project in `wandb`.

10. Wandb also comes with build in feature for doing [hyperparameter sweeping](https://docs.wandb.ai/guides/sweeps)
    which can be beneficial to get a better working model. Look through the documentation on how to do a hyperparameter
    sweep in Wandb. You at least need to create a new file called `sweep.yaml` and make sure that you call `wandb.log`
    in your code on an appropriate value. Note: if you want `hydra` and `wandb` to work together you will need to change
    the `command` config in your `sweep.yaml` file, see this
    [page](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration).

11. In the future it will be important for us to be able to run Wandb inside a docker container (together with whatever
    training or inference we specify). The problem here is that we cannot authenticate Wandb in the same way as the
    previous exercise, it needs to happen automatically. Lets therefore look into how we can do that.

    1. First we need to generate an authentication key, or more precise an API key. This is in general the way any
        service (like a docker container) can authenticate. Start by going <https://wandb.ai/home>, click your profile
        icon in the upper right corner and then go to settings. Scroll down to the danger zone and generate a new API
        key and finally copy it.

    2. Next create a new docker file called `wandb.docker` and add the following code

        ```dockerfile
        FROM python:3.9
        RUN apt update && \
            apt install --no-install-recommends -y build-essential gcc && \
            apt clean && rm -rf /var/lib/apt/lists/*
        RUN pip install wandb
        COPY s4_debugging_and_logging/exercise_files/wandb_tester.py wandb_tester.py
        ENTRYPOINT ["python", "-u", "wandb_tester.py"]
        ```

        please take a look at the script being copied into the image and afterwards build the docker image.

    3. When we want to run the image, what we need to do is including a environment variables that contains the API key
        we generated. This will then authenticate the docker container with the wandb server:

        ```bash
        docker run -e WANDB_API_KEY=<your-api-key> wandb:latest
        ```

        Try running it an confirm that the results are uploaded to the wandb server.

12. Feel free to experiment more with `wandb` as it is a great tool for logging, organizing and sharing experiments.

That is the module on logging. Please note that at this point in the course you will begin to see some overlap between
the different frameworks. While we mainly used `hydra` for configuring our python scripts it can also be used to save
metrics and hyperparameters similar to how `wandb` can. Similar arguments holds for `dvc` which can also be used to
log metrics. In our opinion `wandb` just offers a better experience when interacting with the results after logging.
We want to stress that the combination of tools presented in this course may not be the best for all your future
projects, and we recommend finding a setup that fits you. That said, each framework provide specific features
that the others does not.

Finally, we want to note that we during the course really try to showcase a lot of open source frameworks, Wandb is not
one. It is free to use for personal usage (with a few restrictions) but for enterprise it does require a license. If you
are eager to only work with open-source tools we highly recommend trying out [MLFlow](https://mlflow.org/) which offers
the same overall functionalities as Wandb.
