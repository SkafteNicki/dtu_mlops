![Logo](../figures/icons/profiler.png){ align=right width="130"}

# Profilers

---

!!! info "Core Module"

## Profilers

In general profiling code is about improving the performance of your code. In this session we are going to take a
somewhat narrow approach to what "performance" is: runtime, meaning the time it takes to execute your program.

At the bare minimum, the two questions a proper profiling of your program should be able to answer is:

* *“ How many times is each method in my code called?”*
* *“ How long do each of these methods take?”*

The first question can help us prioritize what to optimize. If two methods `A` and `B` have approximately the same
runtime, but `A` is called 1000 more times than `B` we should probably spend time optimizing `A` over `B` if we want
to speed up our code. The second question directly tells us which methods are expensive to call.

Using profilers can help you find bottlenecks in your code. In this exercise we will look at two different
profilers, with the first one being the [cProfile](https://docs.python.org/3/library/profile.html). `cProfile` is
python's built-in profiler that can help give you an overview runtime of all the functions and methods involved in your
programs.

### ❔ Exercises

1. Run `cProfile` on the `vae_mnist_working.py` script. Hint: you can directly call the profiler on a
    script using the `-m` arg:

    === "Using pip"

        ```bash
        python -m cProfile -s <sort_order> myscript.py
        ```

    === "Using uv"

        ```bash
        uv run python -m cProfile -s <sort_order> myscript.py
        ```

    To write the output to a file you can use the `-o` argument:

    === "Using pip"

        ```bash
        python -m cProfile -s <sort_order> -o profile.txt myscript.py
        ```

    === "Using uv"

        ```bash
        uv run python -m cProfile -s <sort_order> -o profile.txt myscript.py
        ```

    ??? example "Script to debug"

        ```python linenums="1" title="vae_mnist_working.py"
        --8<-- "s4_debugging_and_logging/exercise_files/vae_mnist_working.py"
        ```

2. Try looking at the output of the profiling. Can you figure out which function took the longest to run? How do you
    show the content of the `profile.txt` file?

    ??? success "Solution"

        If you try to open `profile.txt` in a text editor you will see that it is not very human-readable. To get a
        better overview of the profiling you can use the `pstats` module to read the file and print the results in a
        more readable format. For example, to print the 10 functions that took the longest time to run you can use the
        following code:

        ```python
        import pstats
        p = pstats.Stats('profile.txt')
        p.sort_stats('cumulative').print_stats(10)
        ```

3. Can you explain the difference between `tottime` and `cumtime`? Under what circumstances do these differ and
    when are they equal?

    ??? success "Solution"

        `tottime` is the total time spent in the function excluding time spent in subfunctions. `cumtime` is the total
        time spent in the function including time spent in subfunctions. Therefore, `cumtime` is always greater than
        `tottime`.

4. To get a better feeling of the profiled result we can try to visualize it. Python does not
    provide a native solution, but open-source solutions such as [snakeviz](https://jiffyclub.github.io/snakeviz/)
    exist. Try installing `snakeviz` and load a profiled run into it (HINT: snakeviz expects the run to have the file
    format `.prof`).

5. Try optimizing the run! (Hint: The data is not stored as a torch tensor). After optimizing the code make sure
    (using `cProfile` and `snakeviz`) that the code actually runs faster.

    ??? success "Solution"

        For consistency reasons, even though the data in the `MNIST` dataset class from `torchvision` is stored as
        tensors, they are converted to
        [PIL images before being returned](https://github.com/pytorch/vision/blob/d3beb52a00e16c71e821e192bcc592d614a490c0/torchvision/datasets/mnist.py#L141-L143).
        This is the reason the solution is to initialize the dataclass with the transform

        ```python
        mnist_transform = transforms.Compose([transforms.ToTensor()])
        ```

        such that the data is returned as tensors. However, since the data is already stored as tensors, calling this
        transform every time you want to access the data is redundant and can be removed. The easiest way to do this is
        to create a `TensorDataset` from the internal data and labels (which are already tensors).

        ```python
        from torchvision.datasets import MNIST
        from torch.utils.data import TensorDataset
        # the class also internally normalize to [0,1] domain so we need to divide by 255
        train_dataset = MNIST(dataset_path, train=True, download=True)
        train_dataset = TensorDataset(train_dataset.data.float() / 255.0, train_dataset.targets)
        test_dataset = MNIST(dataset_path, train=False, download=True)
        test_dataset = TensorDataset(test_dataset.data.float() / 255.0, test_dataset.targets)
        ```

## PyTorch profiling

Profiling machine learning code can become much more complex because we are suddenly beginning to mix different
devices (CPU+GPU), which can (and should) overlap in some of their computations. When profiling this kind of machine
learning code we are often looking for *bottlenecks*. A bottleneck is simply a place in your code that is
preventing other processes from performing their best. This is the reason that all major deep learning
frameworks also include their own profilers that can help profiling more complex applications.

The image below show a typical report using the
[built-in profiler in pytorch](https://www.google.com/search?client=firefox-b-d&q=pytorch+profiling).
As the image shows, the profiler looks both at the `kernel` time (this is the time spent doing actual computations)
and also transfer times such as `memcpy` (where we are copying data between devices).
It can even analyze your code and give recommendations.

<figure markdown>
![Image](../figures/pytorch_profiler.png){ width="700" }
</figure>

Using the profiler can be as simple as wrapping the code that you want to profile with the `torch.profiler.profile`
decorator:

```python
with torch.profiler.profile(...) as prof:
    # code that I want to profile
    output = model(data)
```

### ❔ Exercises

<!-- markdownlint-disable -->
[Exercise files](https://github.com/SkafteNicki/dtu_mlops/tree/main/s4_debugging_and_logging/exercise_files){ .md-button }
<!-- markdownlint-restore -->

In these investigate the profiler that is build into PyTorch already. Note that these exercises require that you
have PyTorch v1.8.1 installed (or higher). You can always check which version you currently have installed by writing
(in a python interpreter):

```python
import torch
print(torch.__version__)
```

But we always recommend updating to the latest PyTorch version for the best experience. Additionally, to display the
result nicely (like `snakeviz` for `cProfile`) we are also going to use the tensorboard profiler extension.

=== "Using pip"

    ```bash
    pip install torch_tb_profiler
    ```

=== "Using uv"

    ```bash
    uv add torch_tb_profiler
    ```

1. A good starting point is to look at the [API for the profiler](https://pytorch.org/docs/stable/profiler.html). Here
    the important class to look at is the `torch.profiler.profile` class.

2. Let's try out a simple example (taken from
    [the PyTorch profiler recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)):

    1. Try to run the following code:

        ```python
        import torch
        import torchvision.models as models
        from torch.profiler import profile, ProfilerActivity

        model = models.resnet18()
        inputs = torch.randn(5, 3, 224, 224)

        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            model(inputs)
        ```

        This will profile the `forward` pass of a Resnet 18 model.

    2. Running this code will produce a `prof` object that contains all the relevant information about the profiling.
        Try writing the following code:

        ```python
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        ```

        What operation is using most of the cpu?

    3. Try running

        ```python
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))
        ```

        Can you see any correlation between the shape of the input and the cost of the operation?

    4. (Optional) If you have a GPU you can also profile the operations on that device:

        ```python
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            model(inputs)
        ```

    5. (Optional) As an alternative to using `profile` as a
        [context-manager](https://book.pythontips.com/en/latest/context_managers.html), we can also use its `.start` and
        `.stop` methods:

        ```python
        prof = profile(...)
        prof.start()
        ...  # code I want to profile
        prof.stop()
        ```

        Try doing this on the above example.

3. The `torch.profiler.profile` function takes some additional arguments. What argument would you need to
    set to also profile the memory usage? (Hint: this [page](https://pytorch.org/docs/stable/profiler.html))
    Try doing it on the simple example above and make sure to sort the samples by `self_cpu_memory_usage`.

4. As mentioned we can also get a graphical output for better inspection. After having done profiling
    try to export the results with:

    ```python
    prof.export_chrome_trace("trace.json")
    ```

    You should be able to visualize the file by going to `chrome://tracing` in any chromium-based web browser (provided
    you have enabled internal debugging pages at chrome://chrome-urls). Can you still identify the information printed
    in the previous exercises from the visualizations?

5. Running profiling on a single forward step can produce misleading results, as it only provides a single sample that
    may depend on what background processes are running on your computer. Therefore it is recommended to profile
    multiple iterations of your model. If this is the case then we need to include `prof.step()` to tell the profiler
    when we are doing a new iteration.

    ```python
    with profile(...) as prof:
        for i in range(10):
            model(inputs)
            prof.step()
    ```

    Try doing this. Is the conclusion the same on what operations are taking up most of the time? Have the
    percentages changed significantly?

6. Additionally, we can also visualize the profiling results using the profiling viewer in tensorboard.

    1. Start by initializing the `profile` class with an additional argument:

        ```python
        from torch.profiler import profile, tensorboard_trace_handler
        with profile(..., on_trace_ready=tensorboard_trace_handler("./log/resnet18")) as prof:
            ...
        ```

        Try running profiling (using a couple of iterations) and make sure that a file with the `.pt.trace.json` is
        produced in the `log/resnet18` folder.

    2. Now try launching tensorboard

        ```bash
        tensorboard --logdir=./log
        ```

        and open the page <http://localhost:6006/#pytorch_profiler>, where you should hopefully see an image similar
        to the one below:

        <figure markdown>
        ![Image](../figures/profiler_overview.png){ width="600" }
        <figcaption>
        <a href="https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html"> Image credit </a>
        </figcaption>
        </figure>

        Try poking around in the interface.

    3. Tensorboard has a nice feature for comparing runs under the `diff` tab. Try redoing a profiling run but use
        `model = models.resnet34()` instead. Load up both runs and try to look at the `diff` between them.

7. As a final exercise, try to use the profiler on the `vae_mnist_working.py` file from the previous module on
    debugging, where you profile a whole training run (not only the forward pass). What is the bottleneck during the
    training? Is it still the forward pass or is it something else? Can you improve the code somehow based on the
    information from the profiler?

This ends the module on profiling. If you want to go into more details on this topic we recommend looking into
[line_profiler and kernprof](https://github.com/pyutils/line_profiler). A downside of using python's `cProfile` is that
it can only profile at a functional/modular level, which is great for identifying hotspots in your code. However,
sometimes the cause of a computational hotspot is a single line of code in a function, which will not be caught by
`cProfile`. An example would be a simple index operation such as `a[idx] = b`, which for large arrays and
non-sequential indexes is really expensive. For these cases
[line_profiler and kernprof](https://github.com/pyutils/line_profiler) are excellent tools to have in your toolbox.
Additionally, if you do not like cProfile we also recommend [py-spy](https://github.com/benfred/py-spy), which is
another open-source profiling tool for Python programs.
