![Logo](../figures/icons/debugger.png){ align=right width="130"}

# Debugging

---

Debugging is very hard to teach and is one of the skills that just comes with experience. That said, there are good
and bad ways to debug a program. We are all probably familiar with just inserting `print(...)` statements everywhere
in our code. It is easy and can many times help narrow down where the problem happens. That said, this is not a great
way of debugging when dealing with a very large codebase. You should therefore familiarize yourself with the built-in
[python debugger](https://docs.python.org/3/library/pdb.html) as it may come in handy during the course.

<figure markdown>
  ![Image](../figures/debug.jpg){width="700" }
</figure>

To invoke the build in Python debugger you can either:

* Set a trace directly with the Python debugger by calling

    ```python
    import pdb
    pdb.set_trace()
    ```

    anywhere you want to stop the code. Then you can use different commands (see the `python_debugger_cheatsheet.pdf`)
    to step through the code.

* If you are using an editor, then you can insert inline breakpoints (in VS code this can be done by pressing `F9`)
    and then execute the script in debug mode (inline breakpoints can often be seen as small red dots to the left of
    your code). The editor should then offer some interface to allow you step through your code. Here is a guide to
    using the build in debugger [in VScode](https://code.visualstudio.com/docs/python/debugging#_basic-debugging).

* Additionally, if your program is stopping on an error and you automatically want to start the debugger where it
    happens, then you can simply launch the program like this from the terminal

    ```bash
    python -m pdb -c continue my_script.py
    ```

## ❔ Exercises

<!-- markdownlint-disable -->
[Exercise files](https://github.com/SkafteNicki/dtu_mlops/tree/main/s4_debugging_and_logging/exercise_files){ .md-button }
<!-- markdownlint-restore -->

We here provide a script `vae_mnist_bugs.py` which contains a number of bugs to get it running. Start by going over
the script and try to understand what is going on. Hereafter, try to get it running by solving the bugs. The following
bugs exist in the script:

* One device bug (will only show if running on gpu, but try to find it anyways)
* One shape bug
* One math bug
* One training bug

Some of the bugs prevents the script from even running, while some of them influences the training dynamics. Try to
find them all. We also provide a working version called `vae_mnist_working.py` (but please try to find the bugs before
looking at the script). Successfully debugging and running the script should produce three files:

* `orig_data.png` containing images from the standard MNIST training set
* `reconstructions.png` reconstructions from the model
* `generated_samples.png` samples from the model

Again, we cannot stress enough that the exercise is actually not about finding the bugs but **using a proper** debugger
to find them.

??? success "Solution for device bug"

    If you look at the reparametrization function in the `Encoder` class you can see that we initialize a noise tensor

    ```python
    def reparameterization(self, mean, var):
        """Reparameterization trick to sample z values."""
        epsilon = torch.randn(*var.shape)
        return mean + var * epsilon
    ```

    this will fail with a
    `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!` if
    you are running on GPU, because the noise tensor is initialized on the CPU. You can fix this by initializing the
    noise tensor on the same device as the mean and var tensors

    ```python
    def reparameterization(self, mean, var):
        """Reparameterization trick to sample z values."""
        epsilon = torch.randn(*var.shape, device=mean.device)
        return mean + var * epsilon
    ```

??? success "Solution for shape bug"

    In the `Decoder` class we initialize the following fully connected layers

    ```python
    self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
    self.FC_output = nn.Linear(latent_dim, output_dim)
    ```

    which is used in the forward pass as

    ```python
    def forward(self, x):
        """Forward pass of the decoder module."""
        h = torch.relu(self.FC_hidden(x))
        return torch.sigmoid(self.FC_output(h))
    ```

    this means that `h` should be a tensor of shape `[bs, hidden_dim]` but since we initialize the `FC_output`
    layer with `latent_dim` output dimensions, the forward pass will fail with a
    `RuntimeError: size mismatch, m1: [bs, hidden_dim], m2: [bs, latent_dim]` if `hidden_dim != latent_dim`. You can
    fix this by initializing the `FC_output` layer with `hidden_dim` output dimensions

    ```python
    self.FC_output = nn.Linear(hidden_dim, output_dim)
    ```

??? success "Solution for math bug"

    In the Encoder class you have the following code

    ```python
    def forward(self, x):
        """Forward pass of the encoder module."""
        h_ = torch.relu(self.FC_input(x))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)
        z = self.reparameterization(mean, log_var)
        return z, mean, log_var

    def reparameterization(self, mean, var):
        """Reparameterization trick to sample z values."""
        epsilon = torch.randn(*var.shape)
        return mean + var * epsilon
    ```

    from just the naming of the variables you can see that `log_var` is the log of the variance and not the variance
    itself. This means that you should exponentiate `log_var` before using it in the `reparameterization` function

    ```python
    z = self.reparameterization(mean, torch.exp(log_var))
    ```

    alternatively, we can convert to using the standard deviation instead of the variance

    ```python
    z = self.reparameterization(mean, torch.exp(0.5 * log_var))
    ```

    and

    ```python
    epsilon = torch.randn_like(std)
    ```

??? success "Solution for training bug"

    Any training loop in PyTorch should have the following structure

    ```python
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
    ```

    if you look at the code for the training loop in the `vae_mnist_bugs.py` script you can see that the optimizer is
    not zeroed before the backward pass. This means that the gradients will accumulate over the batches and will
    explode. You can fix this by adding the line

    ```python
    optimizer.zero_grad()
    ```

    as the first line of the inner training loop
