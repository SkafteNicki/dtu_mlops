---
layout: default
title: M27 - Pre-commit
parent: S10 - Extra
nav_order: 1
---

# Pre-commit
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

One of the cornerstones of working with git is remembering to commit your work often. Often committing makes sure
that it is easier to identify and revert unwanted changes that you have introduced, because the code changes becomes
smalle per commit.

However, as you hopefully already seen in the course there are a lot of mental task to do before you actually write
`git commit` in the terminal. The most basic thing is of cause making sure that you have saved all your changes, and
you are not committing a not up-to-date file. However, this also includes tasks such as styling, formatting, making
sure all tests succeeds etc. All these mental to-do notes does not mix well with the principal of remembering to commit
often, because you in principal have to do them every time.

The obvious solution to this problem is to automate all or some of our mental task every time that we do a commit. This
is where *pre-commit hooks* comes into play, as they can help us attach additional tasks that should be run every time 
that we do a `git commit`.

## Configuration

Pre-commit simply works by inserting whatever workflow we want to automate in between whenever we do a `git commit` and afterwards would do a `git push`.

<p align="center">
  <img src="../figures/pre_commit.png" width="700" title="credit to https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/">
</p>

The system works by looking for a file called `.pre-commit-config.yaml` that we can configure. If we execute
```bash
pre-commit sample-config > .pre-commit-config.yaml
```
you should get a sample file that looks like
```yaml
# See https://pre-commit.com for more information
# See https://pre-commit.com/hooks.html for more hooks
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v3.2.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files
```
the file structure is very simple:
* It start by listing the reposatories where we want to get our pre-commits from, in this case <https://github.com/pre-commit/pre-commit-hooks> (this repo contains a large collection of pre-commit hooks)
* Next is defined what pre-commit hooks that we want to get by specifying the `id` of the different hooks. The `id` corresponds to an `id` in this file: <https://github.com/pre-commit/pre-commit-hooks/blob/master/.pre-commit-hooks.yaml>

When we are done defining our `.pre-commit-config.yaml` we just need to install it
```bash
pre-commit install
```
this will make sure that the file is automatically executed whenever we run `git commit`

### Exercises

1. Install pre-commit
   ```bash
   pip install pre-commit
   ```

2. Next create the sample file
   ```bash
   pre-commit sample-config > .pre-commit-config.yaml
   ```

3. The sample file already contains 4 hooks. Make sure you understand what each do and if you need them at all.

4. The base repo <https://github.com/pre-commit/pre-commit-hooks> also have a hook for configuring `flake8` to run. Add this to the config file and make sure it works as expected e.g. make something that is not `flake8` compliant and then try to commit that change.

5. Running `black` or `yapf` is not part of the base repo, however it is still possible to include this as pre-commit hooks. Google how to do it, include one of them and then test out that it actually works.

6. Finally, make sure that you also can do commits without running `pre-commit` e.g.
   ```bash
   git commit -m <message> --no-verify
   ```
  
That was all about how `pre-commit` can be used to automatize tasks. If you want to deep dive more into the topic you can checkout this [page](https://pre-commit.com/#python) on how to define your own `pre-commit` hooks.