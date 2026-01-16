# Before starting

When working on the exercises, I in general recommend following this structure:

```txt
<02476_XXX>/                      # call this whatever you like
    ├── dtu_mlops/                # the content of this repository
    │   ├── .venv/
    │   ├── uv.lock
    │   ├── pyproject.toml
    │   └── ...
    ├── <template-project>/       # this is the cookiecutter template project from day 2
    │   ├── .venv/
    │   ├── uv.lock
    │   ├── pyproject.toml
    │   └── ...
    ├── exercises/                # folder for all other exercises
    │   ├── exercise_s<X1>_m<Y1>/   # exercise for session X1, module Y1
    │   │   └── ...
    │   ├── exercise_s<X2>_m<Y2>/   # exercise for session X2, module Y2
    │   │   └── ...
    │   ├── .venv/
    │   ├── uv.lock
    │   ├── pyproject.toml
    │   └── ...
    | ── <exam-project>/          # your exam project will be created on day5
    │   ├── .venv/
    │   ├── uv.lock
    │   ├── pyproject.toml
    │   └── ...
    └── ...                       # any other notes/files etc. related to the course
```

* The `dtu_mlops/` folder contains the content of this repository. You should clone it on day 1 of the course:

    ```bash
    git clone https://github.com/SkafteNicki/dtu_mlops.git
    ```

    and consider running `git pull` from time to time to get the latest updates.

* The `<template-project>/` folder contains the cookiecutter template project that you will create on day 2 of the
    course. This will be a running example throughout the course when working on the exercises.

* The `exercises/` folder contains one-off exercises that you will work on during the course. Each exercise should
    ideally be in its own folder, named according to the session and module it belongs to.

* The `<exam-project>/` folder will contain your exam project that you will create on day 5 of the course.

Here the `<>` indicates that you can name the folders as you like. Please do not use spaces in the folder names, as this
can sometimes lead to issues when working with command line tools. The core point about doing so is keep your work
separated into different virtual environments. This way, you can avoid dependency conflicts between the different parts
of this course.

If you need example code, exercise files or solutions from the course repository (the `dtu_mlops/` folder) you can
simply copy them over into the relevant project folder.
