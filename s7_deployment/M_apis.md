---
layout: default
title: Creating APIs
parent: S10 - Extra
nav_order: 5
---

# Creating APIs
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

Before we can get deployment of our models we need to understand concepts such as APIs and requests. It is essentially
that we do this for one reason: the world does not run on python.

The applications that we are developing in this course are focused on calling python functions directly. However, if we
were to produce an product that some users should use we cannot expect them to want to work with our applications on a
code level. Additionally, we cannot expect that they want to work with python related data types. We should therefore
develop application programming interface (API) such that users can easily interact with to use our applications,
meaning that it should have the correct level of abstraction that users can use our application as they seem fit,
without ever having to look at the code.


We highly recommend that you go over this [comic strip](https://howhttps.works/) if you want to learn more about HTTPS
protokol, but the TLDR is that it provides privacy, integrity and identification over the web.


We will be designing our API around an client-server type of of architechture: the client (user) is going to send
*requests* to a server (our application) and the server will give an *response*. For example the user may send an
request of getting classifying a specific image, which our application will do and then send back the response in
terms of a label.

## Requests



### Exercises

1. Start by install the `requests` package

   ```bash
   pip install requests
   ```

2. Afterwards, create a small script and try to execute the code

   ```python
   import requests
   response = requests.get("https://api.open-notify.org/this-api-doesnt-exist")
   print(response.status_code)
   ```

   This should return status code 404. Take a look at [this page](https://restfulapi.net/http-status-codes/) that
   contains a list of status codes.

3. Next lets call a page that actually exists

   ```python
   import requests
   response = requests.get("https://api.open-notify.org/astros.json")
   print(response.status_code)
   ```

   What is the status code now and what does it mean?

4. Next look at `response.json()`? What does this return? Does it make sense compared to if you open the webpage in your
   browser.

5. Importantly about last exercise is the `json` format. [JSON](https://www.json.org/json-en.html) (JavaScript Object
   Notation) is the language of requests, meaning that whenever we want to make a request where we send data with it
   it needs to be encoded as `json` and whenever we get a response back we also receive a `json` stucture.




## FastAPI

For these exercises we are going to use [FastAPI](https://fastapi.tiangolo.com/) for creating our API. FastAPI is a
*modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints*.
FastAPI is only one of many frameworks for defining APIs, however compared to other frameworks such as
[Flask](https://flask.palletsprojects.com/en/2.0.x/) and [django](https://www.djangoproject.com/) it offers a sweet
spot of being flexible enough to do what you want without having many additional (unnecessary) features.

### Exercises

1. Install FastAPI

   ```bash
   pip install fastapi
   ```

2. Additionally, also install uvicorn which is a package for defining low level server applications.

   ```bash
   pip install uvicorn[standard]
   ```

3. Start by defining a small application like this in a file called `main.py`

   ```python
   from fastapi import FastAPI
   app = FastAPI()
   @app.get('/')
   def root():
       """ root """
       return 'ok'

   @app.get('/item/{id}')
   def read_item(id: int):
       """ return an input item """
       return id
   ```

   explain what the idea behind the two functions are.

4. Next lets lunch our app. In a terminal write

   ```bash
   uvicorn main:app --reload
   ```

   this will launch an server at this page: `http://localhost:8000/`. As you will hopefully see, this
   page will return the content of the `root` function.

   1. With this in mind, what side should you open to get the server to return `1`.

   2. Also checkout the pages: `http://localhost:8000/docs` and `http://localhost:8000/redoc`. What does
      these pages show?

5. With the fundamentals in place lets configure it a bit more:

   1. Lets start by changing the root function to include a bit more info:

      ```python
      from http import HTTPStatus

      @app.get("/")
      def root():
          """ Health check."""
          response = {
              "message": HTTPStatus.OK.phrase,
              "status-code": HTTPStatus.OK,
              "data": {},
          }
          return response
      ```

      try to reload the app and see what is returned now. You should not have to re-launch the app because we
      initialized the app with the `--reload` argument.

   2. something something

6. Next lets look at how we can configure a machine learning application. The core idea behind this set of exercises
   is to get comftable with sending more complex data such as text and images over HTTP requests

   1. We have in the exercise folder provided a very simple machine learning application that can take in multiple
      kinds of input and does different things with it



7. Finally, we want to figure out how do include our FastAPI application in a docker container as it will help us when
   we want to deploy in the cloud because docker as always can take care of the dependencies for our application. For
   the following you can take whatever privious FastAPI application as the base application for the container

   1. Start by creating a `requirement.txt` file for you application. You will atleast need `fastapi` and `unicorn` in
      the file and we as always recommend that you are specific about the version you want to use:

      ```txt
      fastapi>=0.68.0,<0.69.0
      uvicorn>=0.15.0,<0.16.0
      # add anything else you application needs to be able to run
      ```

   3. Next create a `Dockerfile` with the following content

      ```Dockerfile
      FROM python:3.9
      WORKDIR /code
      COPY ./requirements.txt /code/requirements.txt

      RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
      COPY ./app /code/app

      CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
      ```

      The above assumes that your file structure looks like this

      ```txt
      .
      ├── app
      │   ├── __init__.py
      │   └── main.py
      ├── Dockerfile
      └── requirements.txt
      ```

      Hopefully all these step should look familiar if you already went through
      [module M9](../s3_reproducibility/M9_docker.md), except for maybe the last line. However, this is just the
      standard way that we have run our FastAPI applications as the last couple of exercises, this time with some extra
      arguments regarding the ports we allow.

   4. Next build the corresponding docker image

      ```bash
      docker build -t my_fastapi_app .
      ```

   5. Finally, run the image such that a container is spinned up that runs our application. The important part here is
      to remember to specify the `-p` argument (p for port) that should be the same number as the port we have specified
      in the last line of our Dockerfile.

      ```bash
      docker run -d --name mycontainer -p 80:80 myimage
      ```

   6. Check that everything is working by going to the corresponding localhost page <http://localhost/items/5?q=somequery>
      ```

This ends the module on APIs. If you want to go further in this direction we highly recommend that you checkout
[bentoml](https://github.com/bentoml/BentoML) that is an API standard that focuses solely on creating easy to understand
APIs and services for machine learning applications.

<!---
## BentoML

[bentoml](https://github.com/bentoml/BentoML)

In particular

```python
bentoml.pytorch_lightning.save_model
bentoml.pytorch.save
bentoml.onnx.save_model
```

```python
runner = bentoml.pytorch.get("my_torch_model").to_runner()
svc = bentoml.Service(name="test_service", runners=[runner])
@svc.api(input=JSON(), output=JSON())
async def predict(json_obj: JSONSerializable) -> JSONSerializable:
    batch_ret = await runner.async_run([json_obj])
    return batch_ret[0]
```
--->
