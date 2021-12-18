---
layout: default
title: M29 - Creating APIs
parent: S10 - Extra
nav_order: 3
---

# Creating APIs (Under construction)
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

The applications that we are developing in this course are focused on calling python functions directly. However, if we were to produce an product that some users should use we cannot expect them to want to work with our applications on a code level. Additionally, we cannot expect that they want to work with python related data types. We should therefore develop application programming interface (API) such that users can easily interact with to use our applications, meaning that it should have the correct level of abstraction that users can use our application as they seem fit, without ever having to look at the code.

We will be designing our API around an client-server type of of architechture: the client (user) is going to send *requests* to a server (our application) and the server will give an *response*. For example the user may send an request of getting classifing a specific image, which our application will do and then send back the response in terms of a label.

## FastAPI

For these exercises we are going to use [FastAPI](https://fastapi.tiangolo.com/) for creating our API. FastAPI is a *modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints*. FastAPI is only one of many frameworks for defining APIs, however compared to other frameworks such as [Flask](https://flask.palletsprojects.com/en/2.0.x/) and [django](https://www.djangoproject.com/) it offers a sweet spot of being flexible enough to do what you want without having many additional (unnecessary) features. 

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

  



