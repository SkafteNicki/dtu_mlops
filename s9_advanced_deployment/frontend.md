![Logo](../figures/icons/kubernetes.png){ align=right width="130"}

# Frontend

---

!!! danger
    Module is still under development

If you have gone over the [deployment module](../s7_deployment/README.md) you should be at the point where you have a machine learning model running in the cloud. The model can be interacted with by sending HTTP requests to the API endpoint. In general we refer to this as the *backend* of the application. It is the part of our application that are behind-the-scences that the user does not see and it is not really that user-friendly. Instead we want to create a *frontend* that the user can interact with in a more user-friendly way. This is what we will be doing in this module.

Another point of splitting our application into a frontend and a backend has to do with scalability. If we have a lot of users interacting with our application, we might want to scale only the backend and not the frontend, because that is the part that will be running our heavy machine learning model. In general dividing a application into smaller pieces are the pattern that is used in microservice architectures.

<figure markdown>
![](../figures/different_architechtures.png){ width="800" }
<figcaption>
In monollithic applications everything the user may be requesting of our application is handled by a single process/container. In microservice architectures the application is split into smaller pieces that can be scaled independently. This also leads to easier maintainability and faster development. <a href="https://www.researchgate.net/figure/Comparison-between-monolithic-and-microservices-architectures-for-an-application-that_fig5_355361563">Image Credit.</a>
</figcaption>
</figure>

Frontends have for the longest time been created using HTML, CSS and JavaScript. This is still the case, but there are now a lot of frameworks that can help us create a frontend in Python:

* [Django](https://www.djangoproject.com/)
* [Reflex](https://reflex.dev/)
* [Streamlit](https://streamlit.io/)
* [Bokeh](http://bokeh.org/)
* [Gradio](https://www.gradio.app/)

<figure >





In this module we will be looking at two of these frameworks: `streamlit` and `dash`. Both of these frameworks are built on top of Python and are therefore easy to integrate with our machine learning models.



## Streamlit

`steamlit`

### ❔ Exercises

1. Start by installing `streamlit`

   ```bash
   pip install streamlit
   ```

   and run `streamlit hallo` afterwards to check that everything works as expected.
