# Monitoring

[Slides](../slides/Monitoring.pdf){ .md-button }

<div class="grid cards" markdown>

- ![](../figures/icons/evidentlyai.png){align=right : style="height:100px;width:100px"}

    Learn how to detect data drifting using the `evidently` framework

    [:octicons-arrow-right-24: M26: Data Drifting](data_drifting.md)

- ![](../figures/icons/prometheus.png){align=right : style="height:100px;width:100px"}

    Learn how to setup a prometheus monitoring system for your application

    [:octicons-arrow-right-24: M27: System Monitoring](monitoring.md)

</div>

We have now reached the end of our machine-learning pipeline. We have successfully developed, trained and deployed a
machine learning model. However, the question then becomes if you can trust that your newly deployed model still works
as expected after 1 day without you intervening? What about 1 month? What about 1 year?

There may be corner cases where an ML model is working as expected, but the vast majority of ML models will perform
worse over time because they are not generalizing well enough. For example, assume you have just deployed an application
that classifies images from phones when suddenly a new phone comes out with a new kind of sensor that takes images that
either have a very weird aspect ratio or something else your model is not robust towards. There is nothing wrong with
this, you can essentially just retrain your model on new data that accounts for this corner case, however, you need a
mechanism that informs you.

This is very monitoring comes into play. Monitoring practices are in charge of collecting any information about your
application in some format that can then be analyzed and reacted on. Monitoring is essential to securing the longevity
of your applications.

As with many other sub-fields within MLOps, we can divide monitoring into classic monitoring and ML-specific monitoring.
Classic monitoring (known from classic DevOps) is often about

- Errors: Is my application working without problems?
- Logs: What is going on?
- Performance: How fast is my application?

All these are basic information you are interested in regardless of what application type you are trying to deploy.
However, then there is machine learning related monitoring that especially relates data. Take the example above, with
the new phone, this we would in general consider to be data drifting problem e.g. the data you are trying to do
inference on have drifted away from the distribution of data your model was trained on. Such monitoring problems are
unique to machine learning applications and needs to be handled separately.

We are in this session going to see examples of both kinds of monitoring.

!!! tip "Learning objectives"

    The learning objectives of this session are:

    * Understand the concepts of data drifting in machine learning applications
    * Can detect data drifting using the `evidently` framework
    * Understand the importance of different system level monitoring and can conceptually implement it
