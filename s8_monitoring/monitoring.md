
![Logo](../figures/icons/signoz.png){ align=right width="130"}
![Logo](../figures/icons/opentelemetry.png){ align=right width="130"}
![Logo](../figures/icons/monitoring.png){ align=right width="130"}

# Monitoring

---

In this module we are going to look into more classical monitoring of applications. The key concept we are often working
with here is called *telemetry*. Telemetry in general refer to any automatic measurement and wireless transmission of
data from our application. It could be numbers such as:

* The number of requests are our application receiving per minute/hour/day. This number is of interest because it is
  directly proportional to the running cost of application.
* The amount of time (on average) our application runs per request. The number is of interest because it most likely is
  the core contributor to the latency that our users are experience (which we want to be low).
* ...

## ❔ Exercises

The exercise here is simply to follow the instructions in this repository:

<https://github.com/duarteocarmo/dtu-mlops-monitoring>

by [Duarte Carmo](https://duarteocarmo.com/). The repository goes over the steps to setup an simple application that
uses FastAPI to implement a application and [opentelemetry](https://opentelemetry.io/) to extract relevant telemetry
data that is then visualized through [Signoz](https://signoz.io/). Importantly you should try to take a look at the
implemented [application](https://github.com/duarteocarmo/dtu-mlops-monitoring/blob/master/src/api/main.py) to see how
opentelemetry is integrated into a FastAPI application.

If you manage to get the done with the steps in the repository we recommend that you can try to deploy this to GCP
by following [these steps](https://signoz.io/docs/install/kubernetes/gcp/). However, we do not really recommend it as
it requires setting up a Kubernetes cluster (because we are running multiple docker containers) which is not part of
the curriculum in this course

## Alert systems

A core problem within monitoring is alert systems. The alert system is in charge of sending out alerts to relevant
people when some telemetry or metric we are tracking is not behaving as it should. Alert systems are a subjective
choice of when and how many should be send out and in general should be proportional with how important to the of the
metric/telemetry. We commonly run into what is referred to the
[goldielock problem](https://en.wikipedia.org/wiki/Goldilocks_principle) where we want just the *right amount* of alerts
however it is more often the case that we either have

* Too many alerts, such that they become irrelevant and the really important onces are overseen, often referred to as
  alert fatigue
* Or alternatively, we have too little alerts and problems that should have triggered an alert is not dealt with when
  they happen which can have unforeseen consequences.

Therefore, setting up proper alert systems can be as challenging as setting up the systems for actually the metrics we
want to trigger alerts.

### ❔ Exercises

We are in this exercise going to look at how we can setup automatic alerting such that we get an message every time one
of our applications are not behaving as expected.

1. Go to the `Monitoring` service. Then go to `Alerting` tab.
    <figure markdown>
    ![Image](../figures/gcp_alert.png){ width="800" }
    </figure>

2. Start by setting up an notification channel. A recommend setting up with an email.

3. Next lets create a policy. Clicking the `Add Condition` should bring up a window as below. You are free to setup the
   condition as you want but the image is one way bo setup an alert that will react to the number of times an cloud
   function is invoked (actually it measures the amount of log entries from cloud functions).

    <figure markdown>
    ![Image](../figures/gcp_alert_condition.png){ width="800" }
    </figure>

4. After adding the condition, add the notification channel you created in one of the earlier steps. Remember to also
    add some documentation that should be send with the alert to better describe what the alert is actually doing.

5. When the alert is setup you need to trigger it. If you setup the condition as the image above you just need to
    invoke the cloud function many times. Here is a small code snippet that you can execute on your laptop to call a
    cloud function many time (you need to change the url and payload depending on your function):

    ```python
    import time
    import requests
    url = 'https://us-central1-dtumlops-335110.cloudfunctions.net/function-2'
    payload = {'message': 'Hello, General Kenobi'}

    for _ in range(1000):
        r = requests.get(url, params=payload)
    ```

6. Make sure that you get the alert through the notification channel you setup.
