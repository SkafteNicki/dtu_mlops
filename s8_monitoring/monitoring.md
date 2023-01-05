---
layout: default
title: M27 - System monitoring
parent: S8 - Monitoring
nav_order: 2
---

<img style="float: right;" src="../figures/icons/monitoring.png" width="130">

# Monitoring
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

## Exercises

We are in this exercise going to look at how we can setup automatic alerting
such that we get an message every time one of our applications are not behaving
as expected.

1. Go to the `Monitoring` service. Then go to `Alerting` tab.
   <p align="center">
     <img src="../figures/gcp_alert.png" width="800">
   </p>

2. Start by setting up an notification channel. A recommend setting up with an
   email.

3. Next lets create a policy. Clicking the `Add Condition` should bring up a
   window as below. You are free to setup the condition as you want but the
   image is one way bo setup an alert that will react to the number of times
   an cloud function is invoked (actually it measures the amount of log entries
   from cloud functions).
   <p align="center">
     <img src="../figures/gcp_alert_condition.png" width="800">
   </p>

4. After adding the condition, add the notification channel you created in one of
   the earlier steps. Remember to also add some documentation that should be send
   with the alert to better describe what the alert is actually doing.

5. When the alert is setup you need to trigger it. If you setup the condition as
   the image above you just need to invoke the cloud function many times. Here is
   a small code snippet that you can execute on your laptop to call a cloud function
   many time (you need to change the url and payload depending on your function):

   ```python
   import time
   import requests
   url = 'https://us-central1-dtumlops-335110.cloudfunctions.net/function-2'
   payload = {'message': 'Hello, General Kenobi'}

   for _ in range(1000):
      r = requests.get(url, params=payload)
   ```

6. Make sure that you get the alert through the notification channel you setup.
