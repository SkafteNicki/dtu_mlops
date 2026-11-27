![Logo](../figures/icons/prometheus.png){ align=right width="130"}
![Logo](../figures/icons/monitoring.png){ align=right width="130"}

# System monitoring

---

!!! info "Core Module"

A model that passed every test on your laptop can still fail in production: the container runs out of memory, a client
starts sending broken requests, answers get slow, or the images people send stop looking like the ones the model was
trained on. Monitoring is how you find out before your users tell you. We look at a deployed model at four
**monitoring layers**:

| Layer | The question it answers | Example signal |
| --- | --- | --- |
| **System** | Is the machine and the process healthy? | Is the target up, memory, CPU |
| **Service** | Are requests answered, how fast, and how often do they fail? | Request rate, p95 latency, error rate |
| **Data** | Does the input still look like what the model was trained on? | Drift of image features against the **reference data** |
| **Model** | Are the predictions still good? | Share of each predicted class, mean confidence, accuracy once labels arrive |

This session has three modules. In this one, **System monitoring**, you cover the system and service layers: you
instrument an image classifier API with Prometheus metrics, look at it in Grafana, move the whole thing to the cloud and
make an alert reach your phone. In [Data drifting](data_drifting.md) you store a **production window** and fill in the
data and model layers. In Closing the loop, newly labelled data flows back into training and a better model gets
deployed.

```txt
traffic.py ──▶ /predict  (FastAPI API, on Cloud Run) ◀── scrapes /metrics ── Prometheus ◀── queries ── Grafana ──▶ Discord
                                                                             └────── always-on VM ──────┘    (alert)
```

The module takes about 2.5 hours. Steps that take a while say so, so you can plan around them.

!!! tip "Learning objectives"

    The learning objectives of this module are:

    * Know the four monitoring layers and which signals belong to each
    * Be able to instrument a Python API with Prometheus counters and histograms, and read the result with PromQL
    * Be able to run Prometheus and Grafana with docker compose, with the Grafana data source and dashboard provisioned
        from files
    * Understand why scraping a serverless service only works while it runs one instance, and what production systems
        do instead
    * Be able to run a monitoring stack on a VM you sized yourself, with no open ports, and send an alert to a channel

## The API we monitor

Everything in this module happens in one self-contained folder:
[`s8_monitoring/exercise_files/monitoring_stack`](https://github.com/SkafteNicki/dtu_mlops/tree/main/s8_monitoring/exercise_files/monitoring_stack).
It holds a small FastAPI app, much like the one you built in [M22 on APIs](../s7_deployment/apis.md), that classifies
images of hands showing rock, paper or scissors (RPS). You do not need your own project's API to be finished: you
monitor this one first, and apply the same ideas to your project afterwards. The files are:

| File | What it is |
| --- | --- |
| `dataset.py` | Everything specific to the dataset: the class names, the image size and the preprocessing. Nothing else hard-codes the dataset. |
| `model.py` | A small CNN and how its checkpoint is saved and loaded |
| `train.py` | Trains the CNN and saves `models/model.pt` |
| `api.py` | The API, with gaps for you to fill. `api_solution.py` is the filled-in version. |
| `traffic.py` | Replays a folder of images through the API, with a few malformed requests mixed in |
| `api.dockerfile`, `compose.yaml` | The API as a container, and the local stack: API, Prometheus and Grafana |
| `prometheus.yml`, `prometheus.cloud.yml`, `grafana/` | Configuration for Prometheus and Grafana |
| `test_api.py` | Tests for the solution API |

### ❔ Exercises

1. Copy the folder somewhere you can work in it, and set up an environment for it. If you work inside a clone of the
    course repository, `uv sync --group exercises` already gives you everything. Otherwise:

    ```bash
    cp -r dtu_mlops/s8_monitoring/exercise_files/monitoring_stack ~/monitoring_stack
    cd ~/monitoring_stack
    uv init --bare
    uv add -r requirements.txt torch requests
    uv add --dev pytest httpx
    ```

2. Download the dataset and train the model. The download is about 230 MB and takes a minute or two, training takes
    about a minute on a laptop CPU.

    ```bash
    uv run python dataset.py   # data/rps (2520 training images) and data/rps-test-set (372 test images)
    uv run python train.py     # models/model.pt
    ```

    The dataset is [Rock-Paper-Scissors](https://laurencemoroney.com/datasets.html) by Laurence Moroney (CC BY 2.0):
    computer-generated hands on a white background. The model reaches about 80% on the test set. That is good enough
    for this module: we monitor the model, we don't improve it.

3. Look through `dataset.py` and `model.py`. **Where would you change the code if the course switched to a different
    image dataset? What does the API answer with, and where does the model version come from?**

    ??? success "Solution"

        Only `dataset.py`: `CLASSES`, `IMAGE_SIZE`, `CHANNELS`, the download URLs and `preprocess` are defined there,
        and the model, the API and the scripts import them. The API answers with the predicted class, its confidence
        (the softmax probability of that class) and the model version. The version is the start of the checkpoint
        file's SHA-256 hash (`sha-1a2b3c4d`), so two different models can never report the same version. In the
        Closing the loop module the model comes from the W&B model registry instead, and the version with it.

## Instrumenting the API with Prometheus

[Prometheus](https://prometheus.io/) is the standard open-source system for metrics. It works by **pulling**: your
application keeps its metrics in memory and exposes their current values on an HTTP endpoint, usually `/metrics`, in a
simple [text format](https://prometheus.io/docs/instrumenting/exposition_formats/). Every few seconds the Prometheus
server fetches ("scrapes") that endpoint and stores the values as time series. The application never sends anything
anywhere; it only counts.

### ❔ Exercises

1. Read about the [metric types](https://prometheus.io/docs/concepts/metric_types/). **What is the difference between
    a counter, a gauge, a histogram and a summary? Which type would you use for the number of answered requests, for
    the time each request takes, and for the current number of images in a queue?**

    ??? success "Solution"

        * A **counter** only goes up (or resets to zero when the process restarts). Use it for things that happen: the
            number of answered requests, the number of errors. You almost never look at a counter's raw value, but at
            how fast it grows: `rate(...)`.
        * A **gauge** goes up and down. Use it for things that *are*: memory in use, the number of images in a queue,
            the temperature.
        * A **histogram** counts observations in buckets (for example "answered in under 10 ms") plus their sum and
            count. Use it for durations and sizes. Prometheus can compute quantiles such as p95 from the buckets, and
            the buckets of many instances can be added up.
        * A **summary** computes quantiles inside the application. They cannot be added up across instances, so
            histograms are usually the better choice.

        The number of requests is a counter, the time per request a histogram, the queue length a gauge.

2. Open `api.py` and fill in the three `TODO`s with the
    [`prometheus_client`](https://prometheus.github.io/client_python/instrumenting/) library:

    1. a **counter** `predict_requests_total` of answered requests, labelled with the predicted class;
    2. a **histogram** `predict_latency_seconds` of how long each answer takes;
    3. a **counter** `predict_errors_total` of rejected requests, labelled with the reason.

    Prometheus measures time in seconds, so the histogram is in seconds too. Our answers take milliseconds, so the
    default buckets (5 ms up to 10 s) would put almost everything in the first few. Use the millisecond-scale buckets
    given in the `TODO`.

    ??? example "The API with gaps: `api.py`"

        ```python linenums="1" title="api.py"
        --8<-- "s8_monitoring/exercise_files/monitoring_stack/api.py"
        ```

    ??? success "Solution"

        The lines that define and update the metrics are highlighted.

        ```python linenums="1" title="api_solution.py" hl_lines="23-33 64 71-72"
        --8<-- "s8_monitoring/exercise_files/monitoring_stack/api_solution.py"
        ```

        If you are stuck, `cp api_solution.py api.py` and carry on: the rest of the module works with either.

3. Start the API and send it an image:

    ```bash
    uv run uvicorn api:app --port 8000
    ```

    Open <http://localhost:8000/docs>, choose `POST /predict` → *Try it out*, upload an image from
    `data/rps-test-set/` and *Execute*. Then open <http://localhost:8000/metrics>. **Which of your three metrics have a
    value already? Why does one of them show only its `# HELP` and `# TYPE` lines?**

    ??? success "Solution"

        `predict_requests_total` has one sample, for the class that was predicted, and `predict_latency_seconds` has
        its buckets, `_count` and `_sum`. `predict_errors_total` shows only `# HELP` and `# TYPE`: it has a label,
        `reason`, and a labelled metric has one time series per label value. Nothing has been rejected yet, so no
        label value exists and there is no series to show. The series for `reason="not_an_image"` appears the first
        time that reason is counted. A metric without labels would show `0` right away.

        This matters later: until the first error, a query such as `sum(rate(predict_errors_total[1m]))` returns *no
        data*, not zero. Dashboards show an empty panel, and alerts have to be told what "no data" means.

4. **Why are there many more metrics on `/metrics` than the three you added? Which monitoring layer do they belong to?**

    ??? success "Solution"

        `prometheus_client` registers some collectors in its default registry by itself: `process_*` (resident memory,
        CPU seconds, open file descriptors, start time) and `python_*` (garbage collector, interpreter version). They
        describe the process, not the requests, so they belong to the **system** layer. The three metrics you added
        are the **service** layer. (If you only want your own metrics, you can create your own `CollectorRegistry`
        and expose that instead.)

5. Replay the RPS test set through the API, in a second terminal:

    ```bash
    uv run python traffic.py data/rps-test-set
    ```

    `traffic.py` sends the images in shuffled order under a neutral file name, so the API can't read the class from
    the file name, and mixes in a malformed request (not an image, a cut-off image, an empty file) about once every
    50 images, as real clients do. **How many requests were rejected, and for which reasons? Without the error
    counter, where would you have noticed them?**

    ??? example "The traffic script: `traffic.py`"

        ```python linenums="1" title="traffic.py"
        --8<-- "s8_monitoring/exercise_files/monitoring_stack/traffic.py"
        ```

    ??? success "Solution"

        About 7 of 379 requests, split over `not_an_image` (text and cut-off images) and `empty`, which you can read
        from `predict_errors_total`. Without the counter you would only see them as `400` lines in the API's log, or
        not at all if you don't read the log. A client that is silently failing is a service problem the model's
        accuracy will never show.

6. The solution comes with tests that run the API in-process with a tiny untrained model and synthetic images, with
    no data, network or Docker needed. Run them:

    ```bash
    uv run pytest test_api.py
    ```

    **Change the import in the `client` fixture from `api_solution` to `api` and run them against your own version.
    Which tests fail if you forget to count the errors?**

    ??? success "Solution"

        `test_metrics_expose_the_three_service_metrics` (no `predict_errors` family on `/metrics`) and all three cases
        of `test_malformed_request_is_counted_as_an_error`.

## The local stack: Prometheus and Grafana

Prometheus stores the metrics and lets you query them with **PromQL**. [Grafana](https://grafana.com/oss/grafana/)
draws dashboards from those queries and sends alerts. Both run as containers next to the API, started by one compose
file. All three ports are published on `127.0.0.1` only, so nothing is reachable from another machine.

??? example "The stack: `compose.yaml`, `api.dockerfile` and `prometheus.yml`"

    ```yaml linenums="1" title="compose.yaml"
    --8<-- "s8_monitoring/exercise_files/monitoring_stack/compose.yaml"
    ```

    ```dockerfile linenums="1" title="api.dockerfile"
    --8<-- "s8_monitoring/exercise_files/monitoring_stack/api.dockerfile"
    ```

    ```yaml linenums="1" title="prometheus.yml"
    --8<-- "s8_monitoring/exercise_files/monitoring_stack/prometheus.yml"
    ```

### ❔ Exercises

1. Stop the `uvicorn` from before (++ctrl+c++) and start the stack. The first build takes 5 to 10 minutes, mostly to
    download torch; later builds take seconds.

    ```bash
    docker compose up -d --build
    docker compose ps
    ```

    Open Prometheus at <http://localhost:9090>, then *Status* → *Target health*. The `api` target should be **UP**.
    **The Prometheus configuration scrapes `api:8000`, not `localhost:8000`. Why?**

    ??? success "Solution"

        Each container has its own network namespace, so `localhost` inside the Prometheus container is the
        Prometheus container itself. Compose puts all services on one network where each is reachable by its service
        name, so the API is `api:8000`. The same goes for Grafana reaching Prometheus at `http://prometheus:9090`. The
        `127.0.0.1:8000:8000` port mapping is only for you, on the host.

2. Generate some steady traffic (about a minute) and try these four starter queries in Prometheus, under *Query*, in
    the *Graph* tab:

    ```bash
    uv run python traffic.py data/rps-test-set --repeat 5 --delay 0.03
    ```

    ```txt
    sum(rate(predict_requests_total[1m]))
    histogram_quantile(0.95, sum by (le) (rate(predict_latency_seconds_bucket[1m])))
    sum(rate(predict_errors_total[1m]))
    sum by (predicted_class) (rate(predict_requests_total[1m]))
    ```

    **What does each query show, in which unit? What is the p95 latency? Is the service healthy?**

    ??? success "Solution"

        1. The **request rate**: answered requests per second, averaged over the last minute. `rate()` turns an
            ever-growing counter into a per-second speed, and handles the counter resetting when the API restarts.
        2. The **p95 latency** in seconds: 95% of the requests in the last minute were answered faster than this.
            `histogram_quantile` estimates it from the buckets, so it is only as precise as the buckets around it.
            Expect a few milliseconds to a few tens of milliseconds on a laptop.
        3. The **error rate**: rejected requests per second. Divide it by the total to get the share of requests that
            fail, which is what you alert on later.
        4. The **requests by predicted class**, one line per class. This one sits on the border between the service
            and the model layer: it is counted per request, but a change in the mix of predictions is an early hint
            that the input has changed.

        The service is healthy if the target is up, the latency is low and stable, and the error share stays at the
        level the traffic script causes (about 2%).

3. Open Grafana at <http://localhost:3000>. The Prometheus data source is already there: Grafana loads it at every
    start from `grafana/provisioning/datasources/prometheus.yml`. Build a dashboard (*Dashboards* → *New* → *New
    dashboard*) with four **rows**, one per monitoring layer: system, service, data and model. **Which of the metrics
    and queries from above belongs in which row?** The data and model rows stay placeholders until the next module.

    ??? success "Solution"

        | Row | Panels |
        | --- | --- |
        | System | target up, memory and CPU of the API process (`up` and the `process_*` metrics) |
        | Service | request rate (answered and rejected), p95 latency, errors by reason, requests by predicted class |
        | Data | placeholder: drift of the image features, from [Data drifting](data_drifting.md) |
        | Model | placeholder: share of predicted classes against the reference data, mean confidence |

        Set the unit of each panel (*Standard options* → *Unit*): requests/sec, seconds, bytes. A latency of `0.004`
        is easy to misread; `4 ms` is not.

4. Your dashboard lives only in Grafana's database, inside the `grafana-data` volume. Make it reproducible: export it
    (*Export* → *Export as JSON*; if Grafana offers a choice, take the classic JSON model and leave the option to export
    for another instance **off**), save it as `grafana/dashboards/monitoring.json`, then throw away everything and
    start again:

    ```bash
    docker compose down -v      # -v also deletes the volumes: Grafana's database and Prometheus' data
    docker compose up -d
    ```

    **Is your dashboard still there? Why does this matter once the stack moves to another machine?**

    ??? success "Solution"

        Yes: at start-up Grafana loads every JSON file in `/etc/grafana/dashboards`, which `compose.yaml` mounts from
        `grafana/dashboards/` (the provider is configured in `grafana/provisioning/dashboards/dashboards.yml`). A
        dashboard clicked together by hand would have been deleted with the volume. What lives in files can be
        reviewed, kept in git and moved to a new machine; that is exactly what you do next, when the stack moves to a
        VM. The panels refer to the data source by its uid, `prometheus`, which is fixed in the provisioning file, so
        the dashboard works on any machine with the same file.

        Our dashboard, if you want to compare or skip ahead (copy it to `grafana/dashboards/`):

        ```json linenums="1" title="grafana/dashboard_solution.json"
        --8<-- "s8_monitoring/exercise_files/monitoring_stack/grafana/dashboard_solution.json"
        ```

## Moving the stack to the cloud

In the cloud the API runs on **Cloud Run**, as in [M23 on cloud deployment](../s7_deployment/cloud_deployment.md).
Prometheus and Grafana can't run there: Cloud Run scales to zero and restarts containers whenever it likes, and a
monitoring system has to be running exactly when everything else is not. So they run on an always-on **Compute Engine
VM**, which you size yourself.

!!! warning "This part costs money"

    The VM and the Cloud Run service with a minimum of one instance are billed every hour they exist, whether you use
    them or not. The next modules reuse both, so keep them until the end of the day, but stop them as described in
    [Stop what is billed](#stop-what-is-billed) when you are done.

### ❔ Exercises

1. Build the API image for `linux/amd64` (needed on Apple Silicon, harmless elsewhere) and push it to Artifact Registry.
    This takes 5 to 10 minutes, mostly for the upload.

    ```bash
    export PROJECT=$(gcloud config get-value project) REGION=europe-west1 ZONE=europe-west1-b
    gcloud artifacts repositories create monitoring --repository-format=docker --location=$REGION
    gcloud auth configure-docker $REGION-docker.pkg.dev
    export IMAGE=$REGION-docker.pkg.dev/$PROJECT/monitoring/classifier-api:v1
    docker build --platform linux/amd64 -f api.dockerfile -t $IMAGE .
    docker push $IMAGE
    ```

2. Deploy it to Cloud Run with **exactly one instance** and **instance-based billing** (about 2 minutes):

    ```bash
    gcloud run deploy classifier-api --image=$IMAGE --region=$REGION \
        --min-instances=1 --max-instances=1 --no-cpu-throttling \
        --memory=1Gi --allow-unauthenticated
    ```

    `--no-cpu-throttling` is how `gcloud` spells instance-based billing (formerly "CPU always allocated"). Replay some
    traffic against the URL `gcloud` printed, and look at its `/metrics`:

    ```bash
    export API_URL=$(gcloud run services describe classifier-api --region=$REGION --format='value(status.url)')
    uv run python traffic.py data/rps-test-set --url $API_URL
    curl -s $API_URL/metrics | grep predict_
    ```

    **Why do we pin the service to exactly one instance, and why instance-based billing?**

    ??? success "Solution"

        The metrics live in the memory of the instance that counted them. With exactly one instance, every scrape of
        the URL reaches the same counters, so the numbers Prometheus stores are right. `--min-instances=1` keeps that
        instance from scaling to zero (which would reset every counter), `--max-instances=1` keeps a second one from
        appearing. With request-based billing Cloud Run throttles the CPU outside of requests; with instance-based
        billing the instance keeps its CPU between requests, which the next module needs when the API writes the
        production window in the background. Since the instance runs all the time anyway, you pay for it all the time
        in both models.

!!! warning "Why scraping a Cloud Run URL breaks once the service autoscales"

    Prometheus scrapes one URL, but behind it Cloud Run load-balances over all running instances. With three instances,
    each scrape reaches a random one, and each instance has its own counters: one scrape sees 120 requests, the next
    40, the next 300. `rate()` reads every drop as a counter reset and the graphs turn into noise. When an instance is
    shut down, its counts disappear with it. The setup in this module is therefore only correct while
    `max-instances=1`, which is fine for learning and wrong for a real service. Production setups do one of two
    things instead:

    * collect inside each instance: a collector runs as a **sidecar** container next to every instance, for example
        the [Managed Service for Prometheus sidecar for Cloud Run](https://cloud.google.com/stackdriver/docs/managed-prometheus/cloudrun-sidecar),
        and labels the metrics with the instance they came from;
    * **push** instead of being pulled: each instance sends its metrics with OTLP (see [OpenTelemetry](#opentelemetry)
        below) to a collector or to Prometheus' OTLP receiver.

!!! danger "`/metrics` is public"

    `--allow-unauthenticated` makes every endpoint public, `/metrics` included. Anyone with the URL can read how much
    traffic you get and what your model predicts, and Cloud Run serves a container on a single port, so you can't hide
    `/metrics` on a second one. That is acceptable for these exercises. In a real service you would require
    authentication on the endpoint, or use the sidecar or push setups above, which don't expose metrics on the public
    URL at all.

### ❔ Exercises

1. Before you create the VM, decide how big it has to be. Measure what Prometheus and Grafana use on your laptop while
    traffic runs:

    ```bash
    uv run python traffic.py data/rps-test-set --repeat 10 --delay 0.03 &
    docker stats --no-stream monitoring-stack-prometheus-1 monitoring-stack-grafana-1
    docker system df -v | grep monitoring-stack     # how much disk the volumes use so far
    gcloud compute machine-types list --zones=$ZONE --filter="name~'^e2-'"
    ```

    Click around your Grafana dashboard while you measure: Grafana uses most when someone is looking. **How much CPU
    and memory does the stack use? What else runs on the VM, and how much headroom do you add? Which machine type do
    you choose, and what does it cost per month?** Write the numbers down; you will check them on the VM.

    ??? success "Solution"

        There is no single right answer, only a method:

        1. Take the peak, not the average, of the CPU and memory columns of `docker stats` for each container, and add
            them up.
        2. Add what runs besides the containers: the operating system and the Docker daemon need a few hundred MB.
        3. Add what is coming: the next module adds a Pushgateway and a scheduled drift job on the same VM, and the
            drift job loads images into memory. Prometheus also grows with the number of time series and how long it
            keeps them (15 days by default); `docker system df -v` tells you the disk side.
        4. Add headroom, for example 50%: a VM that swaps or kills Prometheus for lack of memory stops monitoring
            exactly when you need it.
        5. Pick the smallest machine type above that, and look up its price in the
            [pricing calculator](https://cloud.google.com/products/calculator). Shared-core types (`e2-micro`,
            `e2-small`, `e2-medium`) are cheap and fine for bursty, light load, but check their memory against your
            sum.

        If you guessed wrong, you can change the machine type later: stop the VM,
        `gcloud compute instances set-machine-type`, start it again.

2. Create the VM with the machine type you chose and install Docker on it. This takes about 5 minutes.

    ```bash
    gcloud compute instances create monitoring-vm --zone=$ZONE --machine-type=<your choice> \
        --image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud --boot-disk-size=20GB
    gcloud compute ssh monitoring-vm --zone=$ZONE --command \
        "sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2 && sudo usermod -aG docker \$USER"
    ```

    We create **no firewall rules**. The only way into the VM is SSH, through the `default-allow-ssh` rule of the
    default network, and Prometheus and Grafana listen on the VM's `127.0.0.1` only.

3. Copy the configuration to the VM (the API stays on Cloud Run, so the VM needs no code, data or model) and fill in
    the `TODO` in `prometheus.cloud.yml` with the host name of your Cloud Run service:

    ```bash
    gcloud compute ssh monitoring-vm --zone=$ZONE --command "mkdir -p ~/monitoring_stack"
    gcloud compute scp --zone=$ZONE --recurse compose.yaml prometheus.cloud.yml grafana monitoring-vm:~/monitoring_stack/
    ```

    ```yaml linenums="1" title="prometheus.cloud.yml"
    --8<-- "s8_monitoring/exercise_files/monitoring_stack/prometheus.cloud.yml"
    ```

    Then start Prometheus and Grafana on the VM (pulling the images takes a minute or two):

    ```bash
    gcloud compute ssh monitoring-vm --zone=$ZONE
    cd ~/monitoring_stack
    PROMETHEUS_CONFIG=./prometheus.cloud.yml docker compose up -d prometheus grafana
    docker stats --no-stream
    ```

    ??? success "Solution"

        The target is the host name only, without `https://` and without a path: `scheme: https` makes Prometheus use
        HTTPS on port 443, and `/metrics` is the default path.

        ```yaml
            static_configs:
              - targets: ["classifier-api-abc123xyz-ew.a.run.app"]
        ```

        `echo $API_URL | sed 's#https://##'` gives you the host name. If you changed the file after starting, restart
        Prometheus: `PROMETHEUS_CONFIG=./prometheus.cloud.yml docker compose up -d --force-recreate prometheus`.

4. Reach Grafana through an **SSH tunnel**. Stop your local stack first (`docker compose down` on your laptop), because
    the tunnel uses the same ports:

    ```bash
    gcloud compute ssh monitoring-vm --zone=$ZONE -- -N -L 3000:localhost:3000 -L 9090:localhost:9090
    ```

    While this runs, <http://localhost:3000> on your laptop is Grafana on the VM, and <http://localhost:9090> is its
    Prometheus. Check that the `api` target is up and that your provisioned dashboard is there, then replay traffic
    against Cloud Run and watch it arrive. **Who on the internet can reach Grafana now? What happens to the monitoring
    when you close the tunnel?**

    ??? success "Solution"

        Only someone who can SSH into the VM, which `gcloud compute ssh` limits to members of your project with the
        right IAM role. Nothing listens on a public port: there is no firewall rule for 3000 or 9090, and the
        containers publish on `127.0.0.1` only, so even a firewall mistake would not expose them. That matters here,
        because this Grafana makes every visitor an admin. Closing the tunnel only closes your window into it:
        Prometheus keeps scraping and Grafana keeps evaluating alerts on the VM.

        For an even stricter setup, delete the `default-allow-ssh` rule and connect with `--tunnel-through-iap`: SSH
        then goes through Google's Identity-Aware Proxy and the VM needs no rule open to the internet at all.

5. Check your sizing: on the VM, run `docker stats --no-stream` and `free -m` while you replay traffic and use the
    dashboard. **Was your estimate right?**

## Alerting

A dashboard only helps when someone looks at it. An alert is a query that Grafana evaluates on a schedule, which sends a
message when the result crosses a threshold for long enough. The hard part is not the mechanics, it is choosing what
deserves to wake someone up. Too many alerts and people stop reading them (alert fatigue); too few and real problems
go unnoticed.

The rule of thumb is **alert on symptoms, put drift on dashboards.** Page a human for what users feel right now: the
service fails requests, or answers slowly. Those are symptoms, they are clear-cut, and someone can act on them at once.
Data drift and model-layer signals are different: a drift statistic crossing a threshold does not mean that users are
affected, the thresholds are judgement calls, and the right response (look into it, collect labels, retrain) is days of
work, not a 3 a.m. fix. They belong on the dashboard you look at during the day, which is where the next module puts
them.

### ❔ Exercises

1. Create a channel for alerts in Discord: on a server you own (create one if needed), *Edit Channel* → *Integrations*
    → *Webhooks* → *New Webhook* → *Copy Webhook URL*. Treat the URL as a secret: anyone who has it can post to your
    channel. Don't commit it.

2. In Grafana, add a **contact point**: *Alerting* → *Contact points* → *Create contact point*, integration
    **Discord**, paste the webhook URL. Use *Test* to check that a message arrives. The contact point is stored in the
    `grafana-data` volume on the VM, not in your files.

3. Create an **alert rule** on the share of failing requests (*Alerting* → *Alert rules* → *New alert rule*). Fire
    when more than 5% of requests fail for a minute, and send it to your Discord contact point. **What do you set for
    "no data", and why?**

    ??? success "Solution"

        * Query (data source Prometheus):

            ```txt
            sum(rate(predict_errors_total[1m]))
              / (sum(rate(predict_requests_total[1m])) + sum(rate(predict_errors_total[1m])))
            ```

        * Alert condition: the query is above `0.05` (in the advanced options: *Reduce* with *Last*, then
            *Threshold* is above `0.05`).
        * Folder and evaluation group: any, evaluated every `1m`, with a pending period of `1m`.
        * *Configure no data and error handling*: set **Alert state if no data** to **Normal**. Until the first error
            the error counter has no series at all (the `# HELP`/`# TYPE` exercise), so the query returns no data, and
            without traffic the division has nothing to divide. Neither means the service is failing. (If the API
            itself is down, `up{job="api"} == 0` is the signal for that, and would be a good second alert.)
        * Contact point: your Discord contact point. Add a summary, for example *More than 5% of /predict requests
            fail*.

        A p95 latency rule works the same way, with
        `histogram_quantile(0.95, sum by (le) (rate(predict_latency_seconds_bucket[1m])))` and a threshold in seconds.

4. Trigger it on purpose, with one malformed request for every second image:

    ```bash
    uv run python traffic.py data/rps-test-set --url $API_URL --malformed-every 2 --repeat 3 --delay 0.1
    ```

    Watch the rule go from *Normal* to *Pending* to *Firing* under *Alerting* → *Alert rules*, and the message arrive
    in Discord. A few minutes after the traffic stops, a *Resolved* message follows. **How long did it take from the
    first bad request to the message? Which settings add up to that delay?**

    ??? success "Solution"

        Usually two to four minutes. The delay adds up from the scrape interval (15 s on the VM), the `[1m]` window of
        `rate()` that has to fill with errors, the evaluation interval (1 min), the pending period (1 min) and the
        notification policy's group wait (30 s by default). Each of them trades speed against false alarms: a shorter
        pending period pages you for every short blip.

## OpenTelemetry

Prometheus metrics are one kind of telemetry; logs and traces (the path of one request through several services) are
the other two. [OpenTelemetry](https://opentelemetry.io/) (OTel) is the vendor-neutral standard for producing all
three: you instrument your code once with the OTel SDK, and send the data with its protocol, **OTLP**, to an
**OpenTelemetry Collector** that processes it and forwards it to whatever backend you use. Prometheus itself can take
metrics pushed over OTLP through its [OTLP receiver](https://prometheus.io/docs/guides/opentelemetry/), which is the
push answer to the autoscaling problem above. We use `prometheus_client` in this course because pull-based metrics are
the simplest to see and reason about, but in a new production system OTel is a good default.

## Apply it to your project

Do these in project time, not during the module:

1. Instrument your project's API with a request counter, a latency histogram with buckets that fit your latencies, and
    an error counter by reason.
2. Run Prometheus and Grafana for your project on an always-on VM that you sized from measurements, reached through an
    SSH tunnel, with the dashboard and data source provisioned from files in your repository.
3. Add a Grafana alert on a service metric that notifies a channel your group reads.

## Stop what is billed

The next module, [Data drifting](data_drifting.md), adds to the same VM and service. When you are done for the day:

```bash
# The API: allow it to scale to zero (it then costs nothing while idle) ...
gcloud run services update classifier-api --region=$REGION --min-instances=0
# ... and stop the VM (you still pay a little for its disk while it is stopped).
gcloud compute instances stop monitoring-vm --zone=$ZONE
```

To continue, `gcloud compute instances start monitoring-vm --zone=$ZONE` (the containers start by themselves) and set
`--min-instances=1` again. When you are done with the session for good, delete everything:

```bash
gcloud run services delete classifier-api --region=$REGION
gcloud compute instances delete monitoring-vm --zone=$ZONE
gcloud artifacts repositories delete monitoring --location=$REGION
```
