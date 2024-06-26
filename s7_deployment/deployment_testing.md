![Logo](../figures/icons/cloudrun.png){ align=right width="130"}

# Deployment Testing

https://cloud.google.com/architecture/application-deployment-and-testing-strategies

The testing patterns discussed in this section are typically used to validate a service's reliability and stability over 
a reasonable period under a realistic level of concurrency and load.

In today's dynamic software development landscape, ensuring seamless and reliable application deployment is crucial for maintaining user satisfaction and operational efficiency. This module will introduce you to various deployment testing patterns, each designed to minimize risk, enhance performance, and ensure a smooth transition from development to production. Whether you're aiming for zero downtime, incremental feature releases, or robust rollback capabilities, understanding these strategies will empower you to implement effective deployment workflows. Join us as we explore canary deployments, blue-green deployments, feature toggles, and more, equipping you with the knowledge to optimize your deployment processes and deliver high-quality software with confidence.


## A/B testing

### ❔ Exercises

1. Geolocation

    ```python
    from fastapi import FastAPI, Request, HTTPException
    import geoip2.database

    app = FastAPI()

    # Load the GeoLite2 database
    reader = geoip2.database.Reader('/path/to/GeoLite2-City.mmdb')

    def get_client_ip(request: Request) -> str:
        if "X-Forwarded-For" in request.headers:
            return request.headers["X-Forwarded-For"].split(",")[0]
        if "X-Real-IP" in request.headers:
            return request.headers["X-Real-IP"]
        return request.client.host

    def get_geolocation(ip: str) -> dict:
        try:
            response = reader.city(ip)
            return {
                "ip": ip,
                "city": response.city.name,
                "region": response.subdivisions.most_specific.name,
                "country": response.country.name,
                "location": {
                    "latitude": response.location.latitude,
                    "longitude": response.location.longitude
                }
            }
        except geoip2.errors.AddressNotFoundError:
            raise HTTPException(status_code=404, detail="IP address not found in the database")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/geolocation")
    async def geolocation(request: Request):
        client_ip = get_client_ip(request)
        geolocation_data = get_geolocation(client_ip)
        return geolocation_data
    ```

## Canary deployment

### ❔ Exercises

Follow [these](https://cloud.google.com/architecture/implementing-cloud-run-canary-deployments-git-branches-cloud-build)
instructions to implement a canary deployment using Git branches and Cloud Build.

1. Use the following command

    ```bash
    gcloud run services update-traffic
    ```

## Shadow deployment

### ❔ Exercises

1. Google Run does not naturally support shadow deployments, because its loadbalancer requires that the traffic adds up
    to 100%, and for shadow deployments, it would be 200%. To proper implement this you would need to use a kubernetes
    cluster and use a service mesh like Istio. So instead we are going to implement a very simple load balancer ourself.

    1. Create a new script called `loadbalancer.py` and add the following code

        ```python
        import random
        from fastapi import FastAPI, HTTPException
        import requests

        app = FastAPI()

        services = {
            "service1": "http://localhost:8000",
            "service2": "http://localhost:8001"
        }

        @app.get("/shadow")
        async def shadow():
            service = random.choice(list(services.keys()))
            response = requests.get(services[service] + "/shadow")
            return {
                "service": service,
                "response": response.json()
            }
        ```
    
    2. Because the loadbalancer is just a simple Python script lets just deploy it to Cloud Functions instead of Cloud
        Run. Create a new Cloud Function and deploy the script.

## 🧠 Knowledge check

1. Try to fill out the following table:

    Testing patter | Zero downtime | Real production traffic testing | Releasing to users based on conditions | Rollback duration | Releasing to users based on conditions |
    --------------- | -------------- | -------------------------------- | -------------------------------------- | ----------------- | -------------------------------------- |
    A/B testing     |                |                                  |                                        |                   |                                        |
    Canary deployment |              |                                  |                                        |                   |                                        |
    Shadow deployment |              |                                  |                                        |                   |                                        |

    ??? success "Solution"

        Testing patter | Zero downtime | Real production traffic testing | Releasing to users based on conditions | Rollback duration | Releasing to users based on conditions |
        --------------- | -------------- | -------------------------------- | -------------------------------------- | ----------------- | -------------------------------------- |
        A/B testing     | No             | No                               | Yes                                    | Short             | No                                     |
        Canary deployment | Yes           | Yes                              | Yes                                    | Short             | Yes                                    |
        Shadow deployment | Yes           | No                               | Yes                                    | Short             | Yes                                     |