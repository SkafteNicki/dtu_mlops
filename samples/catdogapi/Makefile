.PHONY: *

SERVICE_NAME=catdogapi
REGION=europe-west1

build:
	docker build . -t $(SERVICE_NAME):latest

run: build
	docker run -p 5000:5000 -e PORT=5000 $(SERVICE_NAME):latest

invoke:
	curl -o http://localhost:5000/catordog

deploy:
	gcloud run deploy $(SERVICE_NAME) --source . --allow-unauthenticated

endpoint:
	gcloud run services describe $(SERVICE_NAME) --region=$(REGION) --format="value(status.url)"

request:
	curl -X 'GET' \
		$(shell gcloud run services describe $(SERVICE_NAME) --region=$(REGION) --format="value(status.url)")/catordog \
		-H 'accept: application/json' > image.jpg
