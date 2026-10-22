.PHONY: *

requirements:
	pip install -r requirements_backend.txt
	pip install -r requirements_frontend.txt

backend:
	uvicorn backend:app

request_backend:
	curl -X 'POST' \
  		'http://127.0.0.1:8000/classify/' \
  		-H 'accept: application/json' \
  		-H 'Content-Type: multipart/form-data' \
  		-F 'file=@my_cat.jpg;type=image/jpeg'

frontend:
	BACKEND=http://127.0.0.1:8000 streamlit run frontend.py

build:
	docker build -t backend:latest -f backend.dockerfile .
	docker build -t frontend:latest -f frontend.dockerfile .

tag_and_push: build
	docker tag backend:latest europe-west1-docker.pkg.dev/my-personal-mlops-project/frontend-backend/backend:latest
	docker tag frontend:latest europe-west1-docker.pkg.dev/my-personal-mlops-project/frontend-backend/frontend:latest
	docker push europe-west1-docker.pkg.dev/my-personal-mlops-project/frontend-backend/backend:latest
	docker push europe-west1-docker.pkg.dev/my-personal-mlops-project/frontend-backend/frontend:latest

backend_deploy:
	gcloud run deploy backend \
		--image=europe-west1-docker.pkg.dev/my-personal-mlops-project/frontend-backend/backend:latest
		--platform=managed \
		--region=europe-west1

frontend_deploy:
	gcloud run deploy frontend \
		--image=europe-west1-docker.pkg.dev/my-personal-mlops-project/frontend-backend/frontend:latest
		--platform=managed \
		--region=europe-west1

backend_docker:
	docker run -p 8000:8000 --rm -e "PORT=8000" backend:latest

frontend_docker:
	docker run -p 8001:8001 --rm -e "PORT=8001" frontend:latest
