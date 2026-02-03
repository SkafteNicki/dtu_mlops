deploy:
	gcloud run deploy conceptual-qa \
		--source . \
		--platform managed \
		--region europe-west1 \
		--allow-unauthenticated

	gcloud run services add-iam-policy-binding \
    	--region=europe-west1 \
    	--member=allUsers \
    	--role=roles/run.invoker \
		conceptual-qa

	gcloud run services describe conceptual-qa --region=europe-west1 --format="value(status.url)"
