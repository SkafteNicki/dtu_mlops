## This defines all targets as phony targets, i.e. targets that are always out of date
## This is done to ensure that the commands are always executed, even if a file with the same name exists
## See https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html
## Remove this if you want to use this Makefile for real targets
.PHONY: *

GCP_PROJECT_NAME = $(shell gcloud config get-value project)

# construct service account key file name
# upload as github secret and use in github actions
service_account:
	gcloud iam service-accounts create repo-stats-account \
		--description="Service account for repo stats application" --display-name="repo-stats-account"
	gcloud projects add-iam-policy-binding $(GCP_PROJECT_NAME) \
		--member="serviceAccount:repo-stats-account@$(GCP_PROJECT_NAME).iam.gserviceaccount.com" \
		--role="roles/storage.objectUser"
		--project
	gcloud iam service-accounts keys create service_account_key.json \
		--iam-account=repo-stats-account@$(GCP_PROJECT_NAME).iam.gserviceaccount.com
	echo service_account_key.json >> .gitignore

# deploy the leaderboard service
deploy_leaderboard:
	gcloud run deploy repo-stats-leaderboard \
		--source . \
		--platform managed \
		--region europe-west1 \
		--allow-unauthenticated

	gcloud run services add-iam-policy-binding \
    	--region=europe-west1 \
    	--member=allUsers \
    	--role=roles/run.invoker \
		repo-stats-leaderboard

	gcloud run services describe repo-stats-leaderboard --region=europe-west1 --format="value(status.url)"
