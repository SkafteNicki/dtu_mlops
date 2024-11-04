import pickle

import functions_framework
from google.cloud import storage

BUCKET_NAME = "my_sklearn_model_bucket"
MODEL_FILE = "model.pkl"

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.get_blob(MODEL_FILE)
my_model = pickle.loads(blob.download_as_string())


@functions_framework.http
def knn_classifier(request):
    """Simple knn classifier function for iris prediction."""
    request_json = request.get_json()
    if request_json and "input_data" in request_json:
        input_data = request_json["input_data"]
        input_data = [float(in_data) for in_data in input_data]
        input_data = [input_data]
        prediction = my_model.predict(input_data)
        return {"prediction": prediction.tolist()}
    else:
        return {"error": "No input data provided."}
