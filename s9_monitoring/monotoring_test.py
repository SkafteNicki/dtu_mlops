import time
import requests
url = 'https://us-central1-dtumlops-337908.cloudfunctions.net/test_function'
payload = {'message': 'Test error message'}

for _ in range(1000):
   r = requests.get(url, params=payload)