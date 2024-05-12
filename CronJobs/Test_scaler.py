import math
import requests
from kubernetes import client, config
import time
from datetime import datetime, timedelta

traffic_history = []

def query_traffic(timestamp):
    base_url = 'http://20.33.62.78:5001/traffic'

    encoded_timestamp = timestamp.replace(' ', '%20')
    full_url = f"{base_url}?timestamp={encoded_timestamp}"
    response = requests.get(base_url)
    print(base_url)
    
    if response:
        json_response = response.json() 
        traffic_value = json_response.get('traffic', 0) 
        return traffic_value
    else:
        return 0

def calculate_desired_pods(requests_per_minute):
    return math.ceil(requests_per_minute / 5000)

def predict_traffic(requests_per_minute):
    # URL of the prediction service
    url = 'http://20.33.62.78:5000/predict'
    
    # Headers to specify that the request body is JSON
    headers = {'Content-Type': 'application/json'}
    
    # Data to be sent in JSON format
    data = {
        'current_traffic': requests_per_minute
    }
    
    # Send a POST request
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        # Parse the JSON response
        json_response = response.json()
        # Extract the forecast value
        forecast_value = float(json_response['forecast'][0])
        return forecast_value
    else:
        print(f"Failed to get prediction, status code: {response.status_code}")
        return None

def scale_deployment(num_pods):
    config.load_kube_config()
    k8s_client = client.AppsV1Api()
    
    deployment_name = 'teastore-webui'
    namespace = 'default'
    
    k8s_client.patch_namespaced_deployment_scale(
        name=deployment_name,
        namespace=namespace,
        body={'spec': {'replicas': num_pods}}
    )

def main(time_stamp):
    requests_per_minute = query_traffic(time_stamp)
    print("requests per minute: ", requests_per_minute)

    forcasted_traffic = predict_traffic(requests_per_minute)
    print("forcasted traffic: ", forcasted_traffic)
    if forcasted_traffic != None:
        num_pods = calculate_desired_pods(forcasted_traffic)

        scale_deployment(num_pods)
        print(f"Deployment scaled to {num_pods} pods based on {requests_per_minute} requests per minute.")
    else:
        print("Deployment failed to scale")

if __name__ == "__main__":
    response = requests.post('http://20.33.62.78:5001/reset')
    print(response)
    start_time = datetime.strptime("1998-06-24 00:00:00", "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime("1998-06-24 00:16:00", "%Y-%m-%d %H:%M:%S")

    current_time = start_time

    while current_time <= end_time:
        main(current_time.strftime("%Y-%m-%d %H:%M:%S"))

        time.sleep(5)
        current_time += timedelta(minutes=1)
