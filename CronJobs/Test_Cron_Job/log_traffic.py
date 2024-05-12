import requests
import logging
from kubernetes import client, config
import math

def get_traffic(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  
        data = response.json()
        traffic = data.get('traffic', 0)  
        logging.info(f"Successfully called {url}: {response.text}")
        return traffic
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling {url}: {str(e)}")
        return None

def scale_deployment(num_pods):
    try:
        # Use in-cluster configuration instead of kube-config
        config.load_incluster_config()
        k8s_client = client.AppsV1Api()
        
        deployment_name = 'teastore-webui'
        namespace = 'default'
        
        # Scale the deployment
        k8s_client.patch_namespaced_deployment_scale(
            name=deployment_name,
            namespace=namespace,
            body={'spec': {'replicas': num_pods}}
        )
        logging.info(f"Deployment {deployment_name} scaled to {num_pods} pods.")
    except Exception as e:
        logging.error(f"Failed to scale deployment: {str(e)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    url = 'http://20.33.62.78:5001/traffic'

    traffic = get_traffic(url)
    if traffic is not None:
        num_pods = math.floor(traffic / 1000)
        scale_deployment(num_pods)
