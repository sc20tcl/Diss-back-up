from azureml.core import Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core import Workspace, Model

ws = Workspace.from_config()

model = Model.register(model_path="lstm_model.pkl",
                       model_name="lstm_model",
                       description="LSTM model for predicting request loads",
                       workspace=ws)

scaler = Model.register(model_path="scaler.joblib",
                        model_name="scaler",
                        description="MinMaxScaler for LSTM model",
                        workspace=ws)

env = Environment(name="project_environment")
dummy_inference_config = InferenceConfig(
    environment=env,
    source_directory="../",
    entry_script="echo_score.py",
)

deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

service = Model.deploy(
    ws,
    "myservice",
    [model, scaler],
    dummy_inference_config,
    deployment_config,
    overwrite=True,
)
service.wait_for_deployment(show_output=True)

print(service.get_logs())