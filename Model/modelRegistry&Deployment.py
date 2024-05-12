from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.environment import Environment
from azureml.core import Workspace, Model
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.conda_dependencies import CondaDependencies

ws = Workspace.from_config()

model = Model.register(model_path="lstm_model.h5",
                       model_name="lstm_model",
                       description="LSTM model for predicting request loads",
                       workspace=ws)

scaler = Model.register(model_path="scaler.joblib",
                        model_name="scaler",
                        description="MinMaxScaler for LSTM model",
                        workspace=ws)

env = Environment.from_conda_specification(name='lstm_env', file_path='environment.yml')

inference_config = InferenceConfig(entry_script='score.py', environment=env)

deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

service_to_delete = Webservice(workspace=ws, name="lstm-service")
service_to_delete.delete()

service = Model.deploy(workspace=ws, 
                       name='lstm-service', 
                       models=[model, scaler], 
                       inference_config=inference_config, 
                       deployment_config=deployment_config)

service.wait_for_deployment(show_output=True)
