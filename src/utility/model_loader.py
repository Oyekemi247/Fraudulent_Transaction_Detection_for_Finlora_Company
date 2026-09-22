import mlflow.sklearn
import mlflow
from src.utility.mlflow_setup import setup_mlflow

def load_registered_model():
    """
    Load registered model from mlflow 
    """

    setup_mlflow()

    model_name = "Fraud_Detecion_XGBoost_pipeline"
    model_version = "latest"

    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)

    return model 


load_registered_model()