import dagshub
import mlflow
import mlflow.sklearn


def setup_mlflow():
    """
    Initializes MLflow tracking for the project.
    """

    dagshub.init(repo_owner='kemiopendoor247', 
             repo_name='Fraudulent_Transaction_Detection_for_Finlora_Company', 
             mlflow=True)

    # Set experiment
    mlflow.set_experiment("Fraudulent_Transaction_Detection_for_Finlora_Company")