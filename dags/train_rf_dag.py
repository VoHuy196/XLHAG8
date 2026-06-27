from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import numpy as np
import json
import os

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5001')
AIRFLOW_HOME = '/opt/airflow'
DATA_DIR = os.path.join(AIRFLOW_HOME, 'data')
MODELS_DIR = os.path.join(AIRFLOW_HOME, 'models')

def load_array(file_name):
    file_path = os.path.join(DATA_DIR, file_name)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Missing required data file: {file_path}")
    return np.load(file_path)

def run_rf_training(**context):
    print("Starting Plant Disease RF Training with MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    experiment_name = "xlhag8_plant_disease"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id

    # Load data
    try:
        X_train = load_array("X_train.npy")
        X_test = load_array("X_test.npy")
        y_train = load_array("y_train.npy")
        y_test = load_array("y_test.npy")
    except Exception as e:
        print(f"Data not found, skipping training: {e}")
        return

    n_estimators_values = [100, 200]
    depth_values = [20, 30]

    best_f1 = -1
    best_model = None
    best_run_id = None
    best_params = {}

    for n_est in n_estimators_values:
        for max_depth in depth_values:
            with mlflow.start_run(experiment_id=experiment_id) as run:
                print(f"Training with n_estimators={n_est}, max_depth={max_depth}")
                model = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=max_depth,
                    class_weight='balanced',
                    n_jobs=-1,
                    random_state=42
                )
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                
                mlflow.log_params({"n_estimators": n_est, "max_depth": max_depth})
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_score", f1)
                
                mlflow.sklearn.log_model(model, "model")
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model
                    best_run_id = run.info.run_id
                    best_params = {"n_estimators": n_est, "max_depth": max_depth}

    print(f"Best model F1: {best_f1:.4f} with params: {best_params}")
    
    # Save the best model to models directory for the API
    os.makedirs(MODELS_DIR, exist_ok=True)
    p_label = f"n{best_params['n_estimators']}_depth{best_params['max_depth']}"
    model_path = os.path.join(MODELS_DIR, f"rf_best_plant_{p_label}.pkl")
    joblib.dump(best_model, model_path)
    print(f"Best model saved to {model_path}")

default_args = {
    'owner': 'ml_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    dag_id='xlhag8_rf_training',
    default_args=default_args,
    description='Train RF model for plant disease detection',
    schedule_interval='@weekly',
    catchup=False,
    tags=['ml', 'mlflow', 'xlhag8'],
) as dag:
    from airflow.operators.bash import BashOperator

    download_data_task = BashOperator(
        task_id='download_kaggle_dataset',
        bash_command="""
            mkdir -p /opt/airflow/data/dataset
            if [ ! -d "/opt/airflow/data/dataset/Tomato" ] && [ ! -d "/opt/airflow/data/dataset/Tomato___Bacterial_spot" ] && [ -z "$(ls -A /opt/airflow/data/dataset)" ]; then
                echo "Downloading dataset..."
                kaggle datasets download -d emmarex/plantdisease -p /opt/airflow/data/dataset --unzip
                
                # In case kaggle extracts into a subfolder like 'plantvillage' or 'PlantVillage', move them up
                if [ -d "/opt/airflow/data/dataset/plantvillage" ]; then
                    mv /opt/airflow/data/dataset/plantvillage/* /opt/airflow/data/dataset/
                elif [ -d "/opt/airflow/data/dataset/PlantVillage" ]; then
                    mv /opt/airflow/data/dataset/PlantVillage/* /opt/airflow/data/dataset/
                fi
            else
                echo "Dataset already exists, skipping download."
            fi
        """,
    )

    extract_features_task = BashOperator(
        task_id='extract_features',
        bash_command='python /opt/airflow/src/feature_extraction.py',
    )

    preprocessing_task = BashOperator(
        task_id='preprocessing',
        bash_command='python /opt/airflow/src/preprocessing.py',
    )

    task_experiment = PythonOperator(
        task_id='run_rf_training',
        python_callable=run_rf_training,
        provide_context=True,
    )

    download_data_task >> extract_features_task >> preprocessing_task >> task_experiment
