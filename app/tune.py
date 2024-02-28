import typer
import ray
import mlflow
import xgboost as xgb
from hyperopt import hp, fmin, tpe, STATUS_OK
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data import load_data, stratify_split, preprocess
import utils
from typing_extensions import Annotated
import warnings
import datetime
import json
from config import logger, DATASET_LOC

warnings.filterwarnings("ignore")

app = typer.Typer()

@app.command()
def tune_model(experiment_name: Annotated[str, typer.Option()] = 'tune-shot-predictor',
               results_filepath: Annotated[str, typer.Option()] = 'results/tuning_results.json'
               ):
    df = load_data(DATASET_LOC)
    train_df, val_df = stratify_split(df, df.SHOT_RESULT, 0.2)
    X_train, y_train = preprocess(train_df)
    X_val, y_val = preprocess(val_df)

    # Set the experiment name
    mlflow.set_experiment(experiment_name)

    # Define the search space
    space = {
        'max_depth': hp.choice('max_depth', range(1, 9)),
        'min_child_weight': hp.choice('min_child_weight', range(0, 10)),
        'learning_rate': hp.choice('learning_rate', [0.0001, 0.001, 0.01]),
        'n_estimators': hp.choice('n_estimators', range(100, 1000)),
        'gamma': hp.uniform('gamma', 0, 5),
    }

    # Define the objective function
    def objective(params):
        with mlflow.start_run():

            # Train model
            model = xgb.XGBClassifier(**params, verbosity=0)
            model.fit(X_train, y_train)

            # Evaluate model
            predictions = model.predict(X_val)
            accuracy = accuracy_score(y_val, predictions)
            precision = precision_score(y_val, predictions)
            recall = recall_score(y_val, predictions)
            f1 = f1_score(y_val, predictions)

            # Log to MLflow
            mlflow.log_params(params)
            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)
            mlflow.log_metric('f1 score', f1)
            mlflow.xgboost.log_model(model, 'model')

            return {'loss': -f1, 'status': STATUS_OK}

    # Perform optimization
    fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=1)

    sorted_runs = mlflow.search_runs(experiment_names=[experiment_name], order_by=["metrics.f1_score DESC"])

    top_run = sorted_runs.head(1)

    d = {
        "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "params": top_run[['params.max_depth', 'params.min_child_weight',
                           'params.learning_rate', 'params.n_estimators',
                             'params.gamma']]
                             .to_dict(orient='records'),
        "metrics": top_run[['metrics.accuracy', 'metrics.precision',
                             'metrics.recall', 'metrics.f1 score']]
                             .to_dict(orient='records')
    }
    logger.info(json.dumps(d, indent=2))

    if results_filepath:
        utils.save_dict(d, results_filepath)

if __name__ == "__main__":
    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    app()