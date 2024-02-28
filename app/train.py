import json
import os

import ray
import typer
import utils
import xgboost as xgb
from config import DATASET_LOC, logger
from data import load_data, preprocess, stratify_split
from typing_extensions import Annotated

app = typer.Typer()


@app.command()
def train_model(train_config: Annotated[str, typer.Option()] = None, save_path: Annotated[str, typer.Option()] = "saved_models/xgb_cls.pkl"):
    df = load_data(DATASET_LOC)
    train_df, _ = stratify_split(df, df.SHOT_RESULT, 0.2)
    X_train, y_train = preprocess(train_df)

    # Read hyperparameters provided from cli
    train_config = json.loads(train_config)

    # Train model
    model = xgb.XGBClassifier(**train_config, verbosity=0)
    model.fit(X_train, y_train)

    utils.save_model(model, save_path)

    logger.info(f"Training completed. Trained model saved in {os.path.dirname(save_path)}")


if __name__ == "__main__":
    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    app()
