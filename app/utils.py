import json
import os
import pickle
import xgboost as xgb


def save_dict(d: dict, path: str, sortkeys: bool = False) -> None:
    """Saves dictionary to memory

    Args:
        d (dict): Dictionary to save
        path (str): Save path
        sortkeys (bool, optional): Sort keys on keys. Defaults to False.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, "w") as fp:
        json.dump(d, indent=2, fp=fp, sort_keys=sortkeys)
        fp.write("\n")


def save_model(model: xgb.XGBClassifier, path: str) -> None:
    """Saves the trained model to memory

    Args:
        model (xgb.XGBClassifier): Model to be saved
        path (str): Save path
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    pickle.dump(model, open(path, "wb"))


def load_model(path: str) -> xgb.XGBClassifier:
    """Loads the saved model

    Args:
        path (str): Save path 

    Returns:
        xgb.XGBClassifier: Trained model
    """
    model = pickle.load(open(path, "rb"))
    return model
