import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple

warnings.filterwarnings("ignore")


def load_data(dataset_loc: str) -> pd.DataFrame:
    """Loads the data

    Args:
        dataset_loc (str): Dataset filepath

    Returns:
        pd.DataFrame: Pandas dataframe
    """
    df = pd.read_csv(dataset_loc)
    return df


def stratify_split(df: pd.DataFrame, stratify: str, test_size: float, seed: int = 1234) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits data to train and val sets

    Args:
        df (pd.DataFrame): Input data
        stratify (np.array): Column of target variable
        test_size (float): Proportion of data
        seed (int, optional): Random seed. Defaults to 1234.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and val dataframes
    """
    print(stratify)
    train_df, val_df = train_test_split(df, stratify=stratify, test_size=test_size, random_state=seed)
    return train_df, val_df


def encode_categorical(df: pd.DataFrame, column_name: str) -> dict:
    """Encodes categorical features

    Args:
        df (pd.DataFrame): Input dataframe
        column_name (str): Name of categorical column

    Returns:
        dict: Mapping dictionary
    """
    categories = df[column_name].unique().tolist()
    category_to_index = {pt_type: i for i, pt_type in enumerate(categories)}
    return category_to_index


def preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess input data

    Args:
        df (pd.DataFrame): Raw data

    Returns:
        Tuple[np.ndarray, np.ndarray]: X and y instances
    """
    df = df[["SHOT_CLOCK", "DRIBBLES", "TOUCH_TIME", "SHOT_DIST", "PTS_TYPE", "CLOSE_DEF_DIST", "SHOT_RESULT"]]
    df.loc[:, "SHOT_CLOCK"] = df["SHOT_CLOCK"].fillna(0.0)
    pts_type_to_index = encode_categorical(df, column_name="PTS_TYPE")
    df.loc[:, "PTS_TYPE"] = df["PTS_TYPE"].map(pts_type_to_index)
    shot_result_to_index = encode_categorical(df, column_name="SHOT_RESULT")
    df.loc[:, "SHOT_RESULT"] = df["SHOT_RESULT"].map(shot_result_to_index)
    X = df.drop(columns="SHOT_RESULT").values
    y = df["SHOT_RESULT"].values
    return X, y


def convert_data_to_features(data: dict) ->Tuple[float, int, float, float, int, float]:
    """Convert input dict to Tuple of features

    Args:
        data (dict): Request dict

    Returns:
        Tuple[float, int, float, float, int, float]: _description_
    """
    shot_clock = data.shot_clock
    dribbles = data.dribbles
    touch_time = data.touch_time
    shot_dist = data.shot_dist
    pts_type = data.pts_type
    close_def_dist = data.close_def_dist
    return shot_clock, dribbles, touch_time, shot_dist, pts_type, close_def_dist
