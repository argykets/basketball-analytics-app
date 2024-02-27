import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

def load_data(dataset_loc):
    df = pd.read_csv(dataset_loc)
    return df

def stratify_split(df, stratify, test_size, seed=1234):
    # Split dataset
    train_df,  val_df = train_test_split(df, stratify=stratify,
                                            test_size=test_size,
                                            random_state=seed)
    return train_df, val_df

def encode_categorical(df, column_name):
    categories = df[column_name].unique().tolist()
    category_to_index = {pt_type: i for i, pt_type in enumerate(categories)}
    return category_to_index

def preprocess(df):
    """Preprocess the data"""
    df = df[['SHOT_CLOCK', 'DRIBBLES', 'TOUCH_TIME', 'SHOT_DIST', 'PTS_TYPE', 'CLOSE_DEF_DIST', 'SHOT_RESULT']]
    df.loc[:, 'SHOT_CLOCK'] = df['SHOT_CLOCK'].fillna(0.0)
    pts_type_to_index = encode_categorical(df, column_name='PTS_TYPE')
    df.loc[:, 'PTS_TYPE'] = df['PTS_TYPE'].map(pts_type_to_index)
    shot_result_to_index = encode_categorical(df, column_name='SHOT_RESULT')
    df.loc[:, 'SHOT_RESULT'] = df['SHOT_RESULT'].map(shot_result_to_index)
    X = df.drop(columns='SHOT_RESULT').values
    y = df['SHOT_RESULT'].values
    return X, y
