import pandas as pd
import pytest
from app.data import load_data, stratify_split
from app.config import DATASET_LOC

def test_load_data_success():
    """Test loading data successfully."""
    # Assuming shot_logs.csv exists and has the correct format
    df = load_data(DATASET_LOC)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_load_data_failure():
    """Test loading data with invalid file path."""
    with pytest.raises(FileNotFoundError):
        load_data('non_existent_file.csv')

def test_stratify_split():
    """Test stratify split of the data."""
    # Create a sample DataFrame
    df = pd.DataFrame({
        'feature1': range(10),
        'stratify_col': [0, 1, 1, 1, 1, 0, 1, 0, 1, 1]
    })
    train_df, val_df = stratify_split(df, df.stratify_col, 0.2)
    assert not train_df.empty and not val_df.empty
    assert len(train_df) > len(val_df)