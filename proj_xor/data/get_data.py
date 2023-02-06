from tensorflow.keras import utils as tf_utils
from tensorflow import convert_to_tensor
from importlib_resources import files
from tensorflow.data import Dataset
from pandas import read_csv
from tensorflow.dtypes import bool as tfbool, float32


_dataset_path = files("proj_xor.data.datasets")
_dtype = float32
# _int_dtype = 

def train_data():
    df = read_csv(
        _dataset_path.joinpath("training_data.txt"),
        sep=" ",
        names=["labels", "X1", "X2"],
        dtype={"labels": int, "X1": float, "X2": float},
        skipinitialspace=1,
    )

    labels = convert_to_tensor(
        df["labels"],
        dtype=tfbool,
        name="train_labels",
    )

    data = convert_to_tensor(
        df[["X1", "X2"]],
        dtype=_dtype,
        name="train_data",
    )

    dataset = Dataset.from_tensors(
        (data, labels),
        name="train_dataset",
    )
    return dataset


def test_data():
    df = read_csv(
        _dataset_path.joinpath("test_data.txt"),
        sep=" ",
        names=["labels", "X1", "X2"],
        dtype={"labels": int, "X1": float, "X2": float},
        skipinitialspace=1,
    )

    labels = convert_to_tensor(
        df["labels"],
        dtype=tfbool,
        name="test_labels",
    )
    
    data = convert_to_tensor(
        df[["X1", "X2"]],
        dtype=_dtype,
        name="test_data",
    )

    dataset = Dataset.from_tensors(
        (data, labels),
        name="test_dataset",
    )
    return dataset
