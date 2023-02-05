from tensorflow.keras import utils as tf_utils
from tensorflow import convert_to_tensor
from importlib_resources import files
from tensorflow.data import Dataset
from pandas import read_csv
from numpy import uint8, float32

_dataset_path = files("proj_xor.data.datasets")
_dtype = float32
# _int_dtype = 

def train_data():
    df = read_csv(
        files("proj_xor.data.datasets").joinpath("training_data.txt"),
        sep=" ",
        names=["labels", "X1", "X2"],
        dtype={"labels": uint8, "X1": _dtype, "X2": _dtype},
        skipinitialspace=1,
    )

    labels = convert_to_tensor(
        df["labels"],
        dtype=uint8,
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
        files("proj_xor.data.datasets").joinpath("test_data.txt"),
        sep=" ",
        names=["labels", "X1", "X2"],
        dtype={"labels": uint8, "X1": _dtype, "X2": _dtype},
        skipinitialspace=1,
    )

    labels = convert_to_tensor(
        df["labels"],
        dtype=uint8,
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
