from tensorflow.keras import utils as tf_utils
from importlib.resources import files
from pandas import read_csv
from numpy import uint8, float32

_dataset_path = files("proj_xor.data.datasets")
_dtype = float32


def train_data():
    dataset = read_csv(
        files("proj_xor.data.datasets").joinpath("training_data.txt"),
        sep=" ",
        names=["labels", "X1", "X2"],
        dtype={"labels": uint8, "X1": _dtype, "X2": _dtype},
        skipinitialspace=1,
    )

    return dataset


def test_data():
    dataset = read_csv(
        files("proj_xor.data.datasets").joinpath("test_data.txt"),
        sep=" ",
        names=["labels", "X1", "X2"],
        dtype={"labels": uint8, "X1": _dtype, "X2": _dtype},
        skipinitialspace=1,
    )

    return dataset
