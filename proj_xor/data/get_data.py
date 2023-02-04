from tensorflow.keras import utils as tf_utils
from importlib.resources import files
import proj_xor.data
import pandas as pd
import numpy as np

_dataset_path = files("proj_xor.data.datasets")
_dtype = np.float32


def train_data():

    # dataset = get_file(origin=files("proj_xor.data.datasets").joinpath("training_data.txt").as_uri())
    # print(type(dataset))

    dataset = pd.read_csv(
        files("proj_xor.data.datasets").joinpath("training_data.txt"),
        sep=" ",
        names=["labels", "X1", "X2"],
        dtype={"labels": np.uint8, "X1": _dtype, "X2": _dtype},
        skipinitialspace=1,
    )

    return dataset


def test_data():

    # dataset = get_file(origin=files("proj_xor.data.datasets").joinpath("training_data.txt").as_uri())
    # print(type(dataset))

    dataset = pd.read_csv(
        files("proj_xor.data.datasets").joinpath("test_data.txt"),
        sep=" ",
        names=["labels", "X1", "X2"],
        dtype={"labels": np.uint8, "X1": _dtype, "X2": _dtype},
        skipinitialspace=1,
    )

    return dataset
