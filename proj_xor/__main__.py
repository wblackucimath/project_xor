import logging
from proj_xor.data import get_data
from proj_xor.models import ProjXORWrapper
from proj_xor.plots.gen_plots import plot_loss, plot_accuracy

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sys import getsizeof

def main(logging_level=logging.WARNING):
    logging.basicConfig(level=logging_level)
    logging.info("Entering main method.")

    M = ProjXORWrapper(epochs=50 )

    M.fit(
        save_plots=True,
        show_plots=True,
        save_dfs=True,
        show_dfs=True,
        monitor_freq=5,
    )

    logging.info("Exiting main method.")


if __name__ == "__main__":
    main(logging_level=logging.INFO)
