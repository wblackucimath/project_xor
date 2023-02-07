import seaborn as sns
import matplotlib.pyplot as plt
from importlib_resources import files
import tensorflow as tf
import pandas as pd
import numpy as np
from proj_xor.data import get_data


_plot_path = files("proj_xor.plots.data")
_dataset_path = files("proj_xor.data.datasets")


def plot_model_performance(model, save_plt=True, show_plt=False, fname=None):
    for data, labels in get_data.train_data():
        labels = tf.cast(tf.reshape(labels, (-1, 1)), tf.dtypes.uint8)
        pred = tf.cast(model(data) > 0.5, tf.dtypes.uint8)

    df = pd.DataFrame(
        data={
            "x": data[:, 0],
            "y": data[:, 1],
            "Ground Truth": pd.Series(tf.reshape(labels, (-1,))),
            "Prediction": pd.Series(tf.reshape(pred, (-1,))),
            "Correct": pd.Series(tf.reshape(labels == pred, (-1,)), dtype=int),
        },
    )

    plt.figure(figsize=(16, 9))

    xlim = (-1, 2)
    ylim = (-1, 2)
    N_h_gridpoints = 10 * (xlim[1] - xlim[0])
    N_v_gridpoints = 10 * (ylim[1] - ylim[0])

    hmesh = np.linspace(*xlim, num=N_h_gridpoints)
    vmesh = np.linspace(*ylim, num=N_v_gridpoints)

    xgrid, ygrid = np.meshgrid(hmesh, vmesh)

    onehotx, onehoty = xgrid.reshape((-1, 1)), ygrid.reshape((-1, 1))

    onehotgrid = np.hstack((onehotx, onehoty))

    onehotgridpred = model(onehotgrid)

    gridpred = tf.reshape(onehotgridpred, xgrid.shape)

    cf = plt.contourf(
        xgrid,
        ygrid,
        gridpred,
        cmap="RdBu",
    )
    plt.colorbar(cf)

    c = plt.contour(
        xgrid,
        ygrid,
        gridpred,
        levels=[0.5],
    )

    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="Ground Truth",
        hue_order=[1, 0],
        style="Correct",
        style_order=[1, 0],
    )

    plt.title("Model Performance with Decision Boundary")
    plt.legend(loc="center right")
    plt.tight_layout()

    if save_plt:
        if fname is None:
            fname = "performance.png"
        plt.savefig(_plot_path.joinpath(fname))
    if show_plt:
        plt.show()

    plt.close()
