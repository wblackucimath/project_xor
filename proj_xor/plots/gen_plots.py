import seaborn as sns
import matplotlib.pyplot as plt
from importlib_resources import files

_plot_path = files("proj_xor.plots")


def plot_loss(loss_df, save_plt=False, show_plt=False, fname=None):
    plt.figure(figsize=(16, 9))
    plot = sns.lineplot(
        data=loss_df,
        x="epoch",
        y="val",
        hue="run",
    )

    plt.title("Training and Test Loss vs. Epoch")
    plt.grid()
    plt.tight_layout()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    if save_plt:
        if fname is None:
            fname = "loss_plot.png"
        plt.savefig(_plot_path.joinpath(fname))
    if show_plt:
        plt.show()


def plot_accuracy(loss_df, save_plt=False, show_plt=False, fname=None):
    plt.figure(figsize=(16, 9))
    plot = sns.lineplot(
        data=loss_df,
        x="epoch",
        y="val",
        hue="run",
    )

    plt.title("Training and Test Accuracy vs. Epoch")
    plt.grid()
    plt.tight_layout()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    if save_plt:
        if fname is None:
            fname = "accuracy_plot.png"
        plt.savefig(_plot_path.joinpath(fname))
    if show_plt:
        plt.show()
