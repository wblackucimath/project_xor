import seaborn as sns
import matplotlib.pyplot as plt
from importlib_resources import files

_plot_path = files("proj_xor.plots.metrics")


def plot_loss(loss_df, save_plt=False, show_plt=False, fname=None):
    if not (save_plt or show_plt):
        return
    loss_df["run"] = loss_df["run"].replace(["test"], "Test")
    loss_df["run"] = loss_df["run"].replace(["train"], "Train")
    loss_df = loss_df.set_axis(["Epoch", "Run", "Loss"], axis='columns')

    plt.figure(figsize=(16, 9))
    plot = sns.lineplot(
        data=loss_df,
        x="Epoch",
        y="Loss",
        hue="Run",
    )

    plt.title("Training and Test Loss vs. Epoch")
    plt.grid()
    plt.tight_layout()

    if save_plt:
        if fname is None:
            fname = "loss_plot.png"
        plt.savefig(_plot_path.joinpath(fname))
    if show_plt:
        plt.show()


def plot_accuracy(acc_df, save_plt=False, show_plt=False, fname=None):
    if not (save_plt or show_plt):
        return
    acc_df["run"] = acc_df["run"].replace(["test"], "Test")
    acc_df["run"] = acc_df["run"].replace(["train"], "Train")
    acc_df = acc_df.set_axis(["Epoch", "Run", "Accuracy"], axis='columns')

    plt.figure(figsize=(16, 9))
    plot = sns.lineplot(
        data=acc_df,
        x="Epoch",
        y="Accuracy",
        hue="Run",
    )

    plt.title("Training and Test Accuracy vs. Epoch")
    plt.grid()
    plt.tight_layout()

    if save_plt:
        if fname is None:
            fname = "accuracy_plot.png"
        plt.savefig(_plot_path.joinpath(fname))
    if show_plt:
        plt.show()