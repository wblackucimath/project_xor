import logging
from proj_xor.data import get_data
from proj_xor.models import ProjXORWrapper
from proj_xor.plots.gen_plots import plot_loss, plot_accuracy

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


def main(logging_level=logging.WARNING):
    logging.basicConfig(level=logging_level)
    logging.info("Entering main method.")

    # EPOCHS = 50

    # metric_df_cols = ["epoch", "run", "val"]
    # loss_df = pd.DataFrame(columns=metric_df_cols)
    # acc_df = pd.DataFrame(columns=metric_df_cols)

    # model = ProjXORModel()
    # for epoch in trange(EPOCHS):

    #     model.train_loss.reset_states()
    #     model.train_accuracy.reset_states()
    #     model.test_loss.reset_states()
    #     model.test_accuracy.reset_states()

    #     for data, labels in get_data.train_data():
    #         model.train_step(data, labels)

    #     for data, labels in get_data.test_data():
    #         model.test_step(data, labels)

    #     temp_loss_df = pd.DataFrame(
    #         [
    #             [epoch, "train", float(model.train_loss.result())],
    #             [epoch, "test", float(model.test_loss.result())],
    #         ],
    #         columns=metric_df_cols,
    #     )

    #     temp_acc_df = pd.DataFrame(
    #         [
    #             [epoch, "train", float(model.train_accuracy.result())],
    #             [epoch, "test", float(model.test_accuracy.result())],
    #         ],
    #         columns=metric_df_cols,
    #     )

    #     loss_df = pd.concat([loss_df, temp_loss_df], ignore_index=True)
    #     acc_df = pd.concat([acc_df, temp_acc_df], ignore_index=True)

    #     tqdm.write(
    #         f"Epoch {epoch + 1:3.0f}\t"
    #         f"Train Loss: {model.train_loss.result():.8f}\t"
    #         f"Train Accuracy: {model.train_accuracy.result() * 100:.8f}\t"
    #         f"Test Loss: {model.test_loss.result():.8f}\t"
    #         f"Test Accuracy: {model.test_accuracy.result() * 100:.8f}"
    #     )

    # loss_df = loss_df.sort_values(["run", "epoch"], ignore_index=True)
    # acc_df = acc_df.sort_values(["run", "epoch"], ignore_index=True)
    # print(loss_df)
    # print(acc_df)
    # plot_loss(loss_df, save_plt=True, show_plt=False)
    # plot_accuracy(acc_df, save_plt=True, show_plt=False)
    M = ProjXORWrapper()
    M.fit(show_plots=True)
    logging.info("Exiting main method.")


if __name__ == "__main__":
    main(logging_level=logging.INFO)
