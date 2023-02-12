import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense, Discretization
from tensorflow.keras import Model

from tensorflow.keras.losses import BinaryCrossentropy as BinaryCrossentropyLoss
from tensorflow.keras.metrics import (
    BinaryCrossentropy as BinaryCrossentropyMetric,
    BinaryAccuracy,
)
from tensorflow.keras.optimizers.experimental import Nadam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from tensorflow.math import argmax
from numpy import uint8
import pandas as pd
from importlib_resources import files

from tqdm.auto import trange, tqdm

from proj_xor.data import get_data
from proj_xor.plots.plot_metrics import plot_loss, plot_accuracy


class ProjXORModel(Model):
    def __init__(
        self,
        loss=None,
        optimizer=None,
        optimizer_schedule=None,
        train_loss_metric=None,
        test_loss_metric=None,
        train_acc_metric=None,
        test_acc_metric=None,
        layers=None,
    ):
        super(ProjXORModel, self).__init__()

        if layers is None:
            self._layers = [
                Dense(2, activation="swish"),
                Dense(1, activation="sigmoid"),
                # Discretization(
                #     bin_boundaries=[0.5],
                #     output_mode="int",
                #     dtype=float
                # ),
            ]
        else:
            self._layers = layers

        if loss is None:
            self.loss_object = BinaryCrossentropyLoss(from_logits=False)
        else:
            self.loss_object = loss

        if optimizer_schedule is None:
            self._optimizer_schedule = ExponentialDecay(
                2,
                decay_steps=10 ** 2,
                decay_rate=0.8,
                name="exponential_decay_schedule",
            )
        else:
            self._optimizer_schedule = optimizer_schedule
        if optimizer is None:
            self._optimizer = Nadam(
                learning_rate=self._optimizer_schedule,
                name="nadam_optimizer",
            )
        else:
            self._optimizer = optimizer

        if train_loss_metric is None:
            self.train_loss = BinaryCrossentropyMetric(name="train_loss")
        else:
            self.train_loss = train_loss_metric
        if train_acc_metric is None:
            self.train_accuracy = BinaryAccuracy(name="train_accuracy")
        else:
            self.train_accuracy = train_acc_metric

        if test_loss_metric is None:
            self.test_loss = BinaryCrossentropyMetric(name="test_loss")
        else:
            self.test_loss = test_loss_metric
        if test_acc_metric is None:
            self.test_accuracy = BinaryAccuracy(name="test_accuracy")
        else:
            self.test_accuracy = test_acc_metric

    def call(self, x):
        for f in self._layers:
            x = f(x)
        return x


class ProjXORWrapper:
    def __init__(
        self,
        loss=None,
        optimizer=None,
        optimizer_schedule=None,
        train_loss_metric=None,
        test_loss_metric=None,
        train_acc_metric=None,
        test_acc_metric=None,
        layers=None,
        epochs=60,
        batch_size=50,
    ):
        ## I know this function has a lot of repeated code, but fixing it is not a priority
        self._is_fitted = False

        self._epochs = epochs
        self._batch_size = batch_size

        if layers is None:
            self._layers = [
                Dense(2, activation="swish"),
                # swish works much better than relu or sigmoid
                Dense(1, activation="sigmoid"),
            ]
        else:
            self._layers = layers

        if loss is None:
            self.loss_object = BinaryCrossentropyLoss(from_logits=False)
        else:
            self.loss_object = loss

        if optimizer_schedule is None:
            self._optimizer_schedule = ExponentialDecay(
                0.5,
                decay_steps=10 ** 2,
                decay_rate=0.9,
                name="exponential_decay_schedule",
            )
        else:
            self._optimizer_schedule = optimizer_schedule
        if optimizer is None:
            self._optimizer = Nadam(
                learning_rate=self._optimizer_schedule,
                name="nadam_optimizer",
            )
        else:
            self._optimizer = optimizer

        if train_loss_metric is None:
            self.train_loss = BinaryCrossentropyMetric(name="train_loss")
        else:
            self.train_loss = train_loss_metric
        if train_acc_metric is None:
            self.train_accuracy = BinaryAccuracy(name="train_accuracy")
        else:
            self.train_accuracy = train_acc_metric

        if test_loss_metric is None:
            self.test_loss = BinaryCrossentropyMetric(name="test_loss")
        else:
            self.test_loss = test_loss_metric
        if test_acc_metric is None:
            self.test_accuracy = BinaryAccuracy(name="test_accuracy")
        else:
            self.test_accuracy = test_acc_metric

        self._model = ProjXORModel(
            loss=self.loss_object,
            optimizer=self._optimizer,
            optimizer_schedule=self._optimizer_schedule,
            train_loss_metric=self.train_loss,
            test_loss_metric=self.test_loss,
            train_acc_metric=self.train_accuracy,
            test_acc_metric=self.test_accuracy,
            layers=self._layers,
        )

        self.train_ds = get_data.train_data()
        self.test_ds = get_data.test_data()

    @tf.function
    def train_step(self, data, labels):
        with tf.GradientTape() as tape:
            predictions = self._model(data, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))

        los = self.train_loss(labels, predictions)
        acc = self.train_accuracy(labels, predictions)

    @tf.function
    def test_step(self, data, labels):
        predictions = self._model(data, training=False)
        t_loss = self.loss_object(labels, predictions)

        los = self.test_loss(labels, predictions)
        acc = self.test_accuracy(labels, predictions)

    def fit(
        self,
        save_dfs=False,
        show_dfs=False,
        save_plots=False,
        show_plots=False,
        monitor=True,
        monitor_freq=1,
    ):
        EPOCHS = self._epochs

        metric_df_cols = ["epoch", "run", "val"]
        loss_df = pd.DataFrame(columns=metric_df_cols)
        acc_df = pd.DataFrame(columns=metric_df_cols)

        for epoch in trange(EPOCHS):

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

            for data, labels in self.train_ds.shuffle(self._batch_size).batch(
                self._batch_size
            ):
                self.train_step(data, labels)

            for data, labels in self.test_ds.shuffle(self._batch_size).batch(
                self._batch_size
            ):
                self.test_step(data, labels)

            temp_loss_df = pd.DataFrame(
                [
                    [epoch, "train", float(self.train_loss.result())],
                    [epoch, "test", float(self.test_loss.result())],
                ],
                columns=metric_df_cols,
            )

            temp_acc_df = pd.DataFrame(
                [
                    [epoch, "train", float(self.train_accuracy.result())],
                    [epoch, "test", float(self.test_accuracy.result())],
                ],
                columns=metric_df_cols,
            )

            loss_df = pd.concat([loss_df, temp_loss_df], ignore_index=True)
            acc_df = pd.concat([acc_df, temp_acc_df], ignore_index=True)

            if monitor:
                if epoch % monitor_freq == 0:
                    tqdm.write(
                        f"Epoch {epoch:3.0f}\t"
                        f"Train Loss: {self.train_loss.result():.8f}\t"
                        f"Train Accuracy: {self.train_accuracy.result() * 100:.8f}\t"
                        f"Test Loss: {self.test_loss.result():.8f}\t"
                        f"Test Accuracy: {self.test_accuracy.result() * 100:.8f}"
                    )

        loss_df = loss_df.sort_values(["run", "epoch"], ignore_index=True)
        acc_df = acc_df.sort_values(["run", "epoch"], ignore_index=True)
        if save_dfs:
            loss_df.to_csv(files("proj_xor.data.outputs").joinpath("loss.csv"))
            acc_df.to_csv(files("proj_xor.data.outputs").joinpath("accuracy.csv"))
        if show_dfs:
            print(loss_df)
            print(acc_df)
        plot_loss(loss_df, save_plt=save_plots, show_plt=show_plots)
        plot_accuracy(acc_df, save_plt=save_plots, show_plt=show_plots)

        self._is_fitted = True

    def predict(self, X):
        if self._is_fitted:
            return self._model(X)

    def get_model(self):
        return self._model
