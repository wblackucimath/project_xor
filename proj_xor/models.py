import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

from tensorflow.keras.losses import BinaryCrossentropy as BinaryCrossentropyLoss
from tensorflow.keras.metrics import (
    BinaryCrossentropy as BinaryCrossentropyMetric,
    BinaryAccuracy,
)
from tensorflow.keras.optimizers.experimental import Nadam
from tensorflow.keras.optimizers.schedules import ExponentialDecay


class ProjXORModel(Model):
    def __init__(self):
        super(ProjXORModel, self).__init__()

        self._layers = [
            Dense(2, activation="relu"),
            Dense(1, activation="relu"),
        ]

        self._loss_dtype = BinaryCrossentropyLoss
        self.loss_object = self._loss_dtype(from_logits=True)

        self._optimizer_schedule = ExponentialDecay(
            0.1,
            decay_steps=10 ** 5,
            decay_rate=0.96,
            name="exponential_decay_schedule",
        )
        self._optimizer = Nadam(
            learning_rate=self._optimizer_schedule,
            name="nadam_optimizer",
        )

        self._train_test_loss_dtype = BinaryCrossentropyMetric
        self._train_test_accuracy_dtype = BinaryAccuracy

        self.train_loss = self._train_test_loss_dtype(name="train_loss")
        self.train_accuracy = self._train_test_accuracy_dtype(name="train_accuracy")

        self.test_loss = self._train_test_loss_dtype(name="test_loss")
        self.test_accuracy = self._train_test_accuracy_dtype(name="test_accuracy")

    def call(self, x):
        for f in self._layers:
            x = f(x)
        return x

    @tf.function
    def train_step(self, data, labels, ret_loss=False):
        with tf.GradientTape() as tape:
            predictions = self(data, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        los = self.train_loss(labels, predictions)
        acc = self.train_accuracy(labels, predictions)

        if ret_loss:
            return {"loss": los, "accuracy": acc}

    @tf.function
    def test_step(self, data, labels, ret_loss=False):
        predictions = self(data, training=False)
        t_loss = self.loss_object(labels, predictions)

        los = self.test_loss(labels, predictions)
        acc = self.test_accuracy(labels, predictions)

        if ret_loss:
            return {"loss": los, "accuracy": acc}
