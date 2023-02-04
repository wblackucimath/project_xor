from tensorflow.keras.losses import BinaryCrossentropy as BinaryCrossentropyLoss
from tensorflow.keras.metrics import (
    BinaryCrossentropy as BinaryCrossentropyMetric,
    BinaryAccuracy,
)
from tensorflow.keras.optimizers.experimental import Nadam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import tensorflow as tf

_loss_dtype = BinaryCrossentropyLoss
_loss_object = _loss_dtype(from_logits=True)

_optimizer_schedule = ExponentialDecay(
    0.1,
    decay_steps=10 ** 5,
    decay_rate=0.96,
    name="exponential_decay_schedule",
)
_optimizer = Nadam(
    learning_rate=_optimizer_schedule,
    name="nadam_optimizer",
)

_train_test_loss_dtype = BinaryCrossentropyMetric
_train_test_accuracy_dtype = BinaryAccuracy

_train_loss = _train_test_loss_dtype(name="train_loss")
_train_accuracy = _train_test_accuracy_dtype(name="train_accuracy")

_test_loss = _train_test_loss_dtype(name="test_loss")
_test_accuracy = _train_test_accuracy_dtype(name="test_accuracy")


@tf.function
def train_step(data, labels, ret_loss=False):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    los = train_loss(loss)
    acc = train_accuracy(labels, predictions)

    if ret_loss:
        return {"loss": los, "accuracy": acc}


@tf.function
def test_step(data, labels, ret_loss=False):
    predictions = model(data, training=False)
    t_loss = loss_object(labels, predictions)

    los = test_loss(t_loss)
    acc = test_accuracy(labels, predictions)

    if ret_loss:
        return {"loss": los, "accuracy": acc}
