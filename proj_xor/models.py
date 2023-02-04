import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras import Model


class ProjXORModel(Model):
    def __init__(self):
        super(ProjXORModel, self).__init__()

        self._layers = [
            Dense(2, activation="relu"),
            Dense(1, activation="relu"),
        ]

    def call(self, x):
        for f in self._layers:
            x = f(x)
        return x
