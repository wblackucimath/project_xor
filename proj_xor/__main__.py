import logging
from proj_xor.data import get_data
from proj_xor.models import ProjXORModel
import proj_xor.utils


def main(logging_level=logging.WARNING):
    logging.basicConfig(level=logging_level)
    logging.info("Entering main method.")

    EPOCHS = 100

    model = ProjXORModel()
    for epoch in range(EPOCHS):
        model.train_loss.reset_states()
        model.train_accuracy.reset_states()
        model.test_loss.reset_states()
        model.test_accuracy.reset_states()

        for data, labels in get_data.train_data():
            model.train_step(data, labels)

        for data, labels in get_data.test_data():
            model.test_step(data, labels)

    logging.info("Exiting main method.")


if __name__ == "__main__":
    main(logging_level=logging.INFO)
