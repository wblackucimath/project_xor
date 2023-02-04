import logging


def main(logging=logging.WARNING):
    logging.info("Entering main method.")
    logging.info("Exiting main method.")

if "__name__" == "__main__":
    main(logging=logging.INFO)
