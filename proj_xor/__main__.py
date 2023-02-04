import logging

def main(logging_level=logging.WARNING):
    logging.basicConfig(level=logging_level)
    logging.info("Entering main method.")
    logging.info("Exiting main method.")

if __name__ == "__main__":
    main(logging_level=logging.INFO)
