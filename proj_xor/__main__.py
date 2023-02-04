import logging
from proj_xor.data import get_data

def main(logging_level=logging.WARNING):
    logging.basicConfig(level=logging_level)
    logging.info("Entering main method.")

    print(get_data.train_data())
    print(get_data.test_data())
    
    logging.info("Exiting main method.")

if __name__ == "__main__":
    main(logging_level=logging.INFO)
