import logging
import sys


logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(filename="train.log"),
        logging.StreamHandler(stream=sys.stdout),
    ],
)
