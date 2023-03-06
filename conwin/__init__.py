import logging
import pathlib
import sys

fname = pathlib.Path(__file__).parents[1] / "train.log"
if fname.exists():
    fname.unlink()

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(filename=fname),
        logging.StreamHandler(stream=sys.stdout),
    ],
)
