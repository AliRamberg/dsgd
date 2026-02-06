import logging
import sys

# Create logger
logger = logging.getLogger("ddp")
logger.setLevel(logging.DEBUG)

# Create console handler with formatting
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter(
    fmt='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S'
)
handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(handler)
logger.propagate = False
