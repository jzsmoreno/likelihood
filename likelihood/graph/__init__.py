import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from packaging import version

from .graph import *

if version.parse(tf.__version__) > version.parse("2.15.0"):
    from ._nn import *
else:
    from .nn import *
