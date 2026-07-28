import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from packaging import version

if version.parse(tf.__version__) > version.parse("2.15.0"):
    from ._autoencoders import *
    from ._predictor import GetInsights
else:
    from .autoencoders import *
    from .predictor import GetInsights

from .bandit import *
from .gan import *
from .rl import *
