import tensorflow as tf
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
