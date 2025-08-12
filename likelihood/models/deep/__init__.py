import tensorflow as tf
from packaging import version

if version.parse(tf.__version__) > version.parse("2.15.0"):
    from ._autoencoders import *
else:
    from .autoencoders import *

from .gan import *
from .predictor import GetInsights
