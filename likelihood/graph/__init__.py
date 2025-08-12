import tensorflow as tf
from packaging import version

from .graph import *

if version.parse(tf.__version__) > version.parse("2.15.0"):
    from ._nn import *
else:
    from .nn import *
