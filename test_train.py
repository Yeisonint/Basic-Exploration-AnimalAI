# Inhabilita las alertas
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.disable_v2_behavior()

from baselines.common.models import get_network_builder
from baselines.deepq.models import build_q_func


network = build_q_func("cnn", dueling=True)