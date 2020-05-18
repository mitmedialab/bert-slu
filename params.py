"""Hyper-Parameters and Configuration"""

import sys

from absl import flags

# ---------- Pre-Processing ----------

# Paths
flags.DEFINE_string("train_path", "", "")
flags.DEFINE_string("val_path", "", "")
flags.DEFINE_string("test_path", "", "")
flags.DEFINE_string("meta_path", "", "")
flags.DEFINE_string("output_dir", "", "")
flags.DEFINE_bool("atis", False, "")
flags.DEFINE_integer("batch_size", 0, "")
flags.DEFINE_integer("epochs", 0, "")

params = flags.FLAGS
params(sys.argv)