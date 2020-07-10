"""Hyper-Parameters and Configuration"""

import sys

from absl import flags

# ---------- Pre-Processing ----------
# --train_path "" --val_path "" --test_path "" --meta_path ""  --epochs 1 --batch_size 32 --output_dir ""

# Paths
flags.DEFINE_string("train_path", "./data/5PPIsemi-snips/train.pkl", "")
flags.DEFINE_string("val_path", "./data/5PPIsemi-snips/eval.pkl", "")
flags.DEFINE_string("test_path", "./data/5PPIsemi-snips/test.pkl", "")
flags.DEFINE_string("meta_path", "./data/5PPIsemi-snips/metadata.json", "")

flags.DEFINE_string("output_dir", "./model_save/", "")
flags.DEFINE_bool("atis", False, "")
flags.DEFINE_integer("batch_size", 16, "")
flags.DEFINE_integer("epochs", 40, "")

params = flags.FLAGS
params(sys.argv)