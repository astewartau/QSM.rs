"""Dump a QSMnet checkpoint's weights and a TensorFlow reference output, for export_qsmnet.py.

Runs inside QSM-CI's original image (TensorFlow 1.14, SNU-LIST/QSMnet f8782ed, the checkpoint baked
in), which sets QSMNET_CODE, QSMNET_CKPT_DIR and QSMNET_NAME:

    podman run --rm --network none -v $PWD:/s:ro -v $OUT:/out \
        ghcr.io/astewartau/qsm-ci/qsmnet@sha256:7081b078... python /s/dump_qsmnet_tf.py

The activation comes from the checkpoint's own network_info_<name>.npy, as the original's inference
does: ['relu', 'qsmnet_deep'] for QSMnet_64, ['leaky_relu', 'qsmnet_deep'] for QSMnet+_64. (An
earlier export hard-coded 'leaky_relu' for both and checked itself against a TF graph built the same
way, so QSMnet_64's wrong activation went unnoticed: QSM.rs#330.)

Writes to /out: qn_weights.npz (trained variables, no optimiser slots), qn_in.npy (a fixed NDHWC
input), qn_tfout.npy (the TF graph's output for it) and qn_act.txt (the activation).
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.environ["QSMNET_CODE"])
os.chdir("/tmp")  # training_params.py makes ../Checkpoints/... at import time


import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import network_model  # noqa: E402

net_dir, net_name = os.environ["QSMNET_CKPT_DIR"], os.environ["QSMNET_NAME"]
epoch = os.environ.get("QSMNET_EPOCH", "25")
act_func, net_model = (str(v) for v in np.load(os.path.join(net_dir, f"network_info_{net_name}.npy")))

x = np.random.RandomState(0).randn(1, 32, 32, 32, 1).astype(np.float32)
tf.compat.v1.reset_default_graph()
Z = tf.compat.v1.placeholder("float", [None, None, None, None, 1])
pred = getattr(network_model, net_model)(Z, act_func, False, False)
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    tf.compat.v1.train.Saver().restore(sess, os.path.join(net_dir, f"{net_name}-{epoch}"))
    out = sess.run(pred, feed_dict={Z: x})

r = tf.train.load_checkpoint(os.path.join(net_dir, f"{net_name}-{epoch}"))
W = {k: r.get_tensor(k) for k in r.get_variable_to_shape_map()
     if "RMSProp" not in k and "global_step" not in k}
np.savez("/out/qn_weights.npz", **W)
np.save("/out/qn_in.npy", x)
np.save("/out/qn_tfout.npy", out)
open("/out/qn_act.txt", "w").write(act_func)
print(f"{net_name}: {net_model}, {act_func}, {len(W)} variables, TF output {out.shape}")
