"""Export a SynthSeg segmentation U-Net from Keras .h5 to ONNX.

Rebuilds the architecture exactly as SynthSeg/predict_synthseg.py:build_model() does
for the non-robust path, loads the published weights by name, and converts with
tf2onnx. Spatial dims are left dynamic so one graph serves any padded input size;
tract specialises per call.
"""
import os, sys, argparse
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

p = argparse.ArgumentParser()
p.add_argument('--synthseg-dir', default='SynthSeg',
               help='checkout of https://github.com/BBillot/SynthSeg')
p.add_argument('--version', choices=['1.0', '2.0'], default='1.0')
p.add_argument('--weights', default=None)
p.add_argument('--out', default=None)
p.add_argument('--size', type=int, default=0, help='fixed cubic spatial size; 0 = dynamic')
a = p.parse_args()

SS = os.path.abspath(a.synthseg_dir)
sys.path.insert(0, SS)
import numpy as np
import tensorflow as tf
import tf2onnx
from ext.neuron import models as nrn_models

labels_file = ('synthseg_segmentation_labels.npy' if a.version == '1.0'
               else 'synthseg_segmentation_labels_2.0.npy')
labels = np.unique(np.load(os.path.join(SS, 'data/labels_classes_priors', labels_file)))
weights = a.weights or os.path.join(SS, 'models', f'synthseg_{a.version}.h5')
out = a.out or os.path.abspath(f'synthseg_{a.version}.onnx')

print(f'labels: {len(labels)}  weights: {weights}')

dim = a.size if a.size else None
net = nrn_models.unet(input_shape=[dim, dim, dim, 1],
                      nb_labels=len(labels),
                      nb_levels=5,
                      nb_conv_per_level=2,
                      conv_size=3,
                      nb_features=24,
                      feat_mult=2,
                      activation='elu',
                      batch_norm=-1,
                      name='unet')
net.load_weights(weights, by_name=True)
print('params:', net.count_params())

spec = (tf.TensorSpec((1, dim, dim, dim, 1), tf.float32, name='image'),)
tf2onnx.convert.from_keras(net, input_signature=spec, opset=14, output_path=out)
print('wrote', out, os.path.getsize(out) // 1024 // 1024, 'MB')
np.save(out.replace('.onnx', '_labels.npy'), labels)
