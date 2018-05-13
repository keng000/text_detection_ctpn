# coding: utf-8
from __future__ import print_function

import os
import tensorflow as tf

from text_detection_ctpn.lib.networks.factory import get_network
from text_detection_ctpn.lib.fast_rcnn.config import cfg, cfg_from_file

config_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'text.yml')
cfg_from_file(config_file_path)

def load_trained_model(sess):
    # load network

    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()

    check_point_path = cfg.TEST.checkpoints_path
    try:
        ckpt = tf.train.get_checkpoint_state(check_point_path)
        print('Restoring from {}...'.format(check_point_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise Exception('Check your pretrained {}'.format(check_point_path))

    return net


def create_tf_session():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    sess = tf.Session(config=config)
    return sess