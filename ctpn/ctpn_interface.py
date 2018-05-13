# coding: utf-8
from __future__ import print_function

import cv2
import os
import numpy as np
import sys

from . import tf_utils
from text_detection_ctpn.lib.fast_rcnn.test import test_ctpn
from text_detection_ctpn.lib.text_connector.detectors import TextDetector
from text_detection_ctpn.lib.text_connector.text_connect_cfg import Config as TextLineCfg


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


def format_boxes(boxes, scale=1.0):
    """
    :param boxes:
        a list which include the coordination of each boundary boxes.
        this variable is output of TextDetector.detect() and the format is
            [left_top_x, left_top_y, right_top_x, right_top_y, left_bottom_x, left_bottom_y, right_bottom_x, right_bottom_y, score]
    :return:
    """
    coord_list = []
    for idx, box in enumerate(boxes):
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
            continue

        left = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
        top = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
        right = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
        bottom = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))

        coord_list.append((left, top, right, bottom))

    return coord_list


def ctpn(sess, net, in_img):
    """
    :param sess: tensorflow session. use tf_utils.create_session()
    :param net: tensorflow graph. use tf_utils.load_trained_graph()
    :param in_img: (numpy.ndarray) input image of 3 channels for detect text.
    :return: (list) a list of tuple which include each coordination of left, top, right, bottom of text-box-boundary.
    """
    img, scale = resize_im(in_img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    scores, boxes = test_ctpn(sess, net, img)

    text_detector = TextDetector()
    boxes = text_detector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    reformatted_boxes = format_boxes(boxes, scale)
    return reformatted_boxes


if __name__ == '__main__':

    # pass a path of image to argument
    img_path = sys.argv[1]

    # Execution Example
    with tf_utils.create_tf_session() as sess:
        net = tf_utils.load_trained_model(sess)

        # Main processing
        in_img = cv2.imread(img_path)
        coord_list = ctpn(sess, net, in_img)
        print(coord_list)
        
