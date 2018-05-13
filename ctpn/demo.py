import cv2
import sys
import shutil
import glob
import time
import numpy as np

from text_detection_ctpn.ctpn import ctpn_interface, tf_utils
from text_detection_ctpn.lib.fast_rcnn.test import test_ctpn
from text_detection_ctpn.lib.fast_rcnn.config import cfg


if __name__ == '__main__':
    
    argv = sys.argv
    if len(argv) == 1:
        img_path = f"{cfg.DATA_DIR}/demo/010.png"
    else:
        img_path = argv[1]

    sess = tf_utils.create_tf_session()
    net = tf_utils.load_trained_model(sess)

    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in range(2):
        _, _ = test_ctpn(sess, net, im)
    
    start = time.time()
    img = cv2.imread(img_path)
    bbox_coordinations = ctpn_interface.ctpn(sess, net, img)
    print(f"DEMO: Detection took {time.time() - start: .3}s for {len(bbox_coordinations)} object proposals")
