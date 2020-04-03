import tensorflow as tf
import os
import numpy as np
from PIL import Image
import cv2
import random

def get_next_batch(batch_size, image_size):
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    allFiles = os.listdir(FLAGS.data_path)
    imgFiles = []

    for name in allFiles:
        imgList = os.listdir(os.path.join(FLAGS.data_path, name))
        for img in imgList:
            imgFiles.append(os.path.join(FLAGS.data_path, name, img))

    idx = np.random.permutation(len(imgFiles))
    idx = idx[0:batch_size]

    images = []
    labels = []

    for item in idx:
        img = Image.open(imgFiles[item])
        img = np.array(img.resize((image_size, image_size)))
        images.append(img)
        labels.append(int(imgFiles[item].split('/')[-2]))
        
    images = np.reshape(images, [-1, image_size, image_size, 3])
    labels = np.array(labels)
    labels = np.reshape(labels, [batch_size, ])

    return images.astype(np.float32), labels.astype(np.int64)
