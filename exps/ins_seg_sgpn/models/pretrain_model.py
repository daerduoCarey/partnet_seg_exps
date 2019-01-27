import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
import tensorflow as tf
import numpy as np
import tf_util
import pointnet2


def placeholder_inputs(batch_size, num_point, num_cate):

    pointclouds_ph = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))

    pts_seglabels_ph = tf.placeholder(tf.int32, shape=(batch_size, num_point, num_cate))
    pts_seglabel_mask_ph = tf.placeholder(tf.float32, shape=(batch_size, num_point))

    return pointclouds_ph, pts_seglabels_ph, pts_seglabel_mask_ph


def get_model(point_cloud, cate_num, is_training, bn_decay=None):
    #input: point_cloud: BxNx3 (XYZ)

    batch_size = point_cloud.get_shape()[0].value
    print(point_cloud.get_shape())

    F = pointnet2.get_model(point_cloud, is_training, bn=True, bn_decay=bn_decay)

    # Semantic prediction
    Fsem = tf_util.conv2d(F, 128, [1, 1], padding='VALID', stride=[1, 1], bn=False, is_training=is_training, scope='Fsem')

    ptssemseg_logits = tf_util.conv2d(Fsem, cate_num, [1, 1], padding='VALID', stride=[1, 1], activation_fn=None, scope='ptssemseg_logits')
    ptssemseg_logits = tf.squeeze(ptssemseg_logits, [2])

    return ptssemseg_logits


def get_loss(pred, labels):
    """
    input:
        labels:{'semseg','semseg_mask'}
    """

    per_point_ptsseg_loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labels['semseg'])
    per_point_ptsseg_loss = tf.multiply(per_point_ptsseg_loss, labels['semseg_mask'])
    per_shape_ptsseg_loss = tf.divide(tf.reduce_sum(per_point_ptsseg_loss, axis=-1), tf.maximum(tf.reduce_sum(labels['semseg_mask'], axis=-1), 1e-6))
    loss = tf.reduce_mean(per_shape_ptsseg_loss)
    return loss

