import tensorflow as tf
import numpy as np
import math
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module

def get_model(point_cloud, is_training, bn=True, bn_decay=None):
    end_points = {}

    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    l0_xyz = point_cloud
    l0_points = point_cloud

    # Set Abstraction layers
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=64, mlp=[64,64,128], mlp2=None, group_all=False, \
            bn=bn, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, \
            bn=bn, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, \
            bn=bn, is_training=is_training, bn_decay=bn_decay, scope='layer3')

    # Feature Propagation layers
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], \
            bn=bn, is_training=is_training, bn_decay=bn_decay, scope='fa_layer1')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], \
            bn=bn, is_training=is_training, bn_decay=bn_decay, scope='fa_layer2')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, tf.concat([l0_xyz,l0_points],axis=-1), l1_points, [128,128,128], \
            bn=bn, is_training=is_training, bn_decay=bn_decay, scope='fa_layer3')

    # semantic segmentation branch
    seg_net = l0_points
    seg_net = tf_util.conv1d(seg_net, 256, 1, padding='VALID', bn=bn, is_training=is_training, scope='seg/fc1', bn_decay=bn_decay)
    seg_net = tf_util.conv1d(seg_net, 256, 1, padding='VALID', bn=bn, is_training=is_training, scope='seg/fc2', bn_decay=bn_decay)
    seg_net = tf.expand_dims(seg_net, axis=2)
    print 'PointNet++ Output: ', seg_net
    return seg_net

