"""
Input:      a 10k point cloud
Output:     200 disjoint masks, each for a part proposal
            per-point semantic label prediction
            (extra) per-part confidence scores, regress to the IoU between Prediction and Ground-truth
At test time, we combine the mask and the per-point semantics to determine the part semantics
"""

import tensorflow as tf
import numpy as np
import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../../../utils'))
import tf_util
from scipy.optimize import linear_sum_assignment
from pointnet_util import pointnet_sa_module, pointnet_fp_module

def placeholder_inputs(batch_size, num_point, num_ins):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    gt_mask_pl = tf.placeholder(tf.float32, shape=(batch_size, num_ins, num_point))
    gt_valid_pl = tf.placeholder(tf.float32, shape=(batch_size, num_ins))
    gt_other_mask_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, gt_mask_pl, gt_valid_pl, gt_other_mask_pl

def get_model(point_cloud, num_part, num_ins, is_training, weight_decay=0.0, bn_decay=None):
    end_points = {}

    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    l0_xyz = point_cloud
    l0_points = point_cloud

    # Set Abstraction layers
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=64, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')

    # Feature Propagation layers
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='fa_layer1')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer2')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, tf.concat([l0_xyz,l0_points],axis=-1), l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer3')

    # semantic segmentation branch
    seg_net = l0_points
    seg_net = tf_util.conv1d(seg_net, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='seg/fc1', bn_decay=bn_decay)
    seg_net = tf_util.conv1d(seg_net, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='seg/fc2', bn_decay=bn_decay)
    seg_net = tf_util.conv1d(seg_net, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='seg/fc3', bn_decay=bn_decay)
    # also predict the semantic part "other"
    seg_net = tf_util.conv1d(seg_net, num_part+1, 1, padding='VALID', bn=False, activation_fn=None, scope='seg/fc4', bn_decay=bn_decay)

    # instance segmentation branch
    ins_net = l0_points
    ins_net = tf_util.conv1d(ins_net, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='ins/fc1', bn_decay=bn_decay)
    ins_net = tf_util.conv1d(ins_net, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='ins/fc2', bn_decay=bn_decay)
    ins_net = tf_util.conv1d(ins_net, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='ins/fc3', bn_decay=bn_decay)
    # also predict the semantic part "other"
    ins_net = tf_util.conv1d(ins_net, num_ins+1, 1, padding='VALID', bn=False, activation_fn=None, scope='ins/fc4', bn_decay=bn_decay)
    ins_net = tf.nn.softmax(ins_net, dim=-1)   # B x N x (K+1)

    mask_pred = tf.transpose(ins_net[:, :, :-1], perm=[0, 2, 1])  # B x K x N
    other_mask_pred = ins_net[:, :, -1]  # B x N

    # part confidence score
    conf_net = tf.reshape(l3_points, [batch_size, -1])
    conf_net = tf_util.fully_connected(conf_net, 256, bn=True, is_training=is_training, scope='conf/fc1', bn_decay=bn_decay)
    conf_net = tf_util.fully_connected(conf_net, 256, bn=True, is_training=is_training, scope='conf/fc2', bn_decay=bn_decay)
    conf_net = tf_util.fully_connected(conf_net, num_ins, activation_fn=None, scope='conf/fc3')
    conf_net = tf.nn.sigmoid(conf_net)

    return seg_net, mask_pred, other_mask_pred, conf_net, end_points

def get_seg_loss(seg_pred, seg_gt, end_points):
    per_point_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=seg_gt)
    end_points['per_point_seg_loss'] = per_point_loss
    per_shape_loss = tf.reduce_mean(per_point_loss, axis=-1)
    end_points['per_shape_seg_loss'] = per_shape_loss
    loss = tf.reduce_mean(per_shape_loss)
    return loss, end_points

# copy from https://github.com/ericyi/articulated-part-induction/blob/master/model.py
def hungarian_matching(pred_x, gt_x, curnmasks):
    """ pred_x, gt_x: B x nmask x n_point
        curnmasks: B
        return matching_idx: B x nmask x 2 """
    batch_size = gt_x.shape[0]
    nmask = gt_x.shape[1]
    matching_score = np.matmul(gt_x, np.transpose(pred_x, axes=[0, 2, 1])) # B x nmask x nmask
    matching_score = 1 - np.divide(matching_score, np.maximum(np.expand_dims(np.sum(pred_x, 2), 1)+np.sum(gt_x, 2, keepdims=True) - matching_score, 1e-8))
    matching_idx = np.zeros((batch_size, nmask, 2)).astype('int32')
    curnmasks = curnmasks.astype('int32')
    for i, curnmask in enumerate(curnmasks):
        row_ind, col_ind = linear_sum_assignment(matching_score[i, :curnmask, :])
        matching_idx[i, :curnmask, 0] = row_ind
        matching_idx[i, :curnmask, 1] = col_ind
    return matching_idx

# copy from https://github.com/ericyi/articulated-part-induction/blob/master/model.py
def iou(pred_x, gt_x, gt_conf, n_point, nmask, end_points):
    matching_idx = tf.py_func(hungarian_matching, [pred_x, gt_x, tf.reduce_sum(gt_conf,-1)], tf.int32) # B x nmask x 2
    matching_idx = tf.stop_gradient(matching_idx)
    end_points['matching_idx'] = matching_idx

    matching_idx_row = matching_idx[:, :, 0]
    idx = tf.where(tf.greater_equal(matching_idx_row, 0))
    matching_idx_row = tf.concat((tf.expand_dims(tf.cast(idx[:, 0], tf.int32), -1), tf.reshape(matching_idx_row, [-1, 1])), 1)
    gt_x_matched = tf.reshape(tf.gather_nd(gt_x, matching_idx_row), [-1, nmask, n_point])
    
    matching_idx_column = matching_idx[:, :, 1]
    idx = tf.where(tf.greater_equal(matching_idx_column, 0))
    matching_idx_column = tf.concat((tf.expand_dims(tf.cast(idx[:, 0], tf.int32), -1),tf.reshape(matching_idx_column, [-1, 1])), 1)
    pred_x_matched = tf.reshape(tf.gather_nd(pred_x, matching_idx_column), [-1, nmask, n_point])
    
    # compute meaniou
    matching_score = tf.reduce_sum(tf.multiply(gt_x_matched, pred_x_matched),2)
    iou_all = tf.divide(matching_score, tf.reduce_sum(gt_x_matched, 2) + tf.reduce_sum(pred_x_matched, 2) - matching_score + 1e-8)
    end_points['per_shape_all_iou'] = iou_all
    
    meaniou = tf.divide(tf.reduce_sum(tf.multiply(iou_all, gt_conf), 1), tf.reduce_sum(gt_conf, -1) + 1e-8) # B
    return meaniou, end_points

def get_ins_loss(mask_pred, mask_gt, gt_valid, end_points):
    """ Input:  mask_pred   B x K x N
                mask_gt     B x K x N
                gt_valid    B x K
    """
    num_ins = mask_pred.get_shape()[1].value
    num_point = mask_pred.get_shape()[2].value
    meaniou, end_points = iou(mask_pred, mask_gt, gt_valid, num_point, num_ins, end_points)
    end_points['per_shape_mean_iou'] = meaniou
    loss = - tf.reduce_mean(meaniou)
    return loss, end_points

def get_conf_loss(conf_pred, gt_valid, end_points):
    """ Input:  conf_pred       B x K
                gt_valid        B x K
    """
    batch_size = conf_pred.get_shape()[0].value
    nmask = conf_pred.get_shape()[1].value

    iou_all = end_points['per_shape_all_iou']
    matching_idx = end_points['matching_idx']

    matching_idx_column = matching_idx[:, :, 1]
    idx = tf.where(tf.greater_equal(matching_idx_column, 0))
    all_indices = tf.concat((tf.expand_dims(tf.cast(idx[:, 0], tf.int32), -1), tf.reshape(matching_idx_column, [-1, 1])), 1)
    all_indices = tf.reshape(all_indices, [batch_size, nmask, 2])

    valid_idx = tf.where(tf.greater(gt_valid, 0.5))
    predicted_indices = tf.gather_nd(all_indices, valid_idx)
    valid_iou = tf.gather_nd(iou_all, valid_idx)

    conf_target = tf.scatter_nd(predicted_indices, valid_iou, tf.constant([batch_size, nmask]))
    end_points['per_part_conf_target'] = conf_target

    per_part_loss = tf.squared_difference(conf_pred, conf_target)
    end_points['per_part_loss'] = per_part_loss

    target_pos_mask = tf.cast(tf.greater(conf_target, 0.1), tf.float32)
    target_neg_mask = 1.0 - target_pos_mask
    
    pos_per_shape_loss = tf.divide(tf.reduce_sum(target_pos_mask * per_part_loss, axis=-1), tf.maximum(1e-6, tf.reduce_sum(target_pos_mask, axis=-1)))
    neg_per_shape_loss = tf.divide(tf.reduce_sum(target_neg_mask * per_part_loss, axis=-1), tf.maximum(1e-6, tf.reduce_sum(target_neg_mask, axis=-1)))

    per_shape_loss = pos_per_shape_loss + neg_per_shape_loss
    end_points['per_shape_loss'] = per_shape_loss

    loss = tf.reduce_mean(per_shape_loss)
    return loss, end_points

def get_other_ins_loss(other_mask_pred, gt_other_mask, end_points):
    """ Input:  other_mask_pred B x N
                gt_other_mask   B x N
    """
    batch_size = other_mask_pred.get_shape()[0].value
    num_point = other_mask_pred.get_shape()[1].value
    matching_score = tf.reduce_sum(tf.multiply(other_mask_pred, gt_other_mask), axis=-1)
    iou = tf.divide(matching_score, tf.reduce_sum(other_mask_pred, axis=-1) + tf.reduce_sum(gt_other_mask, axis=-1) - matching_score + 1e-8)
    end_points['per_shape_other_iou'] = iou
    loss = - tf.reduce_mean(iou)
    return loss, end_points

def get_l21_norm(mask_pred, other_mask_pred, end_points):
    """ Input:  mask_pred           B x K x N
                other_mask_pred     B x N
    """
    num_point = other_mask_pred.get_shape()[1].value

    full_mask = tf.concat([mask_pred, tf.expand_dims(other_mask_pred, axis=1)], axis=1) + 1e-6
    per_shape_l21_norm = tf.norm(tf.norm(full_mask, ord=2, axis=-1), ord=1, axis=-1) / num_point
    end_points['per_shape_l21_norm'] = per_shape_l21_norm

    loss = tf.reduce_mean(per_shape_l21_norm)
    return loss, end_points


