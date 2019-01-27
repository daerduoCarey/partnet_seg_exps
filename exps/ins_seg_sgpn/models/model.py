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


def placeholder_inputs(batch_size, num_point, num_group, num_cate):

    pointclouds_ph = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))

    pts_seglabels_ph = tf.placeholder(tf.int32, shape=(batch_size, num_point, num_cate))
    pts_grouplabels_ph = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_group))
    pts_seglabel_mask_ph = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    pts_group_mask_ph = tf.placeholder(tf.float32, shape=(batch_size, num_point))

    alpha_ph = tf.placeholder(tf.float32, shape=())

    return pointclouds_ph, pts_seglabels_ph, pts_grouplabels_ph, pts_seglabel_mask_ph, pts_group_mask_ph, alpha_ph


def get_model(point_cloud, cate_num, margin0, is_training, bn_decay=None):
    #input: point_cloud: BxNx3 (XYZ)

    batch_size = point_cloud.get_shape()[0].value
    N = point_cloud.get_shape()[1].value
    print(point_cloud.get_shape())

    F = pointnet2.get_model(point_cloud, is_training, bn=True, bn_decay=bn_decay)

    # Semantic prediction
    Fsem = tf_util.conv2d(F, 128, [1, 1], padding='VALID', stride=[1, 1], bn=False, is_training=is_training, scope='Fsem')

    ptssemseg_logits = tf_util.conv2d(Fsem, cate_num, [1, 1], padding='VALID', stride=[1, 1], activation_fn=None, scope='ptssemseg_logits')
    ptssemseg_logits = tf.squeeze(ptssemseg_logits, [2])

    ptssemseg = tf.nn.softmax(ptssemseg_logits, name="ptssemseg")

    # Similarity matrix
    Fsim = tf_util.conv2d(F, 128, [1, 1], padding='VALID', stride=[1, 1], bn=False, is_training=is_training, scope='Fsim')

    Fsim = tf.squeeze(Fsim, [2])

    r = tf.reduce_sum(Fsim * Fsim, 2)
    r = tf.tile(tf.reshape(r, [batch_size, -1, 1]), [1, 1, N])
    D = r - 2 * tf.matmul(Fsim, tf.transpose(Fsim, perm=[0, 2, 1])) + tf.transpose(r, perm=[0, 2, 1])

    # simmat_logits = tf.maximum(D, 0.)
    simmat_logits = tf.maximum(margin0 * D, 0.)

    # Confidence Map
    Fconf = tf_util.conv2d(F, 128, [1, 1], padding='VALID', stride=[1, 1], bn=False, is_training=is_training, scope='Fsconf')
    conf_logits = tf_util.conv2d(Fconf, 1, [1, 1], padding='VALID', stride=[1, 1], activation_fn=None, scope='conf_logits')
    conf_logits = tf.squeeze(conf_logits, [2])

    conf = tf.nn.sigmoid(conf_logits, name="confidence")

    return {'semseg': ptssemseg,
            'semseg_logits': ptssemseg_logits,
            'simmat': simmat_logits,
            'conf': conf,
            'conf_logits': conf_logits}


def get_loss(net_output, labels, alpha, margin, allow_full_loss):
    """
    input:
        net_output:{'semseg', 'semseg_logits','simmat','conf','conf_logits'}
        labels:{'ptsgroup', 'semseg','semseg_mask','group_mask'}
    """

    pts_group_label = tf.cast(labels['ptsgroup'], tf.float32)
    pts_semseg_label = tf.cast(labels['semseg'], tf.float32)
    group_mask = tf.expand_dims(labels['group_mask'], dim=2)

    pred_confidence_logits = net_output['conf']
    pred_simmat = net_output['simmat']

    # Similarity Matrix loss
    B = pts_group_label.get_shape()[0]
    N = pts_group_label.get_shape()[1]

    onediag = tf.ones([B,N], tf.float32)

    group_mat_label = tf.matmul(pts_group_label,tf.transpose(pts_group_label, perm=[0, 2, 1])) #BxNxN: (i,j) if i and j in the same group
    group_mat_label = tf.matrix_set_diag(group_mat_label,onediag)

    sem_mat_label = tf.matmul(pts_semseg_label,tf.transpose(pts_semseg_label, perm=[0, 2, 1])) #BxNxN: (i,j) if i and j are the same semantic category
    sem_mat_label = tf.matrix_set_diag(sem_mat_label,onediag)

    samesem_mat_label = sem_mat_label
    diffsem_mat_label = tf.subtract(1.0, sem_mat_label)

    samegroup_mat_label = group_mat_label
    diffgroup_mat_label = tf.subtract(1.0, group_mat_label)
    diffgroup_samesem_mat_label = tf.multiply(diffgroup_mat_label, samesem_mat_label)
    diffgroup_diffsem_mat_label = tf.multiply(diffgroup_mat_label, diffsem_mat_label)

    num_samegroup = tf.reduce_sum(samegroup_mat_label)
    num_diffgroup_samesem = tf.reduce_sum(diffgroup_samesem_mat_label)
    num_diffgroup_diffsem = tf.reduce_sum(diffgroup_diffsem_mat_label)

    # Double hinge loss

    C_same = tf.constant(margin[0], name="C_same") # same semantic category
    C_diff = tf.constant(margin[1], name="C_diff") # different semantic category

    pos =  tf.multiply(samegroup_mat_label, pred_simmat) # minimize distances if in the same group
    neg_samesem = alpha * tf.multiply(diffgroup_samesem_mat_label, tf.maximum(tf.subtract(C_same, pred_simmat), 0))
    neg_diffsem = tf.multiply(diffgroup_diffsem_mat_label, tf.maximum(tf.subtract(C_diff, pred_simmat), 0))

    simmat_loss = neg_samesem + neg_diffsem + pos
    group_mask_weight = tf.matmul(group_mask, tf.transpose(group_mask, perm=[0, 2, 1]))
    # simmat_loss = tf.add(simmat_loss, pos)
    simmat_loss = tf.multiply(simmat_loss, group_mask_weight)
    simmat_loss = tf.divide(tf.reduce_sum(simmat_loss, axis=[1, 2]), tf.maximum(1e-6, tf.reduce_sum(group_mask_weight, axis=[1, 2])))
    simmat_loss = tf.reduce_mean(simmat_loss)

    # Semantic Segmentation loss
    ptsseg_loss = tf.nn.softmax_cross_entropy_with_logits(logits=net_output['semseg_logits'], labels=pts_semseg_label)
    ptsseg_loss = tf.multiply(ptsseg_loss, labels['semseg_mask'])
    ptsseg_loss = tf.divide(tf.reduce_sum(ptsseg_loss, axis=-1), tf.maximum(1e-6, tf.reduce_sum(labels['semseg_mask'], axis=-1)))
    ptsseg_loss = tf.reduce_mean(ptsseg_loss)

    # Confidence Map loss
    Pr_obj = tf.reduce_sum(pts_semseg_label, axis=-1)
    ng_label = group_mat_label
    ng_label = tf.greater(ng_label, tf.constant(0.5))
    ng = tf.less(pred_simmat, tf.constant(margin[0]))

    epsilon = tf.constant(np.ones(ng_label.get_shape()[:2]).astype(np.float32) * 1e-6)
    pts_iou = tf.div(tf.reduce_sum(tf.cast(tf.logical_and(ng,ng_label), tf.float32), axis=2),
                     (tf.reduce_sum(tf.cast(tf.logical_or(ng,ng_label), tf.float32), axis=2)+epsilon))
    confidence_label = tf.multiply(pts_iou, Pr_obj) # BxN

    confidence_loss = tf.reduce_mean(tf.squared_difference(confidence_label, tf.squeeze(pred_confidence_logits,[2])))

    loss = simmat_loss + allow_full_loss * ptsseg_loss + allow_full_loss * confidence_loss

    grouperr = tf.abs(tf.cast(ng, tf.float32) - tf.cast(ng_label, tf.float32))

    return loss, tf.reduce_mean(grouperr), \
           tf.reduce_sum(grouperr * diffgroup_samesem_mat_label), num_diffgroup_samesem, \
           tf.reduce_sum(grouperr * diffgroup_diffsem_mat_label), num_diffgroup_diffsem, \
           tf.reduce_sum(grouperr * samegroup_mat_label), num_samegroup

