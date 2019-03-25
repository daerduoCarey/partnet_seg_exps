import argparse
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, '../utils'))
import tf_util
import json
from commons import check_mkdir, force_mkdir
from geometry_utils import *
from progressbar import ProgressBar
from subprocess import call

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', type=str, default='model', help='Model name [default: model]')
parser.add_argument('--category', type=str, default='Chair', help='Category name [default: Chair]')
parser.add_argument('--level_id', type=int, default='3', help='Level ID [default: 3]')
parser.add_argument('--num_ins', type=int, default='200', help='Max Number of Instance [default: 200]')
parser.add_argument('--log_dir', type=str, default='log', help='Log dir [default: log]')
parser.add_argument('--valid_dir', type=str, default='valid', help='Valid dir [default: valid]')
parser.add_argument('--num_point', type=int, default=10000, help='Point Number [default: 10000]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--margin_same', type=float, default=1.0, help='Double hinge loss margin: same semantic [default: 1]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
CKPT_DIR = os.path.join(LOG_DIR, 'trained_models')
if not os.path.exists(LOG_DIR):
    print('ERROR: log_dir %s does not exist! Please Check!' % LOG_DIR)
    exit(1)
LOG_DIR = os.path.join(LOG_DIR, FLAGS.valid_dir)
check_mkdir(LOG_DIR)

# load meta data files
stat_in_fn = '../../stats/after_merging_label_ids/%s-level-%d.txt' % (FLAGS.category, FLAGS.level_id)
print('Reading from ', stat_in_fn)
with open(stat_in_fn, 'r') as fin:
    part_name_list = [item.rstrip().split()[1] for item in fin.readlines()]
print('Part Name List: ', part_name_list)
data_in_dir = '../../data/ins_seg_h5_for_sgpn/%s-%d/' % (FLAGS.category, FLAGS.level_id)
val_h5_fn_list = []
for item in os.listdir(data_in_dir):
    if item.endswith('.h5') and item.startswith('val-'):
        val_h5_fn_list.append(item)

NUM_CLASSES = len(part_name_list)
print('Semantic Labels: ', NUM_CLASSES)
NUM_INS = FLAGS.num_ins
print('Number of Instances: ', NUM_INS)

def load_data(fn):
    out = h5py.File(fn)
    pts = out['pts'][:, :NUM_POINT, :]
    semseg_one_hot = out['semseg_one_hot'][:, :NUM_POINT, :]
    semseg_mask = out['semseg_mask'][:, :NUM_POINT]
    insseg_one_hot = out['insseg_one_hot'][:, :NUM_POINT, :]
    insseg_mask = out['insseg_mask'][:, :NUM_POINT]
    out.close()
    return pts, semseg_one_hot, semseg_mask, insseg_one_hot, insseg_mask

# Adapted from https://github.com/laughtervv/SGPN/blob/master/utils/test_utils.py#L11-L92
def Get_Ths(pts_corr, seg, ins, ths, ths_, cnt):

    pts_in_ins = {}
    for ip, pt in enumerate(pts_corr):
        if ins[ip] in pts_in_ins.keys():
            pts_in_curins_ind = pts_in_ins[ins[ip]]
            pts_notin_curins_ind = (~(pts_in_ins[ins[ip]])) & (seg==seg[ip])
            hist, bin = np.histogram(pt[pts_in_curins_ind], bins=20)

            numpt_in_curins = np.sum(pts_in_curins_ind)
            numpt_notin_curins = np.sum(pts_notin_curins_ind)

            if numpt_notin_curins > 0:

                tp_over_fp = 0
                ib_opt = -2
                for ib, b in enumerate(bin):
                    if b == 0:
                        break
                    tp = float(np.sum(pt[pts_in_curins_ind] < bin[ib])) / float(numpt_in_curins)
                    fp = float(np.sum(pt[pts_notin_curins_ind] < bin[ib])) / float(numpt_notin_curins)

                    if tp <= 0.5:
                        continue

                    if fp == 0. and tp > 0.5:
                        ib_opt = ib
                        break

                    if tp/fp > tp_over_fp:
                        tp_over_fp = tp / fp
                        ib_opt = ib

                if tp_over_fp >  4.:
                    ths[seg[ip]] += bin[ib_opt]
                    ths_[seg[ip]] += bin[ib_opt]
                    cnt[seg[ip]] += 1

        else:
            pts_in_curins_ind = (ins == ins[ip])
            pts_in_ins[ins[ip]] = pts_in_curins_ind
            pts_notin_curins_ind = (~(pts_in_ins[ins[ip]])) & (seg==seg[ip])
            hist, bin = np.histogram(pt[pts_in_curins_ind], bins=20)

            numpt_in_curins = np.sum(pts_in_curins_ind)
            numpt_notin_curins = np.sum(pts_notin_curins_ind)

            if numpt_notin_curins > 0:

                tp_over_fp = 0
                ib_opt = -2
                for ib, b in enumerate(bin):

                    if b == 0:
                        break

                    tp = float(np.sum(pt[pts_in_curins_ind]<bin[ib])) / float(numpt_in_curins)
                    fp = float(np.sum(pt[pts_notin_curins_ind]<bin[ib])) / float(numpt_notin_curins)

                    if tp <= 0.5:
                        continue

                    if fp == 0. and tp > 0.5:
                        ib_opt = ib
                        break

                    if tp / fp > tp_over_fp:
                        tp_over_fp = tp / fp
                        ib_opt = ib

                if tp_over_fp >  4.:
                    ths[seg[ip]] += bin[ib_opt]
                    ths_[seg[ip]] += bin[ib_opt]
                    cnt[seg[ip]] += 1

    return ths, ths_, cnt

def valid():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_ph, ptsseglabel_ph, ptsgroup_label_ph, _, _, _ = \
                    MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_INS, NUM_CLASSES)   # B x N x 3
            is_training_ph = tf.placeholder(tf.bool, shape=())

            group_mat_label = tf.matmul(ptsgroup_label_ph, tf.transpose(ptsgroup_label_ph, perm=[0, 2, 1]))
            net_output = MODEL.get_model(pointclouds_ph, NUM_CLASSES, FLAGS.margin_same, is_training_ph)

            loader = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Load pretrained model
        ckptstate = tf.train.get_checkpoint_state(CKPT_DIR)
        if ckptstate is not None:
            LOAD_MODEL_FILE = os.path.join(CKPT_DIR, os.path.basename(ckptstate.model_checkpoint_path))
            loader.restore(sess, LOAD_MODEL_FILE)
            print("Model loaded in file: %s" % LOAD_MODEL_FILE)
        else:
            print("Fail to load modelfile: %s" % CKPT_DIR)

        # Start to compute statistics on the validation set
        ths = np.zeros(NUM_CLASSES)
        ths_ = np.zeros(NUM_CLASSES)
        cnt = np.zeros(NUM_CLASSES)
        avg_groupsize = np.zeros(NUM_CLASSES)
        avg_groupsize_cnt = np.zeros(NUM_CLASSES)

        batch_pts = np.zeros((BATCH_SIZE, NUM_POINT, 3), dtype=np.float32)

        for item in val_h5_fn_list:
            cur_h5_fn = os.path.join(data_in_dir, item)
            print('Reading data from ', cur_h5_fn)
            pts, semseg_one_hot, _, insseg_one_hot, _ = load_data(cur_h5_fn)

            n_shape = pts.shape[0]
            bar = ProgressBar()
            for j in bar(range(n_shape)):
                cur_pts = pts[j, ...]
                cur_semseg_one_hot = semseg_one_hot[j, ...]
                cur_insseg_one_hot = insseg_one_hot[j, ...]

                # compute the average groupsize per category
                ids = np.arange(NUM_INS)
                group_size = np.sum(cur_insseg_one_hot, axis=0)
                valid_insseg_ids = ids[group_size > 0]
                for k in valid_insseg_ids:
                    ids = np.arange(NUM_POINT)
                    point_id = ids[cur_insseg_one_hot[:, k]][0]
                    cur_group_sem_cat = np.argmax(cur_semseg_one_hot[point_id, :])
                    cur_group_size = group_size[k]
                    avg_groupsize[cur_group_sem_cat] += cur_group_size
                    avg_groupsize_cnt[cur_group_sem_cat] += 1

                # compute the point-wise similarity score threshold
                feed_dict = {pointclouds_ph: np.expand_dims(cur_pts, axis=0),
                             ptsseglabel_ph: np.expand_dims(cur_semseg_one_hot, axis=0),
                             ptsgroup_label_ph: np.expand_dims(cur_insseg_one_hot, axis=0),
                             is_training_ph: False}

                pts_corr_val0, pred_confidence_val0, ptsclassification_val0, pts_corr_label_val0 = \
                                        sess.run([net_output['simmat'],
                                                  net_output['conf'],
                                                  net_output['semseg'],
                                                  group_mat_label],
                                                  feed_dict=feed_dict)

                seg = -1 * np.ones((NUM_POINT), dtype=np.int32)
                ids = np.arange(NUM_POINT)
                valid_ids = ids[np.sum(cur_semseg_one_hot, axis=-1) > 0]
                seg[valid_ids] = np.argmax(cur_semseg_one_hot, axis=-1)[valid_ids]

                ins = -1 * np.ones((NUM_POINT), dtype=np.int32)
                ids = np.arange(NUM_POINT)
                valid_ids = ids[np.sum(cur_insseg_one_hot, axis=-1) > 0]
                ins[valid_ids] = np.argmax(cur_insseg_one_hot, axis=-1)[valid_ids]

                pts_corr_val = np.squeeze(pts_corr_val0[0])
                pred_confidence_val = np.squeeze(pred_confidence_val0[0])
                ptsclassification_val = np.argmax(np.squeeze(ptsclassification_val0[0]),axis=1)

                pts_corr_label_val = np.squeeze(1 - pts_corr_label_val0)

                ths, ths_, cnt = Get_Ths(pts_corr_val, seg, ins, ths, ths_, cnt)


        # compute overall statistics
        ths = [ths[i]/cnt[i] if cnt[i] != 0 else 0.2 for i in range(len(cnt))]
        np.savetxt(os.path.join(LOG_DIR, 'per_category_pointwise_similarity_threshold.txt'), ths)

        avg_groupsize = [int(float(avg_groupsize[i]) / avg_groupsize_cnt[i]) if avg_groupsize_cnt[i] != 0 else 0 for i in range(len(avg_groupsize))]
        np.savetxt(os.path.join(LOG_DIR, 'per_category_average_group_size.txt'), avg_groupsize)

# main
valid()

