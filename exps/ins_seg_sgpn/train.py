""" Finetune Similarity Matrix and Confidence Map
    Need to pre-train PointNet Semantic Segmentation Branch before
"""

import argparse
import tensorflow as tf
import numpy as np
import os
import sys
import importlib
import random
import h5py
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, 'models'))
import provider
from commons import check_mkdir

# Parsing Arguments
parser = argparse.ArgumentParser()
# Experiment Settings
parser.add_argument('--category', type=str, help='category')
parser.add_argument('--level_id', type=int, help='level_id')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', type=str, default="model", help='model to use [default: model]')
parser.add_argument('--epoch', type=int, default=251, help='Number of epochs [default: 50]')
parser.add_argument('--batch', type=int, default=1, help='Batch Size during training [default: 4]')
parser.add_argument('--point_num', type=int, default=10000, help='Point Number')
parser.add_argument('--group_num', type=int, default=200, help='Maximum Group Number in one pc')
parser.add_argument('--margin_same', type=float, default=1., help='Double hinge loss margin: same semantic')
parser.add_argument('--margin_diff', type=float, default=2., help='Double hinge loss margin: different semantic')

# Input&Output Settings
parser.add_argument('--log_dir', type=str, default='log', help='Directory that stores all training logs and trained models [default: log]')
parser.add_argument('--restore_dir', type=str, default='ckpt', help='Pretrained model to load')

FLAGS = parser.parse_args()

# save settings
MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
check_mkdir(LOG_DIR)

os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (__file__, LOG_DIR)) # bkp of train procedure
flog = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
flog.write(str(FLAGS)+'\n')

PRETRAINED_MODEL_PATH = os.path.join(FLAGS.restore_dir, 'trained_models/')

def printout(flog, data):
    print(data)
    flog.write(data + '\n')

# load meta data files
stat_in_fn = '../../stats/after_merging_label_ids/%s-level-%d.txt' % (FLAGS.category, FLAGS.level_id)
print('Reading from ', stat_in_fn)
with open(stat_in_fn, 'r') as fin:
    part_name_list = [item.rstrip().split()[1] for item in fin.readlines()]
print('Part Name List: ', part_name_list)
data_in_dir = '../../data/ins_seg_h5_for_sgpn/%s-%d/' % (FLAGS.category, FLAGS.level_id)
train_h5_fn_list = []
for item in os.listdir(data_in_dir):
    if item.endswith('.h5') and item.startswith('train-'):
        train_h5_fn_list.append(item)
val_h5_fn_list = []
for item in os.listdir(data_in_dir):
    if item.endswith('.h5') and item.startswith('val-'):
        val_h5_fn_list.append(item)

# parameters
POINT_NUM = FLAGS.point_num
printout(flog, 'Point Number: %d' % POINT_NUM)
BATCH_SIZE = FLAGS.batch
printout(flog, 'Batch Size: %d' % BATCH_SIZE)

NUM_CATEGORY = len(part_name_list)
printout(flog, 'Semantic Labels: %d' % NUM_CATEGORY)
NUM_GROUPS = FLAGS.group_num
printout(flog, 'Number of Instance Groups: %d' % NUM_GROUPS)

DECAY_RATE = 0.8
DECAY_STEP = 40000.0

LEARNING_RATE_CLIP = 1e-6
BASE_LEARNING_RATE = 1e-4
MOMENTUM = 0.9

TRAINING_EPOCHES = FLAGS.epoch
MARGINS = [FLAGS.margin_same, FLAGS.margin_diff]
print 'Margins: ', MARGINS

MODEL_STORAGE_PATH = os.path.join(LOG_DIR, 'trained_models')
if not os.path.exists(MODEL_STORAGE_PATH):
    os.mkdir(MODEL_STORAGE_PATH)

LOG_STORAGE_PATH = os.path.join(LOG_DIR, 'logs')
if not os.path.exists(LOG_STORAGE_PATH):
    os.mkdir(LOG_STORAGE_PATH)

SUMMARIES_FOLDER = os.path.join(LOG_DIR, 'summaries')
if not os.path.exists(SUMMARIES_FOLDER):
    os.mkdir(SUMMARIES_FOLDER)

all_data_cache = dict()
def load_data(fn):
    if fn in all_data_cache.keys():
        pts, semseg_one_hot, semseg_mask, insseg_one_hot, insseg_mask = all_data_cache[fn]
        print('Loading from cache.')
    else:
        out = h5py.File(fn)
        pts = out['pts'][:, :POINT_NUM, :]
        semseg_one_hot = out['semseg_one_hot'][:, :POINT_NUM, :]
        semseg_mask = out['semseg_mask'][:, :POINT_NUM]
        insseg_one_hot = out['insseg_one_hot'][:, :POINT_NUM, :]
        insseg_mask = out['insseg_mask'][:, :POINT_NUM]
        out.close()
        all_data_cache[fn] = (pts, semseg_one_hot, semseg_mask, insseg_one_hot, insseg_mask)
    return pts, semseg_one_hot, semseg_mask, insseg_one_hot, insseg_mask

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(FLAGS.gpu)):
            batch = tf.Variable(0, trainable=False, name='batch')
            learning_rate = tf.train.exponential_decay(
                BASE_LEARNING_RATE,  # base learning rate
                batch * BATCH_SIZE,  # global_var indicating the number of steps
                DECAY_STEP,  # step size
                DECAY_RATE,  # decay rate
                staircase=True  # Stair-case or continuous decreasing
            )
            learning_rate = tf.maximum(learning_rate, LEARNING_RATE_CLIP)

            lr_op = tf.summary.scalar('learning_rate', learning_rate)

            pointclouds_ph, ptsseglabel_ph, ptsgroup_label_ph, pts_seglabel_mask_ph, pts_group_mask_ph, alpha_ph = \
                MODEL.placeholder_inputs(BATCH_SIZE, POINT_NUM, NUM_GROUPS, NUM_CATEGORY)
            is_training_ph = tf.placeholder(tf.bool, shape=())

            # disable the two in first 5 epoches
            allow_full_loss_pl = tf.placeholder(tf.float32, shape=())

            labels = {'ptsgroup': ptsgroup_label_ph,
                      'semseg': ptsseglabel_ph,
                      'semseg_mask': pts_seglabel_mask_ph,
                      'group_mask': pts_group_mask_ph}

            net_output = MODEL.get_model(pointclouds_ph, NUM_CATEGORY, MARGINS[0], is_training_ph)
            loss, grouperr, same, same_cnt, diff, diff_cnt, pos, pos_cnt = MODEL.get_loss(net_output, labels, alpha_ph, MARGINS, allow_full_loss_pl)

            total_training_loss_ph = tf.placeholder(tf.float32, shape=())
            group_err_loss_ph = tf.placeholder(tf.float32, shape=())
            total_train_loss_sum_op = tf.summary.scalar('total_training_loss', total_training_loss_ph)
            group_err_op = tf.summary.scalar('group_err_loss', group_err_loss_ph)


        trainer = tf.train.AdamOptimizer(learning_rate)
        train_op = trainer.minimize(loss, global_step=batch)

        variables_to_load = [v for v in tf.global_variables()
                                 if
                                   ('conf_logits' not in v.name) and
                                    ('Fsim' not in v.name) and
                                    ('Fsconf' not in v.name) and
                                    ('batch' not in v.name)
                                ]
        
        loader = tf.train.Saver(variables_to_load)
        saver = tf.train.Saver(max_to_keep=50)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        init = tf.global_variables_initializer()
        sess.run(init)

        train_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/train', sess.graph)

        ckptstate = tf.train.get_checkpoint_state(PRETRAINED_MODEL_PATH)
        if ckptstate is not None:
            LOAD_MODEL_FILE = os.path.join(PRETRAINED_MODEL_PATH, os.path.basename(ckptstate.model_checkpoint_path))
            loader.restore(sess, LOAD_MODEL_FILE)
            printout(flog, "Model loaded in file: %s" % LOAD_MODEL_FILE)
        else:
            printout(flog, "Fail to load modelfile: %s" % PRETRAINED_MODEL_PATH)


        def train_one_epoch(epoch_num):

            ### NOTE: is_training = False: We do not update bn parameters during training due to the small batch size. This requires pre-training PointNet with large batchsize (say 32).
            is_training = False

            allow_full_loss = 0.0
            if epoch_num > 5:
                allow_full_loss = 1.0

            printout(flog, str(datetime.now()))

            random.shuffle(train_h5_fn_list)
            for item in train_h5_fn_list:
                cur_h5_fn = os.path.join(data_in_dir, item)
                printout(flog, 'Reading data from: %s' % cur_h5_fn)

                pts, semseg_one_hot, semseg_mask, insseg_one_hot, insseg_mask = load_data(cur_h5_fn)
 
                # shuffle data order
                n_shape = pts.shape[0]
                idx = np.arange(n_shape)
                np.random.shuffle(idx)
                
                pts = pts[idx, ...]
                semseg_one_hot = semseg_one_hot[idx, ...]
                semseg_mask = semseg_mask[idx, ...]
                insseg_one_hot = insseg_one_hot[idx, ...]
                insseg_mask = insseg_mask[idx, ...]

                # data augmentation to pts
                pts = provider.jitter_point_cloud(pts)
                pts = provider.shift_point_cloud(pts)
                pts = provider.random_scale_point_cloud(pts)
                pts = provider.rotate_perturbation_point_cloud(pts)

                total_loss = 0.0
                total_grouperr = 0.0
                total_same = 0.0
                total_diff = 0.0
                total_same = 0.0
                total_pos = 0.0
                total_acc = 0.0

                num_batch = n_shape // BATCH_SIZE
                for i in range(num_batch):
                    start_idx = i * BATCH_SIZE
                    end_idx = (i + 1) * BATCH_SIZE

                    feed_dict = {
                        pointclouds_ph: pts[start_idx: end_idx, ...],
                        ptsseglabel_ph: semseg_one_hot[start_idx: end_idx, ...],
                        ptsgroup_label_ph: insseg_one_hot[start_idx: end_idx, ...],
                        pts_seglabel_mask_ph: semseg_mask[start_idx: end_idx, ...],
                        pts_group_mask_ph: insseg_mask[start_idx: end_idx, ...],
                        is_training_ph: is_training,
                        alpha_ph: min(10., (float(epoch_num) / 5.) * 2. + 2.),
                        allow_full_loss_pl: allow_full_loss
                    }

                    _, loss_val, lr_val, simmat_val, semseg_logits_val, \
                            grouperr_val, same_val, same_cnt_val, diff_val, diff_cnt_val, pos_val, pos_cnt_val = sess.run([\
                                train_op, loss, learning_rate, \
                                net_output['simmat'], net_output['semseg_logits'], \
                                grouperr, same, same_cnt, diff, diff_cnt, pos, pos_cnt], feed_dict=feed_dict)

                    seg_pred = np.argmax(semseg_logits_val, axis=-1)
                    gt_mask_val = (np.sum(semseg_one_hot[start_idx: end_idx, ...], axis=-1) > 0)
                    seg_gt = np.argmax(semseg_one_hot[start_idx: end_idx, ...], axis=-1)
                    correct = ((seg_pred == seg_gt) + (~gt_mask_val) > 0)
                    acc_val = np.mean(correct)
        
                    total_acc += acc_val
                    total_loss += loss_val
                    total_grouperr += grouperr_val
                    total_diff += diff_val / max(1e-6, diff_cnt_val)
                    total_same += same_val / max(1e-6, same_cnt_val)
                    total_pos += pos_val / max(1e-6, pos_cnt_val)

                    if i % 10 == 9:
                        printout(flog, 'Batch: %d, LR: %f, loss: %f, FullLoss: %f, SegAcc: %f, grouperr: %f, same: %f, diff: %f, pos: %f' % \
                                (i, lr_val, total_loss/10, allow_full_loss, total_acc/10, total_grouperr/10, total_same/10, total_diff/10, total_pos/10))

                        lr_sum, batch_sum, train_loss_sum, group_err_sum = sess.run( \
                            [lr_op, batch, total_train_loss_sum_op, group_err_op], \
                            feed_dict={total_training_loss_ph: total_loss / 10.,
                                       group_err_loss_ph: total_grouperr / 10., })

                        train_writer.add_summary(train_loss_sum, batch_sum)
                        train_writer.add_summary(lr_sum, batch_sum)
                        train_writer.add_summary(group_err_sum, batch_sum)

                        total_grouperr = 0.0
                        total_loss = 0.0
                        total_diff = 0.0
                        total_same = 0.0
                        total_pos = 0.0
                        same_cnt0 = 0
                        total_acc = 0.0


        for epoch in range(TRAINING_EPOCHES):
            printout(flog, '\n>>> Training for the epoch %d/%d ...' % (epoch, TRAINING_EPOCHES))

            train_one_epoch(epoch)
            flog.flush()

            if (epoch+1) % 10 == 0:
                ckpt_fn = saver.save(sess, os.path.join(MODEL_STORAGE_PATH, 'epoch_%03d.ckpt'%epoch))
                printout(flog, 'Successfully store the checkpoint model into ' + ckpt_fn)

        flog.close()


# run
train()

