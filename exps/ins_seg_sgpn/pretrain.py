""" Pretrain the PointNet SemSeg Branch
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
parser.add_argument('--epoch', type=int, default=200, help='Number of epochs [default: 50]')
parser.add_argument('--batch', type=int, default=4, help='Batch Size during training [default: 4]')
parser.add_argument('--point_num', type=int, default=4096, help='Point Number')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning Rate [default: 1e-4]')

# Input&Output Settings
parser.add_argument('--log_dir', type=str, default='log', help='Directory that stores all training logs and trained models [default: log]')

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

DECAY_RATE = 0.8
DECAY_STEP = 40000.0
printout(flog, 'lr_decay_rate: %f, lr_decay_step: %f' % (DECAY_RATE, DECAY_STEP))

LEARNING_RATE_CLIP = 1e-6
BASE_LEARNING_RATE = FLAGS.learning_rate
MOMENTUM = 0.9

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = DECAY_STEP
BN_DECAY_CLIP = 0.99

TRAINING_EPOCHES = FLAGS.epoch

MODEL_STORAGE_PATH = os.path.join(LOG_DIR, 'trained_models')
if not os.path.exists(MODEL_STORAGE_PATH):
    os.mkdir(MODEL_STORAGE_PATH)

LOG_STORAGE_PATH = os.path.join(LOG_DIR, 'logs')
if not os.path.exists(LOG_STORAGE_PATH):
    os.mkdir(LOG_STORAGE_PATH)

SUMMARIES_FOLDER = os.path.join(LOG_DIR, 'summaries')
if not os.path.exists(SUMMARIES_FOLDER):
    os.mkdir(SUMMARIES_FOLDER)


def load_data(fn):
    out = h5py.File(fn)
    pts = out['pts'][:, :POINT_NUM, :]
    semseg_one_hot = out['semseg_one_hot'][:, :POINT_NUM, :]
    semseg_mask = out['semseg_mask'][:, :POINT_NUM]
    out.close()
    return pts, semseg_one_hot, semseg_mask

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

            bn_momentum = tf.train.exponential_decay(
                              BN_INIT_DECAY,
                              batch*BATCH_SIZE,
                              BN_DECAY_DECAY_STEP,
                              BN_DECAY_DECAY_RATE,
                              staircase=True)
            bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)

            pointclouds_ph, ptsseglabel_ph, pts_seglabel_mask_ph = MODEL.placeholder_inputs(BATCH_SIZE, POINT_NUM, NUM_CATEGORY)
            is_training_ph = tf.placeholder(tf.bool, shape=())

            labels = {'semseg': ptsseglabel_ph,
                      'semseg_mask': pts_seglabel_mask_ph}

            net_output = MODEL.get_model(pointclouds_ph, NUM_CATEGORY, is_training_ph, bn_decay=bn_decay)
            loss = MODEL.get_loss(net_output, labels)

            lr_op = tf.summary.scalar('learning_rate', learning_rate)
            bn_decay_op = tf.summary.scalar('bn_decay', bn_decay)
            total_training_loss_ph = tf.placeholder(tf.float32, shape=())
            total_training_acc_ph = tf.placeholder(tf.float32, shape=())
            total_train_loss_sum_op = tf.summary.scalar('total_training_loss', total_training_loss_ph)
            total_train_acc_sum_op = tf.summary.scalar('total_training_acc', total_training_acc_ph)


        trainer = tf.train.AdamOptimizer(learning_rate)
        train_op = trainer.minimize(loss, global_step=batch)

        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        init = tf.global_variables_initializer()
        sess.run(init)

        train_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/train', sess.graph)

        for v in tf.global_variables():
            print v.name

        
        def train_one_epoch(epoch_num):

            is_training = True
            
            printout(flog, str(datetime.now()))

            random.shuffle(train_h5_fn_list)
            for item in train_h5_fn_list:
                cur_h5_fn = os.path.join(data_in_dir, item)
                printout(flog, 'Reading data from: %s' % cur_h5_fn)

                pts, semseg_one_hot, semseg_mask = load_data(cur_h5_fn)
 
                # shuffle data order
                n_shape = pts.shape[0]
                idx = np.arange(n_shape)
                np.random.shuffle(idx)
                
                pts = pts[idx, ...]
                semseg_one_hot = semseg_one_hot[idx, ...]
                semseg_mask = semseg_mask[idx, ...]

                num_batch = n_shape // BATCH_SIZE
                for i in range(num_batch):
                    start_idx = i * BATCH_SIZE
                    end_idx = (i + 1) * BATCH_SIZE

                    feed_dict = {
                        pointclouds_ph: pts[start_idx: end_idx, ...],
                        ptsseglabel_ph: semseg_one_hot[start_idx: end_idx, ...],
                        pts_seglabel_mask_ph: semseg_mask[start_idx: end_idx, ...],
                        is_training_ph: is_training
                    }

                    _, lr_val, bn_decay_val, loss_val, seg_val = sess.run([train_op, \
                            learning_rate, bn_decay, loss, net_output], feed_dict=feed_dict)

                    seg_pred = np.argmax(seg_val, axis=-1)
                    gt_mask_val = (np.sum(semseg_one_hot[start_idx: end_idx, ...], axis=-1) > 0)
                    seg_gt = np.argmax(semseg_one_hot[start_idx: end_idx, ...], axis=-1)
                    correct = ((seg_pred == seg_gt) + (1 - gt_mask_val) > 0)
                    acc_val = np.mean(correct)

                    printout(flog, 'Epoch: %d, Batch: %d, BN-Decay: %f, LR: %f: loss: %f, Acc: %f' % \
                            (epoch_num, i, bn_decay_val, lr_val, loss_val, acc_val))

                    lr_sum, bn_decay_sum, batch_sum, train_loss_sum, train_acc_sum = sess.run(\
                            [lr_op, bn_decay_op, batch, total_train_loss_sum_op, total_train_acc_sum_op], \
                            feed_dict={total_training_loss_ph: loss_val, \
                                       total_training_acc_ph: acc_val
                            })

                    train_writer.add_summary(lr_sum, batch_sum)
                    train_writer.add_summary(bn_decay_sum, batch_sum)
                    train_writer.add_summary(train_loss_sum, batch_sum)
                    train_writer.add_summary(train_acc_sum, batch_sum)


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

