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
import provider
import tf_util
import random
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
parser.add_argument('--num_ins', type=int, default='100', help='Max Number of Instance [default: 100]')
parser.add_argument('--log_dir', type=str, default='log', help='Log dir [default: log]')
parser.add_argument('--visu_dir', type=str, default=None, help='Visu dir [default: None, meaning no visu]')
parser.add_argument('--visu_batch', type=int, default=1, help='visu batch [default: 1]')
parser.add_argument('--num_point', type=int, default=10000, help='Point Number [default: 10000]')
parser.add_argument('--max_epoch', type=int, default=251, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--seg_loss_weight', type=float, default=1.0, help='Semantic Segmentation Loss Weight [default: 1.0]')
parser.add_argument('--ins_loss_weight', type=float, default=1.0, help='Instance Segmentation Loss Weight [default: 1.0]')
parser.add_argument('--other_ins_loss_weight', type=float, default=1.0, help='Instance Segmentation for the Part *Other* Loss Weight [default: 1.0]')
parser.add_argument('--l21_norm_loss_weight', type=float, default=1.0, help='l21 Norm Loss Weight [default: 1.0]')
parser.add_argument('--conf_loss_weight', type=float, default=1.0, help='Conf Loss Weight [default: 1.0]')
parser.add_argument('--num_train_epoch_per_test', type=int, default=1, help='Number of train epochs per testing [default: 1]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
check_mkdir(LOG_DIR)
CKPT_DIR = os.path.join(LOG_DIR, 'trained_models')
force_mkdir(CKPT_DIR)

if FLAGS.visu_dir is not None:
    VISU_DIR = os.path.join(LOG_DIR, FLAGS.visu_dir)
    force_mkdir(VISU_DIR)

os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (__file__, LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

DECAY_RATE = 0.8
DECAY_STEP = 40000.0
log_string('lr_decay_rate: %f, lr_decay_step: %f' % (DECAY_RATE, DECAY_STEP))

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

# load meta data files
stat_in_fn = '../../stats/after_merging2_label_ids/%s-level-%d.txt' % (FLAGS.category, FLAGS.level_id)
print('Reading from ', stat_in_fn)
with open(stat_in_fn, 'r') as fin:
    part_name_list = [item.rstrip().split()[1] for item in fin.readlines()]
print('Part Name List: ', part_name_list)
data_in_dir = '../../data/ins_seg_h5_for_detection/%s-%d/' % (FLAGS.category, FLAGS.level_id)
train_h5_fn_list = []
for item in os.listdir(data_in_dir):
    if item.endswith('.h5') and item.startswith('train-'):
        train_h5_fn_list.append(item)
val_h5_fn_list = []
for item in os.listdir(data_in_dir):
    if item.endswith('.h5') and item.startswith('val-'):
        val_h5_fn_list.append(item)

NUM_CLASSES = len(part_name_list)
print('Semantic Labels: ', NUM_CLASSES)
NUM_INS = FLAGS.num_ins
print('Number of Instances: ', NUM_INS)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pc_pl, label_pl, gt_mask_pl, gt_valid_pl, gt_other_mask_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_INS)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            seg_pred, mask_pred, other_mask_pred, conf_pred, end_points = MODEL.get_model(pc_pl, NUM_CLASSES, NUM_INS, \
                                                                               is_training_pl, bn_decay=bn_decay)
            seg_loss, end_points = MODEL.get_seg_loss(seg_pred, label_pl, end_points) 
            tf.summary.scalar('seg_loss', seg_loss)

            ins_loss, end_points = MODEL.get_ins_loss(mask_pred, gt_mask_pl, gt_valid_pl, end_points)
            tf.summary.scalar('ins_loss', ins_loss)

            other_ins_loss, end_points = MODEL.get_other_ins_loss(other_mask_pred, gt_other_mask_pl, end_points)
            tf.summary.scalar('other_ins_loss', other_ins_loss)

            l21_norm_loss, end_points = MODEL.get_l21_norm(mask_pred, other_mask_pred, end_points)
            tf.summary.scalar('l21_norm_loss', l21_norm_loss)

            conf_loss, end_points = MODEL.get_conf_loss(conf_pred, gt_valid_pl, end_points)
            tf.summary.scalar('conf_loss', conf_loss)

            total_loss = FLAGS.seg_loss_weight * seg_loss + FLAGS.ins_loss_weight * ins_loss + \
                    FLAGS.other_ins_loss_weight * other_ins_loss + FLAGS.l21_norm_loss_weight * l21_norm_loss + \
                    FLAGS.conf_loss_weight * conf_loss
            tf.summary.scalar('total_loss', total_loss)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=50)
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {'pc_pl': pc_pl,
               'label_pl': label_pl,
               'gt_mask_pl': gt_mask_pl,
               'gt_valid_pl': gt_valid_pl,
               'gt_other_mask_pl': gt_other_mask_pl,
               'is_training_pl': is_training_pl,
               'seg_pred': seg_pred,
               'mask_pred': mask_pred,
               'conf_pred': conf_pred,
               'other_mask_pred': other_mask_pred,
               'seg_loss': seg_loss,
               'ins_loss': ins_loss,
               'conf_loss': conf_loss,
               'other_ins_loss': other_ins_loss,
               'l21_norm_loss': l21_norm_loss,
               'loss': total_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'learning_rate': learning_rate,
               'bn_decay': bn_decay,
               'end_points': end_points}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer, epoch)

            if (epoch+1) % FLAGS.num_train_epoch_per_test == 0:
                eval_one_epoch(sess, ops, test_writer, epoch)
                save_path = saver.save(sess, os.path.join(CKPT_DIR, "epoch-%03d.ckpt"%epoch))
                log_string("Model saved in file: %s" % save_path)

def load_h5(fn):
    with h5py.File(fn, 'r') as fin:
        pts = fin['pts'][:]
        gt_label = fin['gt_label'][:]
        gt_mask = fin['gt_mask'][:]
        gt_valid = fin['gt_valid'][:]
        gt_other_mask = fin['gt_other_mask'][:]
        return pts, gt_label, gt_mask, gt_valid, gt_other_mask

def load_json(fn):
    with open(fn, 'r') as fin:
        return json.load(fin)

def load_data(fn):
    cur_json_fn = fn.replace('.h5', '.json')
    record = load_json(cur_json_fn)
    pts, gt_label, gt_mask, gt_valid, gt_other_mask = load_h5(fn)
    return pts, gt_label, gt_mask, gt_valid, gt_other_mask, record

def train_one_epoch(sess, ops, writer, epoch):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string(str(datetime.now()))

    # shuffle training files order
    random.shuffle(train_h5_fn_list)

    for item in train_h5_fn_list:
        cur_h5_fn = os.path.join(data_in_dir, item)
        print('Reading data from ', cur_h5_fn)
        pts, gt_label, gt_mask, gt_valid, gt_other_mask, _ = load_data(cur_h5_fn)
        
        # shuffle data order
        n_shape = pts.shape[0]
        idx = np.arange(n_shape)
        np.random.shuffle(idx)
        
        pts = pts[idx, ...]
        gt_label = gt_label[idx, ...]
        gt_mask = gt_mask[idx, ...]
        gt_valid = gt_valid[idx, ...]
        gt_other_mask = gt_other_mask[idx, ...]

        # data augmentation to pts
        pts = provider.jitter_point_cloud(pts)
        pts = provider.shift_point_cloud(pts)
        pts = provider.random_scale_point_cloud(pts)
        pts = provider.rotate_perturbation_point_cloud(pts)

        num_batch = n_shape // BATCH_SIZE
        for i in range(num_batch):
            start_idx = i * BATCH_SIZE
            end_idx = (i + 1) * BATCH_SIZE

            cur_pts = pts[start_idx: end_idx, ...]
            cur_gt_label = gt_label[start_idx: end_idx, ...]
            cur_gt_mask = gt_mask[start_idx: end_idx, ...]
            cur_gt_valid = gt_valid[start_idx: end_idx, ...]
            cur_gt_other_mask = gt_other_mask[start_idx: end_idx, ...]

            feed_dict = {ops['pc_pl']: cur_pts,
                         ops['label_pl']: cur_gt_label,
                         ops['gt_mask_pl']: cur_gt_mask,
                         ops['gt_valid_pl']: cur_gt_valid,
                         ops['gt_other_mask_pl']: cur_gt_other_mask,
                         ops['is_training_pl']: is_training}

            summary, step, _, lr_val, bn_decay_val, seg_loss_val, ins_loss_val, other_ins_loss_val, l21_norm_loss_val, conf_loss_val, loss_val, seg_pred_val, \
                    mask_pred_val, other_mask_pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['learning_rate'], ops['bn_decay'], \
                        ops['seg_loss'], ops['ins_loss'], ops['other_ins_loss'], ops['l21_norm_loss'], ops['conf_loss'], ops['loss'], \
                        ops['seg_pred'], ops['mask_pred'], ops['other_mask_pred']], feed_dict=feed_dict)

            writer.add_summary(summary, step)

            seg_pred_id = np.argmax(seg_pred_val, axis=-1)
            seg_acc = np.mean(seg_pred_id == cur_gt_label)

            log_string('[Train Epoch %03d, Batch %03d, LR: %f, BN_DECAY: %f] Loss: %f = %f x %f (seg_loss, Seg Acc: %f) + %f x %f (ins_loss) + %f x %f (other_ins_loss) + %f x %f (l21_norm_loss) + %f x %f (conf_loss)' \
                    % (epoch, i, lr_val, bn_decay_val, loss_val, FLAGS.seg_loss_weight, seg_loss_val, seg_acc, \
                    FLAGS.ins_loss_weight, ins_loss_val, FLAGS.other_ins_loss_weight, other_ins_loss_val, \
                    FLAGS.l21_norm_loss_weight, l21_norm_loss_val, FLAGS.conf_loss_weight, conf_loss_val))


def gen_visu(out_dir, visu_batch, pts, record, gt_label, gt_mask, gt_valid, gt_other_mask, \
             seg_pred_id, per_shape_seg_acc, mask_pred_val, other_mask_pred_val, matching_idx_val, \
             per_shape_mean_iou_val, per_shape_all_iou_val, per_shape_other_iou_val, \
             per_shape_seg_loss_val, per_shape_l21_norm_val, conf_gt, conf_pred, \
             threshold1=0.2, threshold2=0.5, threshold3=0.8):
    
    n_shape = pts.shape[0]

    pts_dir = os.path.join(out_dir, 'pts')
    gt_sem_dir = os.path.join(out_dir, 'gt_sem')
    pred_sem_dir = os.path.join(out_dir, 'pred_sem')
    info_dir = os.path.join(out_dir, 'info')
    conf_dir = os.path.join(out_dir, 'conf')
    child_dir = os.path.join(out_dir, 'child')
    if visu_batch == 0:
        os.mkdir(pts_dir)
        os.mkdir(gt_sem_dir)
        os.mkdir(pred_sem_dir)
        os.mkdir(info_dir)
        os.mkdir(conf_dir)
        os.mkdir(child_dir)

    bar = ProgressBar()
    for i in bar(range(n_shape)):
        cur_shape_id = visu_batch * BATCH_SIZE + i
        cur_fn_prefix = 'shape-%02d' % cur_shape_id
        cur_pts = pts[i, ...]
        cur_gt_label = gt_label[i, :]
        cur_pred_label = seg_pred_id[i, :]
        cur_record = record[i]
        out_fn = os.path.join(pts_dir, cur_fn_prefix+'.png')
        render_pts(out_fn, cur_pts)
        out_fn = os.path.join(gt_sem_dir, cur_fn_prefix+'.png')
        render_pts_with_label(out_fn, cur_pts, cur_gt_label)
        out_fn = os.path.join(pred_sem_dir, cur_fn_prefix+'.png')
        render_pts_with_label(out_fn, cur_pts, cur_pred_label)
        out_fn = os.path.join(info_dir, cur_fn_prefix+'.txt')
        with open(out_fn, 'w') as fout:
            fout.write('model_id: %s, anno_id: %s\n' % (cur_record['model_id'], cur_record['anno_id']))
            fout.write('Sem Seg Acc: %f\n' % (per_shape_seg_acc[i]))
            cur_tot_loss = FLAGS.seg_loss_weight * per_shape_seg_loss_val[i] - FLAGS.ins_loss_weight * per_shape_mean_iou_val[i] - \
                    FLAGS.other_ins_loss_weight * per_shape_other_iou_val[i] + FLAGS.l21_norm_loss_weight * per_shape_l21_norm_val[i]
            fout.write('Loss %f = %f x %f (sem_seg) + %f x %f (ins_seg) + %f x %f (other_ins_seg) + %f x %f (l21_norm)\n' % \
                    (cur_tot_loss, FLAGS.seg_loss_weight, per_shape_seg_loss_val[i], FLAGS.ins_loss_weight, -per_shape_mean_iou_val[i], \
                     FLAGS.other_ins_loss_weight, -per_shape_other_iou_val[i], FLAGS.l21_norm_loss_weight, per_shape_l21_norm_val[i]))
        out_fn = os.path.join(conf_dir, cur_fn_prefix+'.txt')
        with open(out_fn, 'w') as fout:
            for j in range(NUM_INS):
                fout.write('part %d\t:gt = %f\tpred = %f\n' % (j, conf_gt[i, j], conf_pred[i, j]))

        cur_child_dir = os.path.join(child_dir, cur_fn_prefix)
        os.mkdir(cur_child_dir)
        child_gt_part_dir = os.path.join(cur_child_dir, 'gt_part')
        os.mkdir(child_gt_part_dir)
        child_pred_part1_dir = os.path.join(cur_child_dir, 'pred_part1')
        os.mkdir(child_pred_part1_dir)
        child_pred_part2_dir = os.path.join(cur_child_dir, 'pred_part2')
        os.mkdir(child_pred_part2_dir)
        child_pred_part3_dir = os.path.join(cur_child_dir, 'pred_part3')
        os.mkdir(child_pred_part3_dir)
        child_info_dir = os.path.join(cur_child_dir, 'info')
        os.mkdir(child_info_dir)

        for j in range(int(np.sum(gt_valid[i, :]))):
            child_fn_prefix = 'part-%02d' % j
            gt_part_id = matching_idx_val[i, j, 0]
            pred_part_id = matching_idx_val[i, j, 1]
            out_fn = os.path.join(child_gt_part_dir, child_fn_prefix+'.png')
            render_pts_with_label(out_fn, cur_pts, (gt_mask[i, gt_part_id, :] > threshold3).astype(np.int32))
            out_fn = os.path.join(child_pred_part1_dir, child_fn_prefix+'.png')
            render_pts_with_label(out_fn, cur_pts, (mask_pred_val[i, pred_part_id, :] > threshold1).astype(np.int32))
            out_fn = os.path.join(child_pred_part2_dir, child_fn_prefix+'.png')
            render_pts_with_label(out_fn, cur_pts, (mask_pred_val[i, pred_part_id, :] > threshold2).astype(np.int32))
            out_fn = os.path.join(child_pred_part3_dir, child_fn_prefix+'.png')
            render_pts_with_label(out_fn, cur_pts, (mask_pred_val[i, pred_part_id, :] > threshold3).astype(np.int32))
            out_fn = os.path.join(child_info_dir, child_fn_prefix+'.txt')
            with open(out_fn, 'w') as fout:
                fout.write('gt %d --> pred %d\n' % (gt_part_id, pred_part_id))
                fout.write('IoU: %f\n' % (per_shape_all_iou_val[i, gt_part_id]))
     
        child_fn_prefix = 'part-other'
        out_fn = os.path.join(child_gt_part_dir, child_fn_prefix+'.png')
        render_pts_with_label(out_fn, cur_pts, (gt_other_mask[i, :] > threshold3).astype(np.int32))
        out_fn = os.path.join(child_pred_part1_dir, child_fn_prefix+'.png')
        render_pts_with_label(out_fn, cur_pts, (other_mask_pred_val[i, :] > threshold1).astype(np.int32))
        out_fn = os.path.join(child_pred_part2_dir, child_fn_prefix+'.png')
        render_pts_with_label(out_fn, cur_pts, (other_mask_pred_val[i, :] > threshold2).astype(np.int32))
        out_fn = os.path.join(child_pred_part3_dir, child_fn_prefix+'.png')
        render_pts_with_label(out_fn, cur_pts, (other_mask_pred_val[i, :] > threshold3).astype(np.int32))
        out_fn = os.path.join(child_info_dir, child_fn_prefix+'.txt')
        with open(out_fn, 'w') as fout:
            fout.write('IoU: %f\n' % (per_shape_other_iou_val[i]))


def eval_one_epoch(sess, ops, writer, epoch):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    
    log_string(str(datetime.now()))

    if FLAGS.visu_dir is not None:
        cur_visu_batch = 0
        cur_visu_dir = os.path.join(VISU_DIR, 'epoch-%03d' % epoch)
        os.mkdir(cur_visu_dir)

    cnt = 0; tot_loss = 0.0; tot_seg_loss = 0.0; tot_ins_loss = 0.0; tot_other_ins_loss = 0.0; tot_l21_norm_loss = 0.0; tot_conf_loss = 0.0; tot_seg_acc = 0.0;
    for item in val_h5_fn_list:
        cur_h5_fn = os.path.join(data_in_dir, item)
        print('Reading data from ', cur_h5_fn)
        pts, gt_label, gt_mask, gt_valid, gt_other_mask, record = load_data(cur_h5_fn)
         
        n_shape = pts.shape[0]
        num_batch = int((n_shape - 1) / BATCH_SIZE) + 1

        cur_pts = np.zeros((BATCH_SIZE, NUM_POINT, 3), dtype=np.float32)
        cur_gt_label = np.zeros((BATCH_SIZE, NUM_POINT), dtype=np.int32)
        cur_gt_mask = np.zeros((BATCH_SIZE, NUM_INS, NUM_POINT), dtype=np.float32)
        cur_gt_valid = np.zeros((BATCH_SIZE, NUM_INS), dtype=np.float32)
        cur_gt_other_mask = np.zeros((BATCH_SIZE, NUM_POINT), dtype=np.float32)

        for i in range(num_batch):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, n_shape)

            cur_pts[:end_idx-start_idx, ...] = pts[start_idx: end_idx, ...]
            cur_gt_label[:end_idx-start_idx, ...] = gt_label[start_idx: end_idx, ...]
            cur_gt_mask[:end_idx-start_idx, ...] = gt_mask[start_idx: end_idx, ...]
            cur_gt_valid[:end_idx-start_idx, ...] = gt_valid[start_idx: end_idx, ...]
            cur_gt_other_mask[:end_idx-start_idx, ...] = gt_other_mask[start_idx: end_idx, ...]
            cur_record = record[start_idx: end_idx]

            feed_dict = {ops['pc_pl']: cur_pts,
                         ops['label_pl']: cur_gt_label,
                         ops['gt_mask_pl']: cur_gt_mask,
                         ops['gt_valid_pl']: cur_gt_valid,
                         ops['gt_other_mask_pl']: cur_gt_other_mask,
                         ops['is_training_pl']: is_training}

            summary, step, seg_loss_val, ins_loss_val, other_ins_loss_val, l21_norm_loss_val, conf_loss_val, loss_val, \
                    per_shape_seg_loss_val, per_shape_l21_norm_val, \
                    seg_pred_val, mask_pred_val, other_mask_pred_val, \
                    matching_idx_val, per_shape_mean_iou_val, per_shape_all_iou_val, per_shape_other_iou_val, \
                    conf_gt, conf_pred \
             = sess.run([ops['merged'], ops['step'], \
                        ops['seg_loss'], ops['ins_loss'], ops['other_ins_loss'], ops['l21_norm_loss'], ops['conf_loss'], ops['loss'], \
                        ops['end_points']['per_shape_seg_loss'], ops['end_points']['per_shape_l21_norm'], \
                        ops['seg_pred'], ops['mask_pred'], ops['other_mask_pred'], \
                        ops['end_points']['matching_idx'], ops['end_points']['per_shape_mean_iou'], \
                        ops['end_points']['per_shape_all_iou'], ops['end_points']['per_shape_other_iou'], \
                        ops['end_points']['per_part_conf_target'], ops['conf_pred']
                    ], feed_dict=feed_dict)

            writer.add_summary(summary, step)

            seg_pred_id = np.argmax(seg_pred_val, axis=-1)
            per_shape_seg_acc = np.mean(seg_pred_id == cur_gt_label, axis=-1)
            per_shape_seg_acc = per_shape_seg_acc[:end_idx-start_idx]
            seg_acc = np.mean(per_shape_seg_acc)

            cnt += end_idx-start_idx
            tot_loss += loss_val * (end_idx-start_idx)
            tot_seg_loss += seg_loss_val * (end_idx-start_idx)
            tot_ins_loss += ins_loss_val * (end_idx-start_idx)
            tot_conf_loss += conf_loss_val * (end_idx-start_idx)
            tot_other_ins_loss += other_ins_loss_val * (end_idx-start_idx)
            tot_l21_norm_loss += l21_norm_loss_val * (end_idx-start_idx)
            tot_seg_acc += seg_acc * (end_idx-start_idx)

            if FLAGS.visu_dir is not None and cur_visu_batch < FLAGS.visu_batch:
                gen_visu(cur_visu_dir, cur_visu_batch, cur_pts[:end_idx-start_idx, ...], cur_record, cur_gt_label[:end_idx-start_idx, ...], \
                        cur_gt_mask[:end_idx-start_idx, ...], cur_gt_valid[:end_idx-start_idx, ...], cur_gt_other_mask[:end_idx-start_idx, ...], \
                        seg_pred_id[:end_idx-start_idx, ...], per_shape_seg_acc[:end_idx-start_idx, ...], mask_pred_val[:end_idx-start_idx, ...], \
                        other_mask_pred_val[:end_idx-start_idx, ...], matching_idx_val[:end_idx-start_idx, ...], \
                        per_shape_mean_iou_val[:end_idx-start_idx, ...], per_shape_all_iou_val[:end_idx-start_idx, ...], per_shape_other_iou_val[:end_idx-start_idx, ...], \
                        per_shape_seg_loss_val[:end_idx-start_idx, ...], per_shape_l21_norm_val[:end_idx-start_idx, ...], conf_gt[:end_idx-start_idx, ...], conf_pred[:end_idx-start_idx, ...])
                cur_visu_batch += 1

    avg_loss = tot_loss / cnt
    avg_seg_loss = tot_seg_loss / cnt
    avg_ins_loss = tot_ins_loss / cnt
    avg_other_ins_loss = tot_other_ins_loss / cnt
    avg_l21_norm_loss = tot_l21_norm_loss / cnt
    avg_conf_loss = tot_conf_loss / cnt
    avg_acc = tot_seg_acc / cnt

    log_string('[Test Epoch %03d] Average Loss: cnt: %d, %f = %f x %f (seg_loss, Seg Acc: %f) + %f x %f (ins_loss) + %f x %f (other_ins_loss) + %f x %f (l21_norm_loss) + %f x %f (conf_loss)' \
            % (epoch, cnt, avg_loss, FLAGS.seg_loss_weight, avg_seg_loss, avg_acc, \
            FLAGS.ins_loss_weight, avg_ins_loss, FLAGS.other_ins_loss_weight, avg_other_ins_loss, \
            FLAGS.l21_norm_loss_weight, avg_l21_norm_loss, FLAGS.conf_loss_weight, avg_conf_loss))

    if FLAGS.visu_dir is not None:
        cmd = 'cd %s && python %s . 1 htmls pts,gt_sem,pred_sem,info:gt_part,pred_part1,pred_part2,pred_part3,info pts,gt_sem,pred_sem,info:gt_part,pred_part1,pred_part2,pred_part3,info' % (cur_visu_dir, os.path.join(BASE_DIR, '../utils/gen_html_hierachy_local.py'))
        log_string(cmd)
        call(cmd, shell=True)

# main
log_string('pid: %s'%(str(os.getpid())))
train()
LOG_FOUT.close()

