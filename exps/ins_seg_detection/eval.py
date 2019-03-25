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
parser.add_argument('--num_ins', type=int, default='100', help='Max Number of Instance [default: 100]')
parser.add_argument('--log_dir', type=str, default='log', help='Log dir [default: log]')
parser.add_argument('--eval_dir', type=str, default='eval', help='Eval dir [default: eval]')
parser.add_argument('--pred_dir', type=str, default='pred', help='Pred dir [default: pred]')
parser.add_argument('--visu_dir', type=str, default=None, help='Visu dir [default: None, meaning no visu]')
parser.add_argument('--visu_batch', type=int, default=1, help='visu batch [default: 1]')
parser.add_argument('--num_point', type=int, default=10000, help='Point Number [default: 10000]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--min_point_score_to_exist', type=float, default=0.5, help='minimum per-point mask score for the point to be present in the final prediction [default: 0.5]')
parser.add_argument('--min_num_point_per_part', type=int, default=10, help='minimum number of points exist in a part for the part to be present in the final prediction [default: 10]')
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
LOG_DIR = os.path.join(LOG_DIR, FLAGS.eval_dir)
check_mkdir(LOG_DIR)
PRED_DIR = os.path.join(LOG_DIR, FLAGS.pred_dir)
force_mkdir(PRED_DIR)
if FLAGS.visu_dir is not None:
    VISU_DIR = os.path.join(LOG_DIR, FLAGS.visu_dir)
    force_mkdir(VISU_DIR)

os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (__file__, LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_eval.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

# load meta data files
stat_in_fn = '../../stats/after_merging_label_ids/%s-level-%d.txt' % (FLAGS.category, FLAGS.level_id)
print('Reading from ', stat_in_fn)
with open(stat_in_fn, 'r') as fin:
    part_name_list = [item.rstrip().split()[1] for item in fin.readlines()]
print('Part Name List: ', part_name_list)
data_in_dir = '../../data/ins_seg_h5_for_detection/%s-%d/' % (FLAGS.category, FLAGS.level_id)
test_h5_fn_list = []
for item in os.listdir(data_in_dir):
    if item.endswith('.h5') and item.startswith('test-'):
        test_h5_fn_list.append(item)
test_h5_fn_list = sorted(test_h5_fn_list)

NUM_CLASSES = len(part_name_list)
print('Semantic Labels: ', NUM_CLASSES)
NUM_INS = FLAGS.num_ins
print('Number of Instances: ', NUM_INS)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def load_h5(fn):
    with h5py.File(fn, 'r') as fin:
        pts = fin['pts'][:, :NUM_POINT, :]
        return pts

def load_json(fn):
    with open(fn, 'r') as fin:
        return json.load(fin)

def load_data(fn):
    cur_json_fn = fn.replace('.h5', '.json')
    record = load_json(cur_json_fn)
    pts = load_h5(fn)
    return pts, record

def save_h5(fn, mask, valid, conf, label):
    fout = h5py.File(fn, 'w')
    fout.create_dataset('mask', data=mask, compression='gzip', compression_opts=4, dtype='bool')
    fout.create_dataset('valid', data=valid, compression='gzip', compression_opts=4, dtype='bool')
    fout.create_dataset('conf', data=conf, compression='gzip', compression_opts=4, dtype='float32')
    fout.create_dataset('label', data=label, compression='gzip', compression_opts=4, dtype='uint8')
    fout.close()

def gen_visu(base_idx, pts, mask, valid, conf, sem, record, num_pts_to_visu=1000):
    n_shape = pts.shape[0]
    n_ins = mask.shape[1]
    
    pts_dir = os.path.join(VISU_DIR, 'pts')
    info_dir = os.path.join(VISU_DIR, 'info')
    child_dir = os.path.join(VISU_DIR, 'child')

    if base_idx == 0:
        os.mkdir(pts_dir)
        os.mkdir(info_dir)
        os.mkdir(child_dir)

    for i in range(n_shape):
        cur_pts = pts[i, ...]
        cur_mask = mask[i, ...]
        cur_valid = valid[i, :]
        cur_conf = conf[i, :]
        cur_sem = sem[i, :]
        cur_record = record[i]

        #cur_idx_to_visu = np.arange(NUM_POINT)
        #np.random.shuffle(cur_idx_to_visu)
        #cur_idx_to_visu = cur_idx_to_visu[:num_pts_to_visu]

        cur_shape_prefix = 'shape-%03d' % (base_idx + i)
        out_fn = os.path.join(pts_dir, cur_shape_prefix+'.png')
        render_pts(out_fn, cur_pts)
        out_fn = os.path.join(info_dir, cur_shape_prefix+'.txt')
        with open(out_fn, 'w') as fout:
            fout.write('model_id: %s, anno_id: %s\n' % (cur_record['model_id'], cur_record['anno_id']))
        
        cur_child_dir = os.path.join(child_dir, cur_shape_prefix)
        os.mkdir(cur_child_dir)
        child_pred_dir = os.path.join(cur_child_dir, 'pred')
        os.mkdir(child_pred_dir)
        child_info_dir = os.path.join(cur_child_dir, 'info')
        os.mkdir(child_info_dir)

        cur_conf[~cur_valid] = 0.0
        idx = np.argsort(-cur_conf)
        for j in range(n_ins):
            cur_idx = idx[j]
            if cur_valid[cur_idx]:
                cur_part_prefix = 'part-%03d' % j
                out_fn = os.path.join(child_pred_dir, cur_part_prefix+'.png')
                render_pts_with_label(out_fn, cur_pts, cur_mask[cur_idx].astype(np.int32))
                out_fn = os.path.join(child_info_dir, cur_part_prefix+'.txt')
                with open(out_fn, 'w') as fout:
                    fout.write('part idx: %d\n' % cur_idx)
                    fout.write('label: %s\n' % part_name_list[cur_sem[cur_idx]])
                    fout.write('score: %f\n' % cur_conf[cur_idx])
                    fout.write('#pts: %d\n' % np.sum(cur_mask[cur_idx, :]))


def eval():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pc_pl, _, _, _, _ = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_INS)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            seg_pred, mask_pred, _, conf_pred, _ = MODEL.get_model(pc_pl, NUM_CLASSES, NUM_INS, is_training_pl)

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
            log_string("Model loaded in file: %s" % LOAD_MODEL_FILE)
        else:
            log_string("Fail to load modelfile: %s" % CKPT_DIR)

        # visu
        if FLAGS.visu_dir is not None:
            cur_visu_batch = 0

        # Start testing
        batch_pts = np.zeros((BATCH_SIZE, NUM_POINT, 3), dtype=np.float32)

        for item in test_h5_fn_list:
            cur_h5_fn = os.path.join(data_in_dir, item)
            print('Reading data from ', cur_h5_fn)
            pts, record = load_data(cur_h5_fn)

            n_shape = pts.shape[0]
            num_batch = int((n_shape - 1) * 1.0 / BATCH_SIZE) + 1
            
            out_mask = np.zeros((n_shape, NUM_INS, NUM_POINT), dtype=np.bool)
            out_valid = np.zeros((n_shape, NUM_INS), dtype=np.bool)
            out_conf = np.zeros((n_shape, NUM_INS), dtype=np.float32)
            out_label = np.zeros((n_shape, NUM_INS), dtype=np.uint8)
            
            bar = ProgressBar()
            for i in bar(range(num_batch)):
                start_idx = i * BATCH_SIZE
                end_idx = min((i + 1) * BATCH_SIZE, n_shape)

                batch_pts[:end_idx-start_idx, ...] = pts[start_idx: end_idx, ...]
                batch_record = record[start_idx: end_idx]

                feed_dict = {pc_pl: batch_pts,
                             is_training_pl: False}

                seg_pred_val, mask_pred_val, conf_pred_val = sess.run([seg_pred, mask_pred, conf_pred], feed_dict=feed_dict)

                seg_pred_val = seg_pred_val[:end_idx-start_idx, ...]    # B x N x (C+1), #0 class is *other*
                mask_pred_val = mask_pred_val[:end_idx-start_idx, ...]  # B x K x N, no other
                conf_pred_val = conf_pred_val[:end_idx-start_idx, ...]  # B x K

                mask_pred_val[mask_pred_val < FLAGS.min_point_score_to_exist] = 0
                mask_sem_val = np.matmul(mask_pred_val, seg_pred_val)   # B x K x (C+1), #0 class is *other*
                mask_sem_idx = np.argmax(mask_sem_val, axis=-1)         # B x K

                mask_hard_val = (mask_pred_val > FLAGS.min_point_score_to_exist)
                mask_valid_val = ((np.sum(mask_hard_val, axis=-1) > FLAGS.min_num_point_per_part) & (mask_sem_idx > 0))

                out_mask[start_idx: end_idx, ...] = mask_hard_val
                out_valid[start_idx: end_idx, ...] = mask_valid_val
                out_conf[start_idx: end_idx, ...] = conf_pred_val
                out_label[start_idx: end_idx, ...] = mask_sem_idx - 1

                if FLAGS.visu_dir is not None and cur_visu_batch < FLAGS.visu_batch:
                    gen_visu(start_idx, batch_pts[:end_idx-start_idx, ...], mask_hard_val, mask_valid_val, \
                            conf_pred_val, mask_sem_idx-1, batch_record)
                    cur_visu_batch += 1

            save_h5(os.path.join(PRED_DIR, item), out_mask, out_valid, out_conf, out_label)

        if FLAGS.visu_dir is not None:
            cmd = 'cd %s && python %s . 1 htmls pts,info:pred,info pts,info:pred,info' % (VISU_DIR, os.path.join(ROOT_DIR, '../utils/gen_html_hierachy_local.py'))
            log_string(cmd)
            call(cmd, shell=True)

# main
log_string('pid: %s'%(str(os.getpid())))
eval()
LOG_FOUT.close()

