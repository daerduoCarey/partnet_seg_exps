#!/usr/bin/python3
"""Testing On Segmentation Task.
Modified by Kaichun Mo to run experiments on PartNet"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import h5py
import argparse
import importlib
import data_utils
import numpy as np
import tensorflow as tf
from datetime import datetime
from progressbar import ProgressBar

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', '-c', help='category name', required=True)
    parser.add_argument('--level', '-l', type=int, help='level id', required=True)
    parser.add_argument('--load_ckpt', '-k', help='Path to a check point file for load', required=True)
    parser.add_argument('--model', '-m', help='Model to use', required=True)
    parser.add_argument('--setting', '-x', help='Setting to use', required=True)
    parser.add_argument('--batch_size', '-b', help='Batch size during testing', default=4, type=int)
    parser.add_argument('--save_ply', '-s', help='Save results as ply', action='store_true')
    parser.add_argument('--save_dir', '-o', help='The output directory', type=str, default=None)
    parser.add_argument('--save_num_shapes', '-u', help='how many shapes to visualize', default=20, type=int)
    args = parser.parse_args()
    print(args)

    if args.save_ply:
        if os.path.exists(args.save_dir):
            print('ERROR: folder %s exists! Please check and delete!' % args.save_dir)
            exit(1)
        os.mkdir(args.save_dir)

    model = importlib.import_module(args.model)
    setting_path = os.path.join(os.path.dirname(__file__), args.model)
    sys.path.append(setting_path)
    setting = importlib.import_module(args.setting)

    sample_num = setting.sample_num
    batch_size = args.batch_size

    args.data_folder = '../../data/partnet_sem_seg/'

    # Load all test data
    args.filelist = os.path.join(args.data_folder, '%s-%d' % (args.category, args.level), 'test_files.txt')
    data_test, _, label_gt = data_utils.load_seg(args.filelist)
    num_shape = data_test.shape[0]
    print('Loaded data: %s shapes in total to test.' % num_shape)

    # Load current category + level statistics
    with open('../../data/partnet_stats/%s-level-%d.txt' % (args.category, args.level), 'r') as fin:
        setting.num_class = len(fin.readlines()) + 1    # with "other"
        print('NUM CLASS: %d' % setting.num_class)


    ######################################################################
    # Placeholders
    is_training = tf.placeholder(tf.bool, name='is_training')
    pts_fts = tf.placeholder(tf.float32, shape=(batch_size, sample_num, setting.data_dim), name='points')
    ######################################################################

    ######################################################################
    pts_fts_sampled = pts_fts
    points_sampled = pts_fts_sampled
    features_sampled = None

    net = model.Net(points_sampled, features_sampled, is_training, setting)
    seg_probs_op = tf.nn.softmax(net.logits, name='seg_probs')

    # for restore model
    saver = tf.train.Saver()

    parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    print('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    # Load the model
    ckptstate = tf.train.get_checkpoint_state(args.load_ckpt)
    if ckptstate is not None:
        LOAD_MODEL_FILE = os.path.join(args.load_ckpt, os.path.basename(ckptstate.model_checkpoint_path))
        saver.restore(sess, LOAD_MODEL_FILE)
        print("Model loaded in file: %s" % LOAD_MODEL_FILE)
    else:
        print("Fail to load modelfile: %s" % args.load_ckpt)

    # Start the testing
    print('{}-Testing...'.format(datetime.now()))

    num_batch = (num_shape - 1) // batch_size + 1
    pts_batch = np.zeros((batch_size, sample_num, 3), dtype=np.float32)
    
    avg_acc = 0.0; avg_cnt = 0;

    shape_iou_tot = 0.0; shape_iou_cnt = 0;
    
    part_intersect = np.zeros((setting.num_class), dtype=np.float32)
    part_union = np.zeros((setting.num_class), dtype=np.float32)

    bar = ProgressBar()
    all_seg_probs = []
    for batch_idx in bar(range(num_batch)):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_shape)

        pts_batch[: end_idx - start_idx, ...] = data_test[start_idx: end_idx]

        seg_probs = sess.run(seg_probs_op, feed_dict = {pts_fts: pts_batch, is_training: False})
        seg_probs = seg_probs[: end_idx - start_idx]
        all_seg_probs.append(seg_probs)

        seg_res = np.argmax(seg_probs[:, :, 1:], axis=-1) + 1

        avg_acc += np.sum(np.mean((seg_res == label_gt[start_idx: end_idx]) | (label_gt[start_idx: end_idx] == 0), axis=-1))
        avg_cnt += end_idx - start_idx

        seg_gt = label_gt[start_idx: end_idx]
        seg_res[seg_gt==0] = 0

        for i in range(end_idx - start_idx):
            cur_pred = seg_res[i]
            cur_gt = seg_gt[i]

            cur_shape_iou_tot = 0.0; cur_shape_iou_cnt = 0;
            for j in range(1, setting.num_class):
                cur_gt_mask = (cur_gt == j)
                cur_pred_mask = (cur_pred == j)

                has_gt = (np.sum(cur_gt_mask) > 0)
                has_pred = (np.sum(cur_pred_mask) > 0)

                if has_gt or has_pred:
                    intersect = np.sum(cur_gt_mask & cur_pred_mask)
                    union = np.sum(cur_gt_mask | cur_pred_mask)
                    iou = intersect / union

                    cur_shape_iou_tot += iou
                    cur_shape_iou_cnt += 1

                    part_intersect[j] += intersect
                    part_union[j] += union

            if cur_shape_iou_cnt > 0:
                cur_shape_miou = cur_shape_iou_tot / cur_shape_iou_cnt
                shape_iou_tot += cur_shape_miou
                shape_iou_cnt += 1

        if args.save_ply and start_idx < args.save_num_shapes:
            for i in range(start_idx, min(end_idx, args.save_num_shapes)):
                out_fn = os.path.join(args.save_dir, 'shape-%02d-pred.ply' % i)
                data_utils.save_ply_property(data_test[i], seg_res[i-start_idx], setting.num_class, out_fn)
                out_fn = os.path.join(args.save_dir, 'shape-%02d-gt.ply' % i)
                data_utils.save_ply_property(data_test[i], label_gt[i], setting.num_class, out_fn)

    all_seg_probs = np.vstack(all_seg_probs)
    np.save('out.npy', all_seg_probs)

    print('{}-Done!'.format(datetime.now()))

    print('Average Accuracy: %f' % (avg_acc / avg_cnt))
    print('Shape mean IoU: %f' % (shape_iou_tot / shape_iou_cnt))

    part_iou = np.divide(part_intersect[1:], part_union[1:])
    mean_part_iou = np.mean(part_iou)
    print('Category mean IoU: %f, %s' % (mean_part_iou, str(part_iou)))

    out_list = ['%3.1f' % (item*100) for item in part_iou.tolist()]
    print('%3.1f;%3.1f;%3.1f;%s' % (avg_acc*100 / avg_cnt, shape_iou_tot*100 / shape_iou_cnt, mean_part_iou*100, '['+', '.join(out_list)+']'))


if __name__ == '__main__':
    main()
