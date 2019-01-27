import argparse
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from eval_utils import eval_per_shape_mean_ap

parser = argparse.ArgumentParser()
parser.add_argument('category', type=str, help='Category name [default: Chair]')
parser.add_argument('level_id', type=int, help='Level ID [default: 3]')
parser.add_argument('pred_dir', type=str, help='log prediction directory [default: log]')
parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU Threshold [default: 0.5]')
FLAGS = parser.parse_args()

stat_in_fn = '../../stats/after_merging2_label_ids/%s-level-%d.txt' % (FLAGS.category, FLAGS.level_id)
with open(stat_in_fn, 'r') as fin:
    part_name_list = [item.rstrip().split()[1] for item in fin.readlines()]
gt_in_dir = '../../data/ins_seg_h5_gt/%s-%d/' % (FLAGS.category, FLAGS.level_id)
PRED_DIR = FLAGS.pred_dir
       
# Compute AP and mAP
shape_mean_aps, shape_valids, mean_mean_ap = eval_per_shape_mean_ap(stat_in_fn, gt_in_dir, PRED_DIR, iou_threshold=FLAGS.iou_threshold)

print('\nmean mean AP %f' % mean_mean_ap)


