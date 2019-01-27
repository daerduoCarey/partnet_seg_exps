import argparse
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from eval_utils import eval_per_class_ap

parser = argparse.ArgumentParser()
parser.add_argument('category', type=str, help='Category name [default: Chair]')
parser.add_argument('level_id', type=int, help='Level ID [default: 3]')
parser.add_argument('pred_dir', type=str, help='log prediction directory [default: log]')
parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU Threshold [default: 0.5]')
parser.add_argument('--plot_dir', type=str, default=None, help='PR Curve Plot Output Directory [default: None, meaning no output]')
FLAGS = parser.parse_args()

stat_in_fn = '../../stats/after_merging2_label_ids/%s-level-%d.txt' % (FLAGS.category, FLAGS.level_id)
with open(stat_in_fn, 'r') as fin:
    part_name_list = [item.rstrip().split()[1] for item in fin.readlines()]
gt_in_dir = '../../data/ins_seg_h5_gt/%s-%d/' % (FLAGS.category, FLAGS.level_id)
PRED_DIR = FLAGS.pred_dir
       
# Compute AP and mAP
aps, ap_valids, gt_npos, mean_ap = eval_per_class_ap(stat_in_fn, gt_in_dir, PRED_DIR, iou_threshold=FLAGS.iou_threshold, plot_dir=FLAGS.plot_dir)
n_labels = len(part_name_list)

print('Computing AP and mAP')
for i in range(n_labels):
    if ap_valids[i]:
        print('%s %d %f' % (part_name_list[i], gt_npos[i], aps[i]))
print('\nmAP %f' % mean_ap)


