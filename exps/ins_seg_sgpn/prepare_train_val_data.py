import os
import sys
import json
import argparse
import h5py
import numpy as np
from progressbar import ProgressBar
from subprocess import call
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from commons import check_mkdir, force_mkdir

parser = argparse.ArgumentParser()
parser.add_argument('category', type=str, help='category')
parser.add_argument('level_id', type=int, help='level_id')
parser.add_argument('split', type=str, help='split train/val/test')
parser.add_argument('--num_point', type=int, default=10000, help='num_point')
parser.add_argument('--num_ins', type=int, default=200, help='num_ins')
args = parser.parse_args()

# load meta data files
stat_in_fn = '../../stats/after_merging_label_ids/%s-level-%d.txt' % (args.category, args.level_id)
print('Reading from ', stat_in_fn)
with open(stat_in_fn, 'r') as fin:
    part_name_list = [item.rstrip().split()[1] for item in fin.readlines()]
print('Part Name List: ', part_name_list)

NUM_CLASSES = len(part_name_list)
print('Semantic Labels: ', NUM_CLASSES)
NUM_INS = args.num_ins
print('Number of Instances: ', NUM_INS)
NUM_POINT = args.num_point
print('Number of Points: ', NUM_POINT)

def load_h5(fn):
    with h5py.File(fn, 'r') as fin:
        pts = fin['pts'][:]
        label = fin['label'][:]
        return pts, label

def load_json(fn):
    with open(fn, 'r') as fin:
        return json.load(fin)

def save_h5(fn, pts, semseg_one_hot, semseg_mask, insseg_one_hot, insseg_mask):
    fout = h5py.File(fn, 'w')
    fout.create_dataset('pts', data=pts, compression='gzip', compression_opts=4, dtype='float32')
    fout.create_dataset('semseg_one_hot', data=semseg_one_hot, compression='gzip', compression_opts=4, dtype='bool')
    fout.create_dataset('semseg_mask', data=semseg_mask, compression='gzip', compression_opts=4, dtype='float32')
    fout.create_dataset('insseg_one_hot', data=insseg_one_hot, compression='gzip', compression_opts=4, dtype='bool')
    fout.create_dataset('insseg_mask', data=insseg_mask, compression='gzip', compression_opts=4, dtype='float32')
    fout.close()

def convert_seg_to_one_hot(labels):
    # labels:BxN

    label_one_hot = np.zeros((labels.shape[0], labels.shape[1], NUM_CLASSES), dtype=np.bool)
    pts_label_mask = np.zeros((labels.shape[0], labels.shape[1]), dtype=np.float32)

    un, cnt = np.unique(labels, return_counts=True)
    label_count_dictionary = dict(zip(un, cnt))
    totalnum = 0
    for k_un, v_cnt in label_count_dictionary.iteritems():
        if k_un != -1:
            totalnum += v_cnt

    for idx in range(labels.shape[0]):
        for jdx in range(labels.shape[1]):
            if labels[idx, jdx] != -1:
                label_one_hot[idx, jdx, labels[idx, jdx]] = True
                pts_label_mask[idx, jdx] = float(totalnum) / float(label_count_dictionary[labels[idx, jdx]])

    return label_one_hot, pts_label_mask


def convert_groupandcate_to_one_hot(grouplabels):
    # grouplabels: BxN

    group_one_hot = np.zeros((grouplabels.shape[0], grouplabels.shape[1], NUM_INS), dtype=np.bool)
    pts_group_mask = np.zeros((grouplabels.shape[0], grouplabels.shape[1]), dtype=np.float32)

    un, cnt = np.unique(grouplabels, return_counts=True)
    group_count_dictionary = dict(zip(un, cnt))
    totalnum = 0
    for k_un, v_cnt in group_count_dictionary.iteritems():
        if k_un != -1:
            totalnum += v_cnt

    for idx in range(grouplabels.shape[0]):
        un = np.unique(grouplabels[idx])
        grouplabel_dictionary = dict(zip(un, range(len(un))))
        for jdx in range(grouplabels.shape[1]):
            if grouplabels[idx, jdx] != -1:
                group_one_hot[idx, jdx, grouplabel_dictionary[grouplabels[idx, jdx]]] = True
                pts_group_mask[idx, jdx] = float(totalnum) / float(group_count_dictionary[grouplabels[idx, jdx]])

    return group_one_hot, pts_group_mask


def reformat_data(in_h5_fn, out_h5_fn):
    # save json
    in_json_fn = in_h5_fn.replace('.h5', '.json')
    record = load_json(in_json_fn)

    out_json_fn = out_h5_fn.replace('.h5', '.json')
    cmd = 'cp %s %s' % (in_json_fn, out_json_fn)
    print cmd
    call(cmd, shell=True)

    # save h5
    pts, label = load_h5(in_h5_fn)

    # get the first NUM_POINT points
    pts = pts[:, :NUM_POINT, :]
    label = label[:, :NUM_POINT]

    n_shape = label.shape[0]

    gt_semseg_labels = np.ones((n_shape, NUM_POINT), dtype=np.int32) * (-1)
    gt_insseg_labels = np.ones((n_shape, NUM_POINT), dtype=np.int32) * (-1)

    bar = ProgressBar()
    for i in bar(range(n_shape)):
        cur_label = label[i, :NUM_POINT]
        cur_record = record[i]
        cur_tot = 0
        for item in cur_record['ins_seg']:
            if item['part_name'] in part_name_list:
                selected = np.isin(cur_label, item['leaf_id_list'])
                sem_id = part_name_list.index(item['part_name'])
                ins_id = cur_tot
                gt_semseg_labels[i, selected] = sem_id
                gt_insseg_labels[i, selected] = ins_id
                cur_tot += 1

    semseg_one_hot, semseg_mask = convert_seg_to_one_hot(gt_semseg_labels)
    insseg_one_hot, insseg_mask = convert_groupandcate_to_one_hot(gt_insseg_labels)

    save_h5(out_h5_fn, pts, semseg_one_hot, semseg_mask, insseg_one_hot, insseg_mask)

# main
data_in_dir = '../../data/ins_seg_h5/%s/' % args.category
data_out_dir = '../../data/ins_seg_h5_for_sgpn'
force_mkdir(data_out_dir)
data_out_dir = os.path.join(data_out_dir, '%s-%d' % (args.category, args.level_id))
force_mkdir(data_out_dir)

print args.category, args.level_id, args.split

h5_fn_list = []
for item in os.listdir(data_in_dir):
    if item.endswith('.h5') and item.startswith('%s-' % args.split):
        h5_fn_list.append(item)

for item in h5_fn_list:
    in_h5_fn = os.path.join(data_in_dir, item)
    out_h5_fn = os.path.join(data_out_dir, item)
    reformat_data(in_h5_fn, out_h5_fn)

