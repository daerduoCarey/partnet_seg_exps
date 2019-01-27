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
parser.add_argument('--num_point', type=int, default=10000, help='num_point')
args = parser.parse_args()

NUM_POINT = args.num_point
print('Number of Points: ', NUM_POINT)

def load_h5(fn):
    with h5py.File(fn, 'r') as fin:
        pts = fin['pts'][:]
        return pts

def load_json(fn):
    with open(fn, 'r') as fin:
        return json.load(fin)

def save_h5(fn, pts):
    fout = h5py.File(fn, 'w')
    fout.create_dataset('pts', data=pts, compression='gzip', compression_opts=4, dtype='float32')
    fout.close()

def reformat_data(in_h5_fn, out_h5_fn):
    # save json
    in_json_fn = in_h5_fn.replace('.h5', '.json')
    record = load_json(in_json_fn)

    out_json_fn = out_h5_fn.replace('.h5', '.json')
    cmd = 'cp %s %s' % (in_json_fn, out_json_fn)
    print cmd
    call(cmd, shell=True)

    # save h5
    pts = load_h5(in_h5_fn)
    print 'pts: ', pts.shape

    # get the first NUM_POINT points
    pts = pts[:, :NUM_POINT, :]

    save_h5(out_h5_fn, pts)

# main
data_in_dir = '../../data/ins_seg_h5/%s/' % args.category
data_out_dir = '../../data/ins_seg_h5_for_detection'
force_mkdir(data_out_dir)
data_out_dir = os.path.join(data_out_dir, '%s-%d' % (args.category, args.level_id))
force_mkdir(data_out_dir)

print args.category, args.level_id, 'test'

h5_fn_list = []
for item in os.listdir(data_in_dir):
    if item.endswith('.h5') and item.startswith('test-'):
        h5_fn_list.append(item)

for item in h5_fn_list:
    in_h5_fn = os.path.join(data_in_dir, item)
    out_h5_fn = os.path.join(data_out_dir, item)
    reformat_data(in_h5_fn, out_h5_fn)

