import os
import sys
import json

in_dir = '../../data/ins_seg_h5/'
out_dir = '../../stats/train_num_ins'

in_cat = sys.argv[1]
in_dir = os.path.join(in_dir, in_cat)

tot = 0
for item in os.listdir(in_dir):
    if item.endswith('.json') and item.startswith('train-'):
        with open(os.path.join(in_dir, item), 'r') as fin:
            tot += len(json.load(fin))

print in_cat, tot

out_fn = os.path.join(out_dir, in_cat+'.txt')
with open(out_fn, 'w') as fout:
    fout.write('%d'%tot)

