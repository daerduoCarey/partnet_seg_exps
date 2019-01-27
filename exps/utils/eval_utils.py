import os
import sys
import h5py
import numpy as np
from progressbar import ProgressBar
from commons import check_mkdir

def load_gt_h5(fn):
    """ Output: pts             B x N x 3   float32
                gt_mask         B x K x N   bool
                gt_mask_label   B x K       uint8
                gt_mask_valid   B x K       bool
                gt_mask_other   B x N       bool
        All the ground-truth masks are represented as 0/1 mask over the 10k point cloud.
        All the ground-truth masks are disjoint, complete and corresponding to unique semantic labels.
        Different test shapes have different numbers of ground-truth masks, they are at the top in array gt_mask indicated by gt_valid.
    """
    with h5py.File(fn, 'r') as fin:
        gt_mask = fin['gt_mask'][:]
        gt_mask_label = fin['gt_mask_label'][:]
        gt_mask_valid = fin['gt_mask_valid'][:]
        gt_mask_other = fin['gt_mask_other'][:]
        return gt_mask, gt_mask_label, gt_mask_valid, gt_mask_other

def load_pred_h5(fn):
    """ Output: mask    B x K x N   bool
                label   B x K       uint8
                valid   B x K       bool
                conf    B x K       float32
        We only evaluate on the part predictions with valid = True.
        We assume no pre-sorting according to confidence score. 
    """
    with h5py.File(fn, 'r') as fin:
        mask = fin['mask'][:]
        label = fin['label'][:]
        valid = fin['valid'][:]
        conf = fin['conf'][:]
        return mask, label, valid, conf

def compute_ap(tp, fp, gt_npos, n_bins=100, plot_fn=None):
    assert len(tp) == len(fp), 'ERROR: the length of true_pos and false_pos is not the same!'

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    rec = tp / gt_npos
    prec = tp / (fp + tp)

    rec = np.insert(rec, 0, 0.0)
    prec = np.insert(prec, 0, 1.0)

    ap = 0.
    delta = 1.0 / n_bins
    
    out_rec = np.arange(0, 1 + delta, delta)
    out_prec = np.zeros((n_bins+1), dtype=np.float32)

    for idx, t in enumerate(out_rec):
        prec1 = prec[rec >= t]
        if len(prec1) == 0:
            p = 0.
        else:
            p = max(prec1)
        
        out_prec[idx] = p
        ap = ap + p / (n_bins + 1)

    if plot_fn is not None:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(out_rec, out_prec, 'b-')
        plt.title('PR-Curve (AP: %4.2f%%)' % (ap*100))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        fig.savefig(plot_fn)
        plt.close(fig)

    return ap

def eval_per_class_ap(stat_fn, gt_dir, pred_dir, iou_threshold=0.5, plot_dir=None):
    """ Input:  stat_fn contains all part ids and names 
                gt_dir contains test-xx.h5
                pred_dir contains test-xx.h5
        Output: aps: Average Prediction Scores for each part category, evaluated on all test shapes
                mAP: mean AP
    """
    print('Evaluation Start.')
    print('Ground-truth Directory: %s' % gt_dir)
    print('Prediction Directory: %s' % pred_dir)

    if plot_dir is not None:
        check_mkdir(plot_dir)

    # read stat_fn
    with open(stat_fn, 'r') as fin:
        part_name_list = [item.rstrip().split()[1] for item in fin.readlines()]
    print('Part Name List: ', part_name_list)
    n_labels = len(part_name_list)
    print('Total Number of Semantic Labels: %d' % n_labels)

    # check all h5 files
    test_h5_list = []
    for item in os.listdir(gt_dir):
        if item.startswith('test-') and item.endswith('.h5'):
            if not os.path.exists(os.path.join(pred_dir, item)):
                print('ERROR: h5 file %s is in gt directory but not in pred directory.')
                exit(1)
            test_h5_list.append(item)

    # read each h5 file and collect per-part-category true_pos, false_pos and confidence scores
    true_pos_list = [[] for item in part_name_list]
    false_pos_list = [[] for item in part_name_list]
    conf_score_list = [[] for item in part_name_list]

    gt_npos = np.zeros((n_labels), dtype=np.int32)

    for item in test_h5_list:
        print('Testing %s' % item)

        gt_mask, gt_mask_label, gt_mask_valid, gt_mask_other = load_gt_h5(os.path.join(gt_dir, item))
        pred_mask, pred_label, pred_valid, pred_conf = load_pred_h5(os.path.join(pred_dir, item))

        n_shape = gt_mask.shape[0]
        gt_n_ins = gt_mask.shape[1]
        pred_n_ins = pred_mask.shape[1]

        for i in range(n_shape):
            cur_pred_mask = pred_mask[i, ...]
            cur_pred_label = pred_label[i, :]
            cur_pred_conf = pred_conf[i, :]
            cur_pred_valid = pred_valid[i, :]
            
            cur_gt_mask = gt_mask[i, ...]
            cur_gt_label = gt_mask_label[i, :]
            cur_gt_valid = gt_mask_valid[i, :]
            cur_gt_other = gt_mask_other[i, :]

            # classify all valid gt masks by part categories
            gt_mask_per_cat = [[] for item in part_name_list]
            for j in range(gt_n_ins):
                if cur_gt_valid[j]:
                    sem_id = cur_gt_label[j]
                    gt_mask_per_cat[sem_id].append(j)
                    gt_npos[sem_id] += 1

            # sort prediction and match iou to gt masks
            cur_pred_conf[~cur_pred_valid] = 0.0
            order = np.argsort(-cur_pred_conf)

            gt_used = np.zeros((gt_n_ins), dtype=np.bool)

            for j in range(pred_n_ins):
                idx = order[j]
                if cur_pred_valid[idx]:
                    sem_id = cur_pred_label[idx]

                    iou_max = 0.0; cor_gt_id = -1;
                    for k in gt_mask_per_cat[sem_id]:
                        if not gt_used[k]:
                            # Remove points with gt label *other* from the prediction
                            # We will not evaluate them in the IoU since they can be assigned any label
                            clean_cur_pred_mask = (cur_pred_mask[idx, :] & (~cur_gt_other))

                            intersect = np.sum(cur_gt_mask[k, :] & clean_cur_pred_mask)
                            union = np.sum(cur_gt_mask[k, :] | clean_cur_pred_mask)
                            iou = intersect * 1.0 / union
                            
                            if iou > iou_max:
                                iou_max = iou
                                cor_gt_id = k
                                
                    if iou_max > iou_threshold:
                        gt_used[cor_gt_id] = True

                        # add in a true positive
                        true_pos_list[sem_id].append(True)
                        false_pos_list[sem_id].append(False)
                        conf_score_list[sem_id].append(cur_pred_conf[idx])
                    else:
                        # add in a false positive
                        true_pos_list[sem_id].append(False)
                        false_pos_list[sem_id].append(True)
                        conf_score_list[sem_id].append(cur_pred_conf[idx])

    # compute per-part-category AP
    aps = np.zeros((n_labels), dtype=np.float32)
    ap_valids = np.ones((n_labels), dtype=np.bool)
    for i in range(n_labels):
        has_pred = (len(true_pos_list[i]) > 0)
        has_gt = (gt_npos[i] > 0)

        if not has_gt:
            ap_valids[i] = False
            continue

        if has_gt and not has_pred:
            continue

        cur_true_pos = np.array(true_pos_list[i], dtype=np.float32)
        cur_false_pos = np.array(false_pos_list[i], dtype=np.float32)
        cur_conf_score = np.array(conf_score_list[i], dtype=np.float32)

        # sort according to confidence score again
        order = np.argsort(-cur_conf_score)
        sorted_true_pos = cur_true_pos[order]
        sorted_false_pos = cur_false_pos[order]

        out_plot_fn = None
        if plot_dir is not None:
            out_plot_fn = os.path.join(plot_dir, part_name_list[i].replace('/', '-')+'.png')

        aps[i] = compute_ap(sorted_true_pos, sorted_false_pos, gt_npos[i], plot_fn=out_plot_fn)

    # compute mean AP
    mean_ap = np.sum(aps * ap_valids) / np.sum(ap_valids)

    return aps, ap_valids, gt_npos, mean_ap

def eval_per_shape_mean_ap(stat_fn, gt_dir, pred_dir, iou_threshold=0.5):
    """ Input:  stat_fn contains all part ids and names 
                gt_dir contains test-xx.h5
                pred_dir contains test-xx.h5
        Output: mean_aps:       per-shape mean aps, which is the mean AP on each test shape,
                                for each shape, we only consider the parts that exist in either gt or pred
                shape_valids:   If a shape has valid parts to evaluate or not
                mean_mean_ap:   mean per-shape mean aps
    """
    print('Evaluation Start.')
    print('Ground-truth Directory: %s' % gt_dir)
    print('Prediction Directory: %s' % pred_dir)

    # read stat_fn
    with open(stat_fn, 'r') as fin:
        part_name_list = [item.rstrip().split()[1] for item in fin.readlines()]
    print('Part Name List: ', part_name_list)
    n_labels = len(part_name_list)
    print('Total Number of Semantic Labels: %d' % n_labels)

    # check all h5 files
    test_h5_list = []
    for item in os.listdir(gt_dir):
        if item.startswith('test-') and item.endswith('.h5'):
            if not os.path.exists(os.path.join(pred_dir, item)):
                print('ERROR: h5 file %s is in gt directory but not in pred directory.')
                exit(1)
            test_h5_list.append(item)

    mean_aps = []
    shape_valids = []

    # read each h5 file
    for item in test_h5_list:
        print('Testing %s' % item)

        gt_mask, gt_mask_label, gt_mask_valid, gt_mask_other = load_gt_h5(os.path.join(gt_dir, item))
        pred_mask, pred_label, pred_valid, pred_conf = load_pred_h5(os.path.join(pred_dir, item))

        n_shape = gt_mask.shape[0]
        gt_n_ins = gt_mask.shape[1]
        pred_n_ins = pred_mask.shape[1]

        for i in range(n_shape):
            cur_pred_mask = pred_mask[i, ...]
            cur_pred_label = pred_label[i, :]
            cur_pred_conf = pred_conf[i, :]
            cur_pred_valid = pred_valid[i, :]
            
            cur_gt_mask = gt_mask[i, ...]
            cur_gt_label = gt_mask_label[i, :]
            cur_gt_valid = gt_mask_valid[i, :]
            cur_gt_other = gt_mask_other[i, :]

            # per-shape evaluation
            true_pos_list = [[] for item in part_name_list]
            false_pos_list = [[] for item in part_name_list]
            gt_npos = np.zeros((n_labels), dtype=np.int32)

            # classify all valid gt masks by part categories
            gt_mask_per_cat = [[] for item in part_name_list]
            for j in range(gt_n_ins):
                if cur_gt_valid[j]:
                    sem_id = cur_gt_label[j]
                    gt_mask_per_cat[sem_id].append(j)
                    gt_npos[sem_id] += 1

            # sort prediction and match iou to gt masks
            cur_pred_conf[~cur_pred_valid] = 0.0
            order = np.argsort(-cur_pred_conf)

            gt_used = np.zeros((gt_n_ins), dtype=np.bool)

            # enumerate all pred parts
            for j in range(pred_n_ins):
                idx = order[j]
                if cur_pred_valid[idx]:
                    sem_id = cur_pred_label[idx]

                    iou_max = 0.0; cor_gt_id = -1;
                    for k in gt_mask_per_cat[sem_id]:
                        if not gt_used[k]:
                            # Remove points with gt label *other* from the prediction
                            # We will not evaluate them in the IoU since they can be assigned any label
                            clean_cur_pred_mask = (cur_pred_mask[idx, :] & (~cur_gt_other))

                            intersect = np.sum(cur_gt_mask[k, :] & clean_cur_pred_mask)
                            union = np.sum(cur_gt_mask[k, :] | clean_cur_pred_mask)
                            iou = intersect * 1.0 / union
                            
                            if iou > iou_max:
                                iou_max = iou
                                cor_gt_id = k
                                
                    if iou_max > iou_threshold:
                        gt_used[cor_gt_id] = True

                        # add in a true positive
                        true_pos_list[sem_id].append(True)
                        false_pos_list[sem_id].append(False)
                    else:
                        # add in a false positive
                        true_pos_list[sem_id].append(False)
                        false_pos_list[sem_id].append(True)

            # evaluate per-part-category AP for the shape
            aps = np.zeros((n_labels), dtype=np.float32)
            ap_valids = np.zeros((n_labels), dtype=np.bool)
            for j in range(n_labels):
                has_pred = (len(true_pos_list[j]) > 0)
                has_gt = (gt_npos[j] > 0)

                if has_pred and has_gt:
                    cur_true_pos = np.array(true_pos_list[j], dtype=np.float32)
                    cur_false_pos = np.array(false_pos_list[j], dtype=np.float32)
                    aps[j] = compute_ap(cur_true_pos, cur_false_pos, gt_npos[j])
                    ap_valids[j] = True
                elif has_pred and not has_gt:
                    aps[j] = 0.0
                    ap_valids[j] = True
                elif not has_pred and has_gt:
                    aps[j] = 0.0
                    ap_valids[j] = True

            # compute mean AP for the current shape
            if np.sum(ap_valids) > 0:
                mean_aps.append(np.sum(aps * ap_valids) / np.sum(ap_valids))
                shape_valids.append(True)
            else:
                mean_aps.append(0.0)
                shape_valids.append(False)

    # compute mean mean AP
    mean_aps = np.array(mean_aps, dtype=np.float32)
    shape_valids = np.array(shape_valids, dtype=np.bool)

    mean_mean_ap = np.sum(mean_aps * shape_valids) / np.sum(shape_valids)

    return mean_aps, shape_valids, mean_mean_ap


def eval_hier_mean_iou(gt_labels, pred_labels, tree_constraint, tree_parents):
    return eval_hier_part_mean_iou(gt_labels, pred_labels, tree_constraint, tree_parents)


def eval_hier_part_mean_iou(gt_labels, pred_labels, tree_constraint, tree_parents):
    """
        Input:  
                gt_labels           B x N x (C+1), boolean
                pred_logits         B x N x (C+1), boolean
                tree_constraint     T x (C+1), boolean
                tree_parents        T, int32
        Output: 
                mean_iou            Scalar, float32
                part_iou            C, float32
    """
    assert gt_labels.shape[0] == pred_labels.shape[0], 'ERROR: gt and pred have different num_shape'
    assert gt_labels.shape[1] == pred_labels.shape[1], 'ERROR: gt and pred have different num_point'
    assert gt_labels.shape[2] == pred_labels.shape[2], 'ERROR: gt and pred have different num_class+1'

    num_shape = gt_labels.shape[0]
    num_point = gt_labels.shape[1]
    num_class = gt_labels.shape[2] - 1
    
    assert tree_constraint.shape[0] == tree_parents.shape[0], 'ERROR: tree_constraint and tree_parents have different num_constraint'
    assert tree_constraint.shape[1] == num_class + 1 , 'ERROR: tree_constraint.shape[1] != num_class + 1'
    assert len(tree_parents.shape) == 1, 'ERROR: tree_parents is not a 1-dim array'

    # make a copy of the prediction
    pred_labels = np.array(pred_labels, dtype=np.bool)

    num_constraint = tree_constraint.shape[0]

    part_intersect = np.zeros((num_class+1), dtype=np.float32)
    part_union = np.zeros((num_class+1), dtype=np.float32)

    part_intersect[1] = np.sum(pred_labels[:, :, 1] & gt_labels[:, :, 1])
    part_union[1] = np.sum(pred_labels[:, :, 1] | gt_labels[:, :, 1])

    all_idx = np.arange(num_class+1)
    all_visited = np.zeros((num_class+1), dtype=np.bool)

    all_visited[1] = True
    for i in range(num_constraint):
        cur_pid = tree_parents[i]
        if all_visited[cur_pid]:
            cur_cons = tree_constraint[i]

            idx = all_idx[cur_cons]
            gt_other = ((np.sum(gt_labels[:, :, idx], axis=-1) == 0) & gt_labels[:, :, cur_pid])

            for j in list(idx):
                pred_labels[:, :, j] = (pred_labels[:, :, cur_pid] & pred_labels[:, :, j] & (~gt_other))
                part_intersect[j] += np.sum(pred_labels[:, :, j] & gt_labels[:, :, j])
                part_union[j] += np.sum(pred_labels[:, :, j] | gt_labels[:, :, j])
                all_visited[j] = True

    all_valid_part_ids = all_idx[all_visited]
    part_iou = np.divide(part_intersect[all_valid_part_ids], part_union[all_valid_part_ids])
    mean_iou = np.mean(part_iou)

    return mean_iou, part_iou, part_intersect[all_valid_part_ids], part_union[all_valid_part_ids]


def eval_hier_shape_mean_iou(gt_labels, pred_labels, tree_constraint, tree_parents):
    """
        Input:  
                gt_labels           B x N x (C+1), boolean
                pred_logits         B x N x (C+1), boolean
                tree_constraint     T x (C+1), boolean
                tree_parents        T, int32
        Output: 
                mean_iou            Scalar, float32
                part_iou            C, float32
    """
    assert gt_labels.shape[0] == pred_labels.shape[0], 'ERROR: gt and pred have different num_shape'
    assert gt_labels.shape[1] == pred_labels.shape[1], 'ERROR: gt and pred have different num_point'
    assert gt_labels.shape[2] == pred_labels.shape[2], 'ERROR: gt and pred have different num_class+1'

    num_shape = gt_labels.shape[0]
    num_point = gt_labels.shape[1]
    num_class = gt_labels.shape[2] - 1
    
    assert tree_constraint.shape[0] == tree_parents.shape[0], 'ERROR: tree_constraint and tree_parents have different num_constraint'
    assert tree_constraint.shape[1] == num_class + 1 , 'ERROR: tree_constraint.shape[1] != num_class + 1'
    assert len(tree_parents.shape) == 1, 'ERROR: tree_parents is not a 1-dim array'

    # make a copy of the prediction
    pred_labels = np.array(pred_labels, dtype=np.bool)

    num_constraint = tree_constraint.shape[0]

    all_idx = np.arange(num_class+1)
    all_visited = np.zeros((num_class+1), dtype=np.bool)

    all_visited[1] = True
    for i in range(num_constraint):
        cur_pid = tree_parents[i]
        if all_visited[cur_pid]:
            cur_cons = tree_constraint[i]

            idx = all_idx[cur_cons]
            gt_other = ((np.sum(gt_labels[:, :, idx], axis=-1) == 0) & gt_labels[:, :, cur_pid])

            for j in list(idx):
                pred_labels[:, :, j] = (pred_labels[:, :, cur_pid] & pred_labels[:, :, j] & (~gt_other))
                all_visited[j] = True

    all_valid_part_ids = all_idx[all_visited]

    tot = 0.0; cnt = 0;
    for i in range(num_shape):
        local_tot = 0.0; local_cnt = 0;
        for j in all_valid_part_ids:
            has_gt = (np.sum(gt_labels[i, :, j]) > 0)
            has_pred = (np.sum(pred_labels[i, :, j]) > 0)
            if has_gt or has_pred:
                intersect = np.sum(gt_labels[i, :, j] & pred_labels[i, :, j])
                union = np.sum(gt_labels[i, :, j] | pred_labels[i, :, j])
                iou = intersect * 1.0 / union
                local_tot += iou
                local_cnt += 1
        if local_cnt > 0:
            tot += local_tot / local_cnt
            cnt += 1

    return tot / cnt

