# PartNet: A Large-scale Benchmark for Fine-grained and Hierarchical Part-level 3D Object Understanding 

This repository contains code and scripts for PartNet segmentation experiments.

## About this repository

```
    data/
        sem_seg_h5/                 # the train/val/test data for Sec 5.1
        ins_seg_h5/                 # an intermediate data format for Sec 5.3
        ins_seg_h5_for_detection/   # the train/val data for our proposed method in Sec 5.3
        ins_seg_h5_for_sgpn/        # the train/val data for SGPN baseline in Sec 5.3
        ins_seg_h5_gt/              # the ground-truth test data in Sec 5.3
    exps/
        sem_seg_pointcnn            # the code for PointCNN baseline in Sec 5.1
        ins_seg_detection/          # the code for our proposed method in Sec 5.3
        ins_seg_sgpn/               # the code for SGPN baseline in Sec 5.3
        utils/                      # some utility functions
        tf_ops/                     # some customized Tensorflow layers (you may need to re-compile them on your machine)
    stats/
        all_valid_anno_info.txt         # Store all valid PartNet Annotation meta-information
                                        # <anno_id, version_id, category, shapenet_model_id, annotator_id>
        before_merging_label_ids/       # Store all expert-defined part semantics before merging
            Chair.txt
            ...
        merging_hierarchy_mapping/      # Store all merging criterion
            Chair.txt
            ...
        after_merging_label_ids/        # Store the part semantics after merging
            Chair.txt                   # all part semantics
            Chair-hier.txt              # all part semantics that are selected for Sec 5.2 experiments
            Chair-level-1.txt           # all part semantics that are selected for Sec 5.1 and 5.3 experiments for chair level-1
            Chair-level-2.txt           # all part semantics that are selected for Sec 5.1 and 5.3 experiments for chair level-2
            Chair-level-3.txt           # all part semantics that are selected for Sec 5.1 and 5.3 experiments for chair level-3
            ...
        train_val_test_split/           # An attemptive train/val/test splits (may be changed for official v1 release and PartNet challenges)
            Chair.train.json
            Chair.val.json
            Chair.test.json
```

## Dataset Repo

Please check [the dataset repo](https://github.com/daerduocarey/partnet_dataset) for downloading the dataset and helper scripts for data usage.


## Questions

Please post issues for questions and more helps on this Github repo page. For data annotation error, please fill in [this errata](https://docs.google.com/spreadsheets/d/1Q_6r9EblZdP9Grhhm2ob4u0FQ8xurAThlgK-qAcjYP0/edit?usp=sharing).


## License

MIT Licence

