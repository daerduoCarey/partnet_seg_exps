To train, check

        bash run_train.sh

To eval, check

        bash run_eval.sh


To obtain the final Mean Average Precision (mAP) number, check


        python compute_per_category_ap.py --help


The following files are for generating the *../../data/ins_seg_h5_for_detection/* from *../../data/ins_seg_h5/*

        prepare_train_val_data.py 
        prepare_test_data.py

You don't need to use them for now.
