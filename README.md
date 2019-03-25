# PartNet: A Large-scale Benchmark for Fine-grained and Hierarchical Part-level 3D Object Understanding 

## Section 5.3 Instance Segmentation Code Repo

* The *data* folder contains the data. 
* The *exp* folder contains the code, under which you will find two experiment folders and two helper utils folders:
  1. *ins_seg_detecion* for PartNet Sec 5.3 Baseline Method;
  2. *ins_seg_sgpn* for SGPN Baseline;
  3. *utils* for utility functions;
  4. *tf_ops* for some customized Tensorflow layers (you may need to dive into each folder and run *tf_xxx_compile.sh* to compile the layer for your system)
* The *stats* folder contains some meta-data for the PartNet dataset that will be used by the code.

## Dataset Repo

Please check [the dataset repo](https://github.com/daerduocarey/partnet_dataset) for downloading the dataset and helper scripts for data usage.

## Citations

    @article{mo2018partnet,
        title={{PartNet}: A Large-scale Benchmark for Fine-grained and Hierarchical Part-level {3D} Object Understanding},
        author={Mo, Kaichun and Zhu, Shilin and Chang, Angel and Yi, Li and Tripathi, Subarna and Guibas, Leonidas and Su, Hao},
        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2019}
    }


Please also cite ShapeNet if you use ShapeNet models.

    @article{chang2015shapenet,
        title={Shapenet: An information-rich 3d model repository},
        author={Chang, Angel X and Funkhouser, Thomas and Guibas, Leonidas and Hanrahan, Pat and Huang, Qixing and Li, Zimo and Savarese, Silvio and Savva, Manolis and Song, Shuran and Su, Hao and others},
        journal={arXiv preprint arXiv:1512.03012},
        year={2015}
    }


