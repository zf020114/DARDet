# DARDet
**PyTorch implementation of "*DARDet: A Dense Anchor-free Rotated Object Detector in Aerial Images*",  [<a href="https://raw.github.com/zf020114/DARDet/master/Figs/GRSL.pdf">pdf</a>].**<br><br>


## *Highlights:*
#### 1. *We develop a new dense anchor-free rotated object detection architecture (DARDet), which directly predicts five parameters of OBB at each spatial location.*
  <p align="center"> <img src="https://raw.github.com/zf020114/DARDet/master/Figs/framework.png" width="100%"></p>
 
#### 2. *Our DARDet significantly achieve state-of-the-art performance on the DOTA, UCAS-AOD, and HRSC2016 datasets with high efficiency..*
  <p align="center"> <img src="https://raw.github.com/zf020114/DARDet/master/Figs/table.png" width="100%"></p>
  <p align="center"> <img src="https://raw.github.com/zf020114/DARDet/master/Figs/result.png" width="100%"></p>


## Benchmark and model zoo

|Model          |    Backbone     |    MS  |  Rotate | Lr schd  | Inf time (fps) | box AP| Download|
|:-------------:| :-------------: | :-----:| :-----: | :-----:  | :------------: | :----: | :---------------------------------------------------------------------------------------: |
|DARDet         |    R-50-FPN     |   -     |  -     |   1x     |      12.7      |  77.61 | [cfg](configs/ReDet/dardet_r50_fpn_1x_dcn_test.py)[model](https://pan.baidu.com/s/1aspypaz8a7QvFyUdDR986g)    |
|DARDet         |    R-50-FPN     |   -     |  ✓    |   2x     |      12.7      |  78.74 |  [cfg](configs/ReDet/dardet_r50_fpn_1x_dcn_rotate_test.py)[model](https://pan.baidu.com/s/1VPsAB3Kb90IqJTluH6lFHw)     |


## Installation
## Prerequisites

- Linux or macOS (Windows is in experimental support)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

The compatible MMDetection and MMCV versions are as below. Please install the correct version of MMCV to avoid installation issues.

| MMDetection version |    MMCV version     |
|:-------------------:|:-------------------:|
| 2.13.0              | mmcv-full>=1.3.3, <1.4.0 |


Note: You need to run `pip uninstall mmcv` first if you have mmcv installed.
If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.

## Installation

0. You can simply install mmdetection with the following commands:
    `pip install mmdet`

1. Create a conda virtual environment and activate it.

    ```shell
    conda create -n open-mmlab python=3.7 -y
    conda activate open-mmlab
    ```

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

    ```shell
    conda install pytorch torchvision -c pytorch
    ```

    Note: Make sure that your compilation CUDA version and runtime CUDA version match.
    You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

    `E.g.1` If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install
    PyTorch 1.5, you need to install the prebuilt PyTorch with CUDA 10.1.

    ```shell
    conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
    ```

3. Install mmcv-full, we recommend you to install the pre-build package as below.

    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
    ```

    Please replace `{cu_version}` and `{torch_version}` in the url to your desired one. For example, to install the latest `mmcv-full` with `CUDA 11` and `PyTorch 1.7.0`, use the following command:

    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
    ```

    See [here](https://github.com/open-mmlab/mmcv#install-with-pip) for different versions of MMCV compatible to different PyTorch and CUDA versions.
    Optionally you can choose to compile mmcv from source by the following command

    ```shell
    git clone https://github.com/open-mmlab/mmcv.git
    cd mmcv
    MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full will be installed after this step
    cd ..
    ```

    Or directly run

    ```shell
    pip install mmcv-full
    ```

4. Clone the DARDet repository.

    cd DARDet
    ```

5. Install build requirements and then install DARDet

    ```shell
    pip install -r requirements/build.txt
    pip install -v -e .  # or "python setup.py develop"

6. Install DOTA_devkit
    ```
    sudo apt-get install swig
    cd DOTA_devkit/polyiou
    swig -c++ -python csrc/polyiou.i
    python setup.py build_ext --inplace
    ```

## Prepare DOTA dataset.
    It is recommended to symlink the dataset root to `ReDet/data`.

    Here, we give an example for single scale data preparation of DOTA-v1.5.

    First, make sure your initial data are in the following structure.
    ```
    data/dota15
    ├── train
    │   ├──images
    │   └── labelTxt
    ├── val
    │   ├── images
    │   └── labelTxt
    └── test
        └── images
    ```
    Split the original images and create COCO format json. 
    ```
    python DOTA_devkit/prepare_dota1_5.py --srcpath path_to_dota --dstpath path_to_split_1024
    ```
    Then you will get data in the following structure
    ```
    dota15_1024
    ├── test1024
    │   ├── DOTA_test1024.json
    │   └── images
    └── trainval1024
        ├── DOTA_trainval1024.json
         └── images
    ```
    For data preparation with data augmentation, refer to "DOTA_devkit/prepare_dota1_5_v2.py"


Examples:

Assume that you have already downloaded the checkpoints to `work_dirs/DARDet_r50_fpn_1x/`.

* Test DARDet on DOTA.

```shell
python tools/test.py configs/DARDet/dardet_r50_fpn_1x_dcn_val.py \
    work_dirs/dardet_r50_fpn_1x_dcn_val/epoch_12.pth \ 
    --out work_dirs/dardet_r50_fpn_1x_dcn_val/res.pkl
```
*If you want to evaluate the result on DOTA test-dev, zip the files in ```work_dirs/dardet_r50_fpn_1x_dcn_val/result_after_nms``` and submit it to the  [evaluation server](https://captain-whu.github.io/DOTA/index.html).


## Inference
To inference multiple images in a folder, you can run:

```
python demo/demo_inference.py ${CONFIG_FILE} ${CHECKPOINT} ${IMG_DIR} ${OUTPUT_DIR}
```

## Train a model

MMDetection implements distributed training and non-distributed training,
which uses `MMDistributedDataParallel` and `MMDataParallel` respectively.

All outputs (log files and checkpoints) will be saved to the working directory,
which is specified by `work_dir` in the config file.

**\*Important\***: The default learning rate in config files is for 8 GPUs and 2 img/gpu (batch size = 8*2 = 16).
According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you need to set the learning rate proportional to the batch size if you use different GPUs or images per GPU, e.g., lr=0.01 for 4 GPUs * 2 img/gpu and lr=0.08 for 16 GPUs * 4 img/gpu.

### Train with a single GPU

```shell
python tools/train.py ${CONFIG_FILE}
```

If you want to specify the working directory in the command, you can add an argument `--work_dir ${YOUR_WORK_DIR}`.

### Train with multiple GPUs

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Optional arguments are:

- `--validate` (**strongly recommended**): Perform evaluation at every k (default value is 1, which can be modified like [this](../configs/mask_rcnn_r50_fpn_1x.py#L174)) epochs during the training.
- `--work_dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume_from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.

Difference between `resume_from` and `load_from`:
`resume_from` loads both the model weights and optimizer status, and the epoch is also inherited from the specified checkpoint. It is usually used for resuming the training process that is interrupted accidentally.
`load_from` only loads the model weights and the training epoch starts from 0. It is usually used for finetuning.

### Train with multiple machines

If you run MMDetection on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script `slurm_train.sh`.

```shell
./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR} [${GPUS}]
```

Here is an example of using 16 GPUs to train Mask R-CNN on the dev partition.

```shell
./tools/slurm_train.sh dev mask_r50_1x configs/mask_rcnn_r50_fpn_1x.py /nfs/xxxx/mask_rcnn_r50_fpn_1x 16
```

You can check [slurm_train.sh](../tools/slurm_train.sh) for full arguments and environment variables.

If you have just multiple machines connected with ethernet, you can refer to
pytorch [launch utility](https://pytorch.org/docs/stable/distributed_deprecated.html#launch-utility).
Usually it is slow if you do not have high speed networking like infiniband.


## Contact
**Any question regarding this work can be addressed to [zhangfeng01@nudt.edu.cn](zhangfeng01@nudt.edu.cn).**
