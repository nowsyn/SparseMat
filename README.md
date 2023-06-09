# SparseMat
Repository for *Ultrahigh Resolution Image/Video Matting with Spatio-Temporal Sparsity*, which has been accepted by CVPR2023.

<img src="figures/framework.png" style="width:800px;" />

### Overview

Commodity ultrahigh definition (UHD) displays are becoming more affordable which demand imaging in ultrahigh resolution (UHR). This paper proposes SparseMat, a computationally efficient approach for UHR image/video matting.  Note that it is infeasible to directly process UHR images at full resolution in one shot using existing matting algorithms without running out of memory on consumer-level computational platforms, e.g., Nvidia 1080Ti with 11G memory, while patch-based approaches can introduce unsightly artifacts due to patch partitioning. Instead, our method resorts to spatial and temporal sparsity for addressing general UHR matting. When processing videos, huge computation redundancy can be reduced by exploiting spatial and temporal sparsity. In this paper, we show how to effectively detect spatio-temporal sparsity, which serves as a gate to activate input pixels for the matting model. Under the guidance of such sparsity, our method with sparse high-resolution module (SHM) can avoid patch-based inference while memory efficient for full-resolution matte refinement. Extensive experiments demonstrate that SparseMat can effectively and efficiently generate high-quality alpha matte for UHR images and videos at the original high resolution in a single pass.

### Environment
The recommended pytorch and torchvision version is v1.9.0 and v0.10.0.

- torch
- torchvision
- easydict
- toml
- pillow
- scikit-image
- scipy
- spconv. Please install sparse conv module refer to [traveller59/spconv](https://github.com/traveller59/spconv/tree/v1.2.1). Note that we use version 1.2.1 instead of the latest version.  

### Dataset
Existing datasets suffer from limited resolution. Thus, in this paper we contribute the first UHR human matting dataset, composed of HHM50K for training and HHM2K for evaluation. HHM50K and HHM2K consist of respectively 50,000 and 2,000 unique UHR images (with an average resolution of 4K) encompassing a wide range of human poses and matting scenarios. We provide the downloading link below.
- HHM50K: [BaiduDisk](https://pan.baidu.com/s/1txjXk7OH3vIH7yrmpfNThA), password 2tsc
- HHM2K: [BaiduDisk](https://pan.baidu.com/s/1RKu3qJRRMlgfZbIN7P4j4w), password ymyr

You can download and put them under `data` directory. Then run the following command to generate file lists.
```
python3 data/generate_filelist.py
```

### Code
###### Training
Run the following command to train the model. To train SparseMat with our self-trained low-resolution prior network, please download [here](https://drive.google.com/file/d/1_zDQbul-lCM-tFEWNcdw0D4jr3WaK1ir/view?usp=sharing) and put it under the `pretrained` directory.
```
work_dir=/PATH/TO/SparseMat
cd $work_dir
export PYTHONPATH=$PYTHONPATH:$work_dir
python3 train.py -c configs/sparsemat.toml
```

###### Testing
Run the following command to evalute the model. You can download our pretrained model [here](https://drive.google.com/file/d/19MX3USM4BK3sYi0o3AHNUxJ8bZEAGXg9/view?usp=sharing) and put it under the `pretrained` directory.
```
work_dir=/PATH/TO/SparseMat
cd $work_dir
export PYTHONPATH=$PYTHONPATH:$work_dir
python3 test.py -c configs/sparsemat.toml
```

###### Inference
You can use the following command to inference the model on images or videos.
```
work_dir=/PATH/TO/SparseMat
cd $work_dir
export PYTHONPATH=$PYTHONPATH:$work_dir
python3 demo.py -c configs/sparsemat.toml --input <INPUT_PATH> --save_dir <SAVE_DIR>
```

### Reference
```
@InProceedings{Sun_2023_CVPR,
    author    = {Sun, Yanan and Tang, Chi-Keung and Tai, Yu-Wing},
    title     = {Ultrahigh Resolution Image/Video Matting With Spatio-Temporal Sparsity},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {14112-14121}
}
```
