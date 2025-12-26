# Hi-LOAM
<p align="center">
  <h1 align="center">✨ Hi-LOAM: Hierarchical Implicit Neural Fields for LliDAR Odometry and Mapping</h1>
  <p align="center">
    <strong>Zhiliu Yang</strong></a>
    ·
    <strong>Jianyuan Zhang</strong></a>
    ·
   <strong>Lianhui Zhao</strong></a>
    ·
   <strong>Jinyu Dai</strong></a>
    ·
   <strong>Zhu Yang</strong></a>
    
   
  </p>
<p align="center">
  <img src="https://github.com/Zhangjyhhh/Hi-LOAM/blob/main/OverviewHiLOAMV5.png" width="100%" />
</p>







## Abstract
LiDAR Odometry and Mapping (LOAM) is a pivotal
technique for embodied-AI applications such as autonomous driving and robot navigation. Existing LOAM frameworks either rely
on the explicit representation, restricted to the supervision signal,or lack of the reconstruction fidelity, which are deficient in representing large-scale complex scenes. To overcome these limitations, we propose a multi-scale implicit neural localization and mapping
framework using LiDAR sensor, called Hi-LOAM. Hi-LOAM
receives LiDAR point cloud as the input data modality, learns
and stores hierarchical latent features in multiple levels of hash
tables based on an octree structure, then these multi-scale latent
features are decoded into signed distance value through shallow
Multilayer Perceptrons (MLPs) in the mapping procedure. For
pose estimation procedure, we rely on a correspondence-free,
scan-to-implicit matching paradigm to estimate optimal pose
and register current scan into the submap. The entire training
process is conducted in a self-supervised manner, which waives
the model pre-training and manifests its generalizability when
applied to diverse environments. Extensive experiments on multiple real-world and synthetic datasets demonstrate the superior
performance, in terms of the effectiveness and generalization
capabilities, of our Hi-LOAM compared to existing state-of-the-art methods

----

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#abstract">Abstract</a>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#prepare-data">Data</a>
    </li>
    <li>
      <a href="#run">How to run</a>
    </li>
    <li>
      <a href="#evaluation">Evaluation</a>
    </li>
    <li>
      <a href="#acknowledgment">Acknowledgment</a>
    </li>
  </ol>
</details>

----

## 1.Installation

Platform requirement
* Ubuntu OS (tested on 18.04)
* With GPU (recommended),GPU memory requirement (> 8 GB and the more, the better)


### 1.1 Clone Hi-LOAM repository
```
git clone https://github.com/Zhangjyhhh/Hi-LOAM.git
cd Hi-LOAM
```
### 1.2 Set up conda environment
```
conda create --name Hi-LOAM python=3.7
conda activate Hi-LOAM
```
### 1.3 Install the key requirement kaolin

Kaolin depends on Pytorch (>= 1.8, <= 1.13.1), please install the corresponding Pytorch for your CUDA version (can be checked by ```nvcc --version```). You can find the installation commands [here](https://pytorch.org/get-started/previous-versions/).

For example, for CUDA version >=11.6, you can use:
```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

Kaolin now supports installation with wheels. For example, to install kaolin 0.12.0 over torch 1.12.1 and cuda 11.6:
```
pip install kaolin==0.12.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.1_cu116.html
```

<details>
  <summary>[Or you can build kaolin by yourself (click to expand)]</summary>

Follow the [instructions](https://kaolin.readthedocs.io/en/latest/notes/installation.html) to install [kaolin](https://kaolin.readthedocs.io/en/latest/index.html). Firstly, clone kaolin to a local directory:

```
git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
cd kaolin
```

Then install kaolin by:
```
python setup.py develop
```

Use ```python -c "import kaolin; print(kaolin.__version__)"``` to check if kaolin is successfully installed.
</details>


### 1.4 Install the other requirements
```
pip install open3d scikit-image wandb tqdm natsort pyquaternion
```
----


## Download data

Generally speaking, you only need to provide:
* `pc_path` : the folder containing the point cloud (`.bin`, `.ply` or `.pcd` format) for each frame.
* `calib_path` : the calib file (`.txt`) containing the static transformation between sensor and body frames (optional, would be identity matrix if set as `''`).

Datasets are available for download on their official websites. you may run the script in Folder A to download test samples for program testing.

After preparing the data, you need to correctly set the data path (`pc_path` and `calib_path`) in the config files under `config` folder. You may also set a path `output_root` to store the experiment results and logs.




##  How to Run

we show some example run commands for one scene from each dataset. After SLAM, the trajectory error will be evaluated along with the rendering metrics. The results will be saved to `./experiments` by default.

*To run KITTI dataset, please use the following command:
```
python Hi_LOAM.py ./config/kitti/kitti_general.yaml
```

*To run Mai City dataset, please use the following command:
```
python Hi_LOAM.py ./config/maicity/maicity_general.yaml
```

*To run Newer College dataset, please use the following command:
```
python Hi_LOAM.py ./config/ncd/ncd_general.yaml
```

*To run Hilti dataset, please use the following command:
```
python Hi_LOAM.py ./config/hilti/hilti_general.yaml
```

## Evaluation

* To evaluate the reconstruction quality, you need to provide the (reference) ground truth point cloud and your reconstructed mesh. The ground truth point cloud can be found (or sampled from) the downloaded folder of MaiCity and Newer College . 
Please change the data path and evaluation set-up in `./eval/evaluator.py` and then run:

```
python ./eval/evaluator.py
```

* To evaluate the odometry, we use EVO tools(https://github.com/MichaelGrupp/evo). For Hilti-21 dataset, we use this tool (https://github.com/Hilti-Research/hilti-slam-challenge-2021) to evaluate our odometry



## Acknowledgment

Additionally, we thank greatly for the authors of the following opensource projects:

- [NGLOD](https://github.com/nv-tlabs/nglod) 
- [PIN-SLAM](https://github.com/PRBonn/PIN_SLAM)


