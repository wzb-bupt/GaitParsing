# GaitParsing

Official code for "[GaitParsing: Human Semantic Parsing for Gait Recognition](https://ieeexplore.ieee.org/document/10288081)" (IEEE TMM).

<div align="left">
<a href="https://ieeexplore.ieee.org/document/10288081"><img src="https://img.shields.io/badge/Paper Link-GaitParsing-red"></a>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://visitor-badge.laobi.icu/badge?page_id=wzb-bupt/GaitParsing" alt="Visitors">
</div>

## Quick Start
### 1. Clone this repo.
    
  ```
  git clone git@github.com:wzb-bupt/GaitParsing.git
  ```
    
### 2. Prepare the environments 
  - pytorch >= 1.10
  - torchvision
  - pyyaml
  - tensorboard
  - opencv-python
  - tqdm
  - py7zr
  - kornia
  - einops
  - six

  Install dependenices by [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html):
  ```
  conda install tqdm pyyaml tensorboard opencv kornia einops six -c conda-forge
  conda install pytorch==1.10 torchvision -c pytorch
  ```    
  Or, Install dependencies by pip:
  ```
  pip install tqdm pyyaml tensorboard opencv-python kornia einops six
  pip install torch==1.10 torchvision==0.11
  ```
### 3. Prepare the datasets (with RGB modality)
  
  _e.g.,_ [CCPG](https://github.com/BNU-IVC/CCPG), [CASIA-B*](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp), [FVG](https://cvlab.cse.msu.edu/frontal-view-gaitfvg-database.html), and [CCVID](https://github.com/guxinqian/Simple-CCReID) datasets.
  
  Waiting for more valuable gait datasets ...
  
### 4. Download the **pre-trained parsing model** from [Baidu Netdisk](https://pan.baidu.com/s/1G7NlZ4MIKfEHWiTDxZtKXQ?pwd=yyds) or [Google Drive](https://drive.google.com/file/d/1GYhzbQmWO80ZsiNO9D0XyuWdD9OjQCJD/view?usp=drive_link) and place it in the root.
  
  ```
  parsing_u2net.pth
  ```
  
### 5. Run the simple script:

 For a single folder (We provide three examples in [./images](images), and the output folder is [./results](results))
  
  ```
  python demo_for_single_folder.py
  ```

 For a dataset (_e.g.,_ CCPG DataSet, you can modify this for other datasets.)
 
  ```
  python demo_for_CCPG_Parsing.py
  ```

## If you find our parsing model useful in your research, please consider citing:

  ```
  @article{wang2023gaitparsing,
    title={GaitParsing: Human Semantic Parsing for Gait Recognition},
    author={Wang, Zengbin and Hou, Saihui and Zhang, Man and Liu, Xu and Cao, Chunshui and Huang, Yongzhen},
    journal={IEEE Transactions on Multimedia},
    year={2023},
    publisher={IEEE}
  }
  ```
