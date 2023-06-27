# News Recommendation

Implementation and modification of NRMS model for MIND dataset. Props to yflyl613 for initial implementation.

### Requirements

- python
- pytorch
- numpy
- scikit-learn
- nltk
- tqdm
- recbole (optional)


### Usage

- **Clone this repository**	

  ```bash
  git clone https://github.com/Mikezz1/NewsRecommendation.git
  cd NewsRecommendation
  ```
  
- **Prepare data**

  A scirpt `download_data.sh` is provided to download and unzip all the required data. It will create a new folder `data/` under `NewsRecommendation/`.
  
  ```bash
  # In NewsRecommendation/
  chmod +x download_data.sh
  ./download_data.sh
  ```
  
- **Start training**

  Script `run.sh` and `demo_local2.sh` are provied for model training and testing, in which you can modify parameters for the experiment. Please refer to `parameters.py` for more details.
  
  ```bash
  # In NewsRecommendation/data/
  cd ../src
  chmod +x run.sh
  
  # train
  ./run.sh train
  
  # test
  ./run.sh test <checkpoint name>
  # E.g. ./run.sh test epoch-1.pt
  ```

- **To run baseline models using Recbole**

  First, download MIND-small and convert it to atomic files
    ```bash
    cd baselines
    sh prepare_data.sh
    ```

  Then, you can modify configs in `configs` directory as you wish and run training with the following command (if you want to use default models)
    ```bash
    sh run_experiments.sh
    ```

  or this command if you wish to use custom model

    ```bash
    python train_custom.py
    ```



### Results on MIND-small validation set<sup>[1]</sup>
  
- **NRMS<sup>[3]</sup>**

  | News information |  AUC  |  MRR  | nDCG@10 |                 Configuration                 |
  | :--------------: | :---: | :---: | :-----: | :-------------------------------------------: |
  |      title       | 66.64 | 31.90 |  41.50  | batch size 128 (32*4)<br> 4 epochs<br>lr 3e-4 |



### Reference

[1] Fangzhao Wu, Ying Qiao, Jiun-Hung Chen, Chuhan Wu, Tao Qi, Jianxun Lian, Danyang Liu, Xing Xie, Jianfeng Gao, Winnie Wu and Ming Zhou. [MIND: A Large-scale Dataset for News Recommendation](https://msnews.github.io/assets/doc/ACL2020_MIND.pdf). ACL 2020.

[2] Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang, and Xing Xie. [Neural News Recommendation with Multi-Head Self-Attention](https://www.aclweb.org/anthology/D19-1671.pdf). EMNLP-IJCNLP. 2019.

