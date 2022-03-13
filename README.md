# GraphIQA
GraphIQA: Learning Distortion Graph Representations for Blind Image Quality Assessment
![image](https://github.com/geekyutao/GraphIQA/blob/main/fig/framework.png)
> [**GraphIQA**](https://arxiv.org/abs/2103.07666),            
> **Simeng Sun***, **Tao Yu*** (Equal contribution), Jiahua Xu, Wei Zhou, Zhibo Chen,        
> *arXiv preprint ([arXiv:2103.07666](https://arxiv.org/abs/2103.07666))*  
> accepted to IEEE TMM 2022

## Abstract
Learning-based blind image quality assessment (BIQA) methods have recently drawn much attention for their superior performance compared to traditional methods. However, most of them do not effectively leverage the relationship between distortion-related factors, showing limited feature representation capacity. In this paper, we show that human perceptual quality is highly correlated with distortion type and degree, and is biased by image content in most IQA systems. Based on this observation, we propose a Distortion Graph Representation (DGR) learning framework for IQA, called GraphIQA. In GraphIQA, each distortion is represented as a graph i.e. DGR. One can distinguish distortion types by comparing different DGRs, and predict image quality by learning the relationship between different distortion degrees in DGR. Specifically, we develop two sub-networks to learn the DGRs: a) Type Discrimination Network (TDN) that embeds DGR into a compact code for better discriminating distortion types; b) Fuzzy Prediction Network (FPN) that extracts the distributional characteristics of the samples in DGR and predicts fuzzy degrees based on a Gaussian prior. Experiments show that our GraphIQA achieves the state-of-the-art performance on many benchmark datasets of both synthetic and authentic distortion.

## Preparation
This project is run on GPU (NVIDIA RTX 2080Ti) with CUDA 10.0 and CPU (Intel(R) Xeon(R) Silver 4110 CPU @ 2.10GHz).

1. [Optional but recommended] Create a env environment using miniconda, run:
    ~~~
    conda create --name GraphIQA python ==3.7.7
    source activate GraphIQA
    ~~~
2. Install pytorch 1.5.0:
    ~~~
    conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch
    ~~~
3. Install required packages by running:
    ~~~
    pip install -r requirements.txt
    ~~~
4. Download [Kadis700k](http://database.mmsp-kn.de/kadid-10k-database.html) dataset and generate the distorted images according to the official guidance, then restore them in `./data/kadis700k/dist_imgs`.
Doneload [LIVE](http://live.ece.utexas.edu/research/Quality/), [LIVEC](http://live.ece.utexas.edu/research/Quality/), [KonIQ10k](http://database.mmsp-kn.de/koniq-10k-database.html) and [CSIQ](https://computervisiononline.com/dataset/1105138666) in `./data/databaserelease2`, `./data/ChallengeDB`, `./data/KonIQ` and `./data/csiq` respectively.

## Usages
When pretraining GraphIQA model from scratch on kadid10k dataset, you need to run:
~~~
python pretrain.py --dataset kadid-P
~~~
When pretraining GraphIQA model from scratch on kadis700k dataset, you need to run:
~~~
python pretrain.py --dataset kadis-P
~~~
Some available options:
* `--dataset`: Training and testing dataset, support datasets: kadid-P | kadis-P
* `--gpus`: The number of required gpus. For pretraining on kadis700k dataset, up to 4 gpus are needed.
* `--gpu_ids`: Visible devices.
* `--bs`: Batch size.
* `--margin`: Margin for triplet loss.

The pretrained model is released [Model](https://drive.google.com/file/d/1mvnfFC4v7P80KhcN2IWCAdQa3y20B_g_/view?usp=sharing).

When finetuning GraphIQA model on target dataset, you need to run:
~~~
python finetune.py --dataset DATASET --restore --ckpt PATH/TO/PRETRAINED_MODEL --gpus 1 --gpu_ids 0 --bs 32
~~~
Some available options:
* `--dataset`: Training and testing dataset, support datasets: live | livec | koniq-10k | csiq | kadid10k
* `--restore`: Enable to load pretrained model.
* `--ckpt`: Path to pretrained model.
* `--gpus`: The number of required gpus. For finetuning, 1 gpu is enough.
* `--gpu_ids`: Visible devices.
* `--bs`: Batch size.

## Cite us
Please cite us if you find this work helps.

```
@article{sun2022graphiqa,
  title={Graphiqa: Learning distortion graph representations for blind image quality assessment},
  author={Sun, Simen and Yu, Tao and Xu, Jiahua and Zhou, Wei and Chen, Zhibo},
  journal={IEEE Transactions on Multimedia},
  year={2022},
  publisher={IEEE}
}
```
