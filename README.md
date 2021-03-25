# GraphIQA
GraphIQA: Learning Distortion Graph Representations for Blind Image Quality Assessment
![image](https://github.com/geekyutao/GraphIQA/blob/main/fig/framework.png)
> [**GraphIQA**](https://arxiv.org/abs/2103.07666),            
> Simeng Sun, Tao Yu, Jiahua Xu, Jianxin Lin, Wei Zhou, Zhibo Chen,        
> *arXiv technical report ([arXiv:2103.07666](https://arxiv.org/abs/2103.07666))*  
The code is coming soon.

## Abstract
Learning-based blind image quality assessment (BIQA) methods have recently drawn much attention for their superior performance compared to traditional methods. However, most of them do not effectively leverage the relationship between distortion-related factors, showing limited feature representation capacity. In this paper, we show that human perceptual quality is highly correlated with distortion type and degree, and is biased by image content in most IQA systems. Based on this observation, we propose a Distortion Graph Representation (DGR) learning framework for IQA, called GraphIQA. In GraphIQA, each distortion is represented as a graph i.e. DGR. One can distinguish distortion types by comparing different DGRs, and predict image quality by learning the relationship between different distortion degrees in DGR. Specifically, we develop two sub-networks to learn the DGRs: a) Type Discrimination Network (TDN) that embeds DGR into a compact code for better discriminating distortion types; b) Fuzzy Prediction Network (FPN) that extracts the distributional characteristics of the samples in DGR and predicts fuzzy degrees based on a Gaussian prior. Experiments show that our GraphIQA achieves the state-of-the-art performance on many benchmark datasets of both synthetic and authentic distortion.

## Preparation

## Train GraphIQA

## Test GraphIQA

## Visualization

## Cite Us
Please cite us if you find this work helps.

```
@article{sun2021graphiqa,
  title={GraphIQA: Learning Distortion Graph Representations for Blind Image Quality Assessment},
  author={Sun, Simeng and Yu, Tao and Xu, Jiahua and Lin, Jianxin and Zhou, Wei and Chen, Zhibo},
  journal={arXiv preprint arXiv:2103.07666},
  year={2021}
}
```
