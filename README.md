# Old can be Gold: Better Gradient Flow can make Vanilla-GCNs Great Again
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

https://arxiv.org/abs/2210.08122

## Abstract

Despite the enormous success of Graph Convolutional Networks (GCNs) in mod-
elling graph-structured data, most of the current GCNs are shallow due to the
notoriously challenging problems of over-smoothening and information squashing
along with conventional difficulty caused by vanishing gradients and over-fitting.
Previous works have been primarily focused on the study of over-smoothening and
over-squashing phenomenon in training deep GCNs. Surprisingly, in comparison
with CNNs/RNNs, very limited attention has been given towards understanding
how healthy gradient flow can benefit the trainability of deep GCNs. In this paper,
firstly, we provide a new perspective of gradient flow to understand the substandard
performance of deep GCNs and hypothesize that by facilitating healthy gradient
flow, we can significantly improve their trainability, as well as achieve state-of-the-
art (SOTA) level performance from vanilla-GCNs [1]. Next, we argue that blindly
adopting the Glorot initialization for GCNs is not optimal, and derive a topology-
aware isometric initialization scheme for vanilla-GCNs based on the principles
of isometry. Additionally, contrary to ad-hoc addition of skip-connections, we
propose to use gradient-guided dynamic rewiring of vanilla-GCNs with skip-
connections. Our dynamic rewiring method uses the gradient flow within each
layer during training to introduce skip-connections on-demand basis. We provide
extensive empirical evidence across multiple datasets that our methods improves
gradient flow in deep vanilla-GCNs and significantly boost their performance to
comfortably compete and outperform many fancy state-of-the-art methods. 

![image](https://user-images.githubusercontent.com/6660499/193684200-ef091d81-cb91-4496-8e6e-d7e297c573e1.png)

## Benefits of our proposed techniques

![image](https://user-images.githubusercontent.com/6660499/193684330-c1762f74-6fb6-478d-b305-b02d145f8fcb.png)

![image](https://user-images.githubusercontent.com/6660499/193684419-42a18f00-6386-4439-ab3a-6685f1537a43.png)


![image](https://user-images.githubusercontent.com/6660499/193684907-10f6d567-4922-4677-987c-1bffb6111658.png)


![image](https://user-images.githubusercontent.com/6660499/193684522-455cf185-b410-453f-81a9-1be0c621c25c.png)

If you find our work helpful in your research, please cite our paper

## Citation

If you find our code implementation helpful for your own resarch or work, please cite our paper.
```
@inproceedings{Jaiswal22GradientGCN,
  title={Old can be Gold: Better Gradient Flow can make Vanilla-GCNs Great Again},
  author={Ajay Jaiswal, Peihao Wang, Tianlong Chen, Justin F Rousseau, Ying Ding, Zhangyang Wang},
  booktitle={NeurIPS 2022},
  year={2022}
}
```
