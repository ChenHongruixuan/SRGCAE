# Unsupervised Multimodal Change Detection Based on Structural Relationship Graph Representation Learning
Code for [Unsupervised Multimodal Change Detection Based on Structural Relationship Graph Representation Learning.](https://ieeexplore.ieee.org/document/9984688)

<img src="./figures/SRGCAE.jpg">

## Abstract
Unsupervised multimodal change detection is a practical and challenging topic that can play an important role in time-sensitive emergency applications. To address the challenge that multimodal remote sensing images cannot be directly compared due to their modal heterogeneity, we take advantage of two types of modality-independent structural relationships in multimodal images. In particular, we present a structural relationship graph representation learning framework for measuring the similarity of the two structural relationships. Firstly, structural graphs are generated from preprocessed multimodal image pairs by means of an object-based image analysis approach. Then, a structural relationship graph convolutional autoencoder (SR-GCAE) is proposed to learn robust and representative features from graphs. Two loss functions aiming at reconstructing vertex information and edge information are presented to make the learned representations applicable for structural relationship similarity measurement. Subsequently, the similarity levels of two structural relationships are calculated from learned graph representations and two difference images are generated based on the similarity levels. After obtaining the difference images, an adaptive fusion strategy is presented to fuse the two difference images. Finally, a morphological filtering-based postprocessing approach is employed to refine the detection results. Experimental results on six datasets with different modal combinations demonstrate the effectiveness of the proposed method.

## Requirements

```
python==3.9.7
pytorch==1.9.0
scikit-learn==0.18.3
imageio=2.9.0
numpy==1.20.3
gdal==3.0.2
opencv==4.5.5
```

## Citation
Please cite our paper if you use this code in your research.
```
@article{chen2022unsupervised,
  author={Chen, Hongruixuan and Yokoya, Naoto and Wu, Chen and Du, Bo},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Unsupervised Multimodal Change Detection Based on Structural Relationship Graph Representation Learning}, 
  year={2022},
  volume={},
  number={},
  pages={1-18},
  doi={10.1109/TGRS.2022.3229027}
}
```
## Q & A
**For any questions, please [contact us.](mailto:Qschrx@gmail.com)**
