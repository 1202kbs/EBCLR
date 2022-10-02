# EBCLR
Official PyTorch implementation of [Energy-Based Contrastive Learning of Visual Representations](https://arxiv.org/abs/2202.04933), NeurIPS 2022.

We propose a visual representation learning framework, Energy-Based Contrastive Learning (EBCLR), that combines Energy-Based Models (EBMs) with contrastive learning. EBCLR associates distance on the projection space with the density of positive pairs to learn useful visual representations. The figure below illustrates the general idea of EBCLR. We find EBCLR shows accelerated convergence and robustness to small number of negative pairs per positive pair.

<p align="center">
  <img src="https://github.com/1202kbs/EBCLR/blob/main/assets/main.png" />
</p>

## How to Run This Code

Example codes for training and linear / KNN evaluation can be found in `Train.ipynb` and `Evaluation.ipynb`, respectively.

## References

If you find the code useful for your research, please consider citing
```bib
@inproceedings{
  kim2022ebclr,
  title={Energy-Based Contrastive Learning of Visual Representations},
  author={Beomsu Kim and Jong Chul Ye},
  booktitle={NeurIPS},
  year={2022}
}
```
