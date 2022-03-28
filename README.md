# <a href="https://arxiv.org/abs/2203.04181" target="_blank"> Selective-Supervised Contrastive Learning with Noisy Labels </a> - Official PyTorch Code (CVPR 2022)

### Abstract:
Deep networks have strong capacities of embedding data into latent representations and finishing following tasks. However, the capacities largely come from high-quality annotated labels, which are expensive to collect. Noisy labels are more affordable, but result in corrupted representations, leading to poor generalization performance. To learn robust representations and handle noisy labels, we propose selective-supervised  contrastive learning (Sel-CL) in this paper. Specifically, Sel-CL extend supervised contrastive learning (Sup-CL), which is powerful in representation learning, but is degraded when there are noisy labels. Sel-CL tackles the direct cause of the problem of Sup-CL. That is, as Sup-CL works in a pair-wise manner, noisy pairs built by noisy labels mislead representation learning. To alleviate the issue, we select confident pairs out of noisy ones for Sup-CL without knowing noise rates. In the selection process, by measuring the agreement between learned representations and given labels, we first identify confident examples that are exploited to build confident pairs. Then, the representation similarity distribution in the built confident pairs is exploited to identify more confident pairs out of noisy pairs. All obtained confident pairs are finally used for Sup-CL to enhance representations. Experiments on multiple noisy datasets demonstrate the robustness of the learned representations by our method, following the state-of-the-art performance.


### Requirements:
* Python 3.8.10
* Pytorch 1.8.0 (torchvision 0.9.0)
* Numpy 1.19.5
* scikit-learn 1.0.1
* apex 0.1


### Running the code on CIFAR-10/100:
We provide the code used to simulate CIFAR-10/100 datasets with symmetric and asymmetric label noise and provide example scripts to run our approach with both noises.

To run the code use the provided scripts in CIFAR folders. Datasets are downloaded automatically when setting "--download True". The dataset has to be placed in dataset folder (should be done automatically). The training has two stages: first pre-train the backbone by running train_Sel-CL.py, and then fine-tune it by running train_Sel-CL_fine-tuning.py. During training, the results can be obtained in the log file in out folder.

We also provide another version of Sel-CL training code on the first training stage, which reduces the computational complexity of KNN search and pair selection and can be applied on large-scale datasets. It can be used by running train_Sel-CL_v2.py.

### Running the code on WebVision-50:
We provide the code and scripts used to evaluate our approach on WebVision-50 dataset. The dataset should be downloaded and placed into the data directory before running code.


### Citation:
If you find the code useful in your research, please consider citing our paper:

```
 @InProceedings{Li2022SelCL,
  title = {Selective-Supervised Contrastive Learning with Noisy Labels},
  authors = {Shikun Li and Xiaobo Xia and Shiming Ge and Tongliang Liu},
  year={2022},
  booktitle ={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
 } 
```


Note: Our implementation uses parts of some public codes [1-3].

[1] MOIT https://https://github.com/DiegoOrtego/LabelNoiseMOIT

[2] MoCo https://github.com/facebookresearch/moco

[3] RRL https://github.com/salesforce/RRL/ 
