This repository contains the implementation of the following paper:
    SRDiff: Single image super-resolution with diffusion probabilistic models
    Haoying Li, Yifan Yang, Meng Chang, Shiqi Chen, Huajun Feng, Zhihai Xu, Qi Li, Yueting Chen
    Neurocomputing, Volume 479, pp 47-59
    https://doi.org/10.1016/j.neucom.2022.01.029

# Get Started

## Environment Installation

```bash
python -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset Preparation

1. To download DIV2K, Flickr2K and CelebA, and post-processing the data, please refer to https://github.com/andreas128/SRFlow

2. Please put the downloaded DIV2K, Flickr2K and CelebA dataset in  
    data/raw/DIV2K, 
    data/raw/Flickr2K, 
    data/raw/CelebA, 
    respectively

3. Pack to pickle for training
```bash
# pack the CelebA dataset
python data_gen/celeb_a.py --config configs/celeb_a.yaml 
# pack the DIV2K dataset
python data_gen/df2k.py --config configs/df2k4x.yaml
```

## Pretrained Model
- celeb
    - rrdb: 
    - srdiff: checkpoints/srdiff_pretrained_celebA/model_ckpt_steps_300000.ckpt
- df2k
    - rrdb: pretrained/xx
    - srdiff: srdiff_221105/checkpoints/srdiff_pretrained_div2k/model_ckpt_steps_400000.ckpt

## Train & Evaluate
1. Prepare datasets. Please refer to Dataset Preparation.
2. Modify config files.
    - CelebA
        - rrdb: configs/rrdb/celeb_a_pretrain.yaml
        - srdiff: configs/celeb_a.yaml
    - DIV2K
        - rrdb: configs/rrdb/df2k4x_pretrain.yaml
        - srdiff: configs/diffsr_df2k4x.yaml
3. Run training / evaluation code. The code is for training on 1 GPU.

### CelebA

```bash
# train rrdb-based conditional net
CUDA_VISIBLE_DEVICES=0 python tasks/trainer.py --config configs/rrdb/celeb_a_pretrain.yaml --exp_name rrdb_celebA_1 --reset
# train srdiff
CUDA_VISIBLE_DEVICES=0 python tasks/trainer.py --config configs/diffsr_celeb.yaml --exp_name diffsr_celebA_1 --reset --hparams="rrdb_ckpt=checkpoints/rrdb_celebA_1"

# tensorboard
tensorboard --logdir checkpoints/diffsr_celebA_1

# evaluate
CUDA_VISIBLE_DEVICES=0 python tasks/trainer.py --config configs/diffsr_celeb.yaml --exp_name diffsr_celebA_1 --infer

# evaluate with pretrained model
CUDA_VISIBLE_DEVICES=0 python tasks/trainer.py --config configs/diffsr_celeb.yaml --exp_name srdiff_pretrained_celebA --infer
```

### DIV2K

```bash
# train rrdb-based conditional net
CUDA_VISIBLE_DEVICES=0 python tasks/trainer.py --config configs/rrdb/df2k4x_pretrain.yaml --exp_name rrdb_div2k_1 --reset
# train srdiff
CUDA_VISIBLE_DEVICES=0 python tasks/trainer.py --config configs/diffsr_df2k4x.yaml --exp_name diffsr_div2k_1 --reset --hparams="rrdb_ckpt=checkpoints/rrdb_div2k_1"

# tensorboard
tensorboard --logdir checkpoints/diffsr_div2k_1

# evaluate
CUDA_VISIBLE_DEVICES=0 python tasks/trainer.py --config configs/diffsr_df2k4x.yaml --exp_name diffsr_div2k_1 --infer

# evaluate with pretrained model
CUDA_VISIBLE_DEVICES=0 python tasks/trainer.py --config configs/diffsr_df2k4x.yaml --exp_name srdiff_pretrained_div2k --infer
```

# Results
| Task | PSNR | SSIM| LPIPS| LR_PSNR | Total_loss |
| ---  | ---  | --- | ---  | ---     |      ---   |
| diffsr_celeb | 25.454 | 0.746 | 0.106 | 53.094 | 0.106 |
| diffsr_div2k | 27.160 | 0.786 | 0.129 | 53.675 | 0.129 |

# Citation
@article{LI202247,
title = {SRDiff: Single image super-resolution with diffusion probabilistic models},
journal = {Neurocomputing},
volume = {479},
pages = {47-59},
year = {2022},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2022.01.029},
url = {https://www.sciencedirect.com/science/article/pii/S0925231222000522},
author = {Haoying Li and Yifan Yang and Meng Chang and Shiqi Chen and Huajun Feng and Zhihai Xu and Qi Li and Yueting Chen},
keywords = {Single image super-resolution, Diffusion probabilistic model, Diverse results, Deep learning},
abstract = {Single image super-resolution (SISR) aims to reconstruct high-resolution (HR) images from given low-resolution (LR) images. It is an ill-posed problem because one LR image corresponds to multiple HR images. Recently, learning-based SISR methods have greatly outperformed traditional methods. However, PSNR-oriented, GAN-driven and flow-based methods suffer from over-smoothing, mode collapse and large model footprint issues, respectively. To solve these problems, we propose a novel SISR diffusion probabilistic model (SRDiff), which is the first diffusion-based model for SISR. SRDiff is optimized with a variant of the variational bound on the data likelihood. Through a Markov chain, it can provide diverse and realistic super-resolution (SR) predictions by gradually transforming Gaussian noise into a super-resolution image conditioned on an LR input. In addition, we introduce residual prediction to the whole framework to speed up model convergence. Our extensive experiments on facial and general benchmarks (CelebA and DIV2K datasets) show that (1) SRDiff can generate diverse SR results with rich details and achieve competitive performance against other state-of-the-art methods, when given only one LR input; (2) SRDiff is easy to train with a small footprint(The word “footprint” in this paper represents “model size” (number of model parameters).); (3) SRDiff can perform flexible image manipulation operations, including latent space interpolation and content fusion.}
}
