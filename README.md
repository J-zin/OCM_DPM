<h1 align="center">OCM DPM</h1>

This repo contains PyTorch implementation of the paper "[Improving Probabilistic Diffusion Models With Optimal Covariance Matching](https://arxiv.org/abs/2406.10808)"

by [Zijing Ou](https://j-zin.github.io/), [Mingtian Zhang](https://mingtian.ai/), [Andi Zhang](https://andi.ac/), [Tim Xiao](https://timx.me/), [Yingzhen Li](http://yingzhenli.net/home/en/), and [David Barber](http://web4.cs.ucl.ac.uk/staff/D.Barber/pmwiki/pmwiki.php).

> we leverage the covariance moment matching technique and introduce a novel method for learning the diagonal covariances of diffusion models. Unlike traditional data-driven covariance approximation approaches, our method involves directly regressing the optimal analytic covariance using a new, unbiased objective named Optimal Covariance Matching (OCM). This approach can significantly reduce the approximation error in covariance prediction. We demonstrate how our method can substantially enhance the sampling efficiency, recall rate and likelihood of both diffusion models and latent diffusion models.

## Installation

Our implementation is based on the [Extended Analytic-DPM](https://github.com/baofff/Extended-Analytic-DPM) repository. To set up the environment, please follow the installation instructions provided in that repository. The main functionality of our code closely mirrors the original repo, and we provide detailed usage instructions below.

## Training

To train the model, you can use the following command:

```bash
python run_train.py --pretrained_path path/to/pretrained_dpm --dataset dataset --workspace path/to/working_directory $train_hparams
```

* `pretrained_path` is the path to a pretrained diffusion probabilistic model (DPM). Here are the links to the pretrained models:
  [CIFAR10 (LS)](https://drive.google.com/file/d/1rhZBWUDK3_q37Iac3sXq6WnxR_OHhPyI/view), [CIFAR10 (CS)](https://drive.google.com/file/d/1ONNLpqPDLr4NesC0TfVZ3dCyaVBu7Xw0/view), [CelebA64](https://drive.google.com/file/d/1bGQGTsFOnqQ2z3FN5rdkj1FPN1_5nYF4/view), [ImageNet64](https://drive.google.com/file/d/1evlXbMOg55y2BIjiALcD6Smbm07k7XGW/view), [LSUN-Bedroom](https://drive.google.com/file/d/1fVxn3C5uaXdZM4cc8WnQ6GXexS5-274k/view).
* `dataset` represents the training dataset, one of <`cifar10`|`celeba64`|`imagenet64`|`lsun_bedroom`>.
* `workspace` is the place to put training outputs, e.g., logs and checkpoints.
* `train_hparams` specify other hyperparameters used in training. 

We provide the `train_hparams` used in training for our models on each dataset:

  * CIFAR10 (LS): `--method pred_eps_hes_pretrained`
  * CIFAR10 (CS): `--method pred_eps_hes_pretrained --schedule cosine_1000`
  * CelebA64: `--method pred_eps_hes_pretrained`
  * ImageNet64: `--method pred_eps_hes_pretrained --mode complex`
  * LSUN-Bedroom: `--method pred_eps_hes_pretrained --mode complex`

As an example, to train the CIFAR10 (LS) model, you can run:

```bash
python run_train.py --pretrained_path path/to/pretrained_dpm --dataset cifar10 --workspace path/to/working_directory --method pred_eps_hes_pretrained
```

## Evaluation

To evaluate the model, you can use the following command:

```bash
python run_eval.py --pretrained_path path/to/evaluated_model --dataset dataset --workspace path/to/working_directory --phase phase --sample_steps sample_steps --batch_size batch_size --method pred_eps_hes_pretrained $eval_hparams
```
* `pretrained_path` is the path to a model to evaluate. We provide all checkpoints trained with the proposed OCM approach [here](https://drive.google.com/drive/u/1/folders/10fIkU03aEc8qL4op01K9KFQ5sCpFOKym).
* `dataset` represents the dataset the model is trained on, one of <`cifar10`|`celeba64`|`imagenet64`|`lsun_bedroom`>.
* `workspace` is the place to put evaluation outputs, e.g., logs, samples and bpd values.
* `phase` specifies running FID or likelihood evaluation, one of <`sample4test`|`nll4test`>.
* `sample_steps` is the number of steps to run during inference, the samller this value the faster the inference.
* `batch_size` is the batch size, e.g., 500.
* `eval_hparams` specifies other optional hyperparameters used in evaluation.

We provide `eval_hparams` for the FID and NLL results in this paper.
- FID Evaluation (DDPM)
  * CIFAR10 (LS): `--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 2`
  * CIFAR10 (CS): `--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 1 --schedule cosine_1000`
  * CelebA64: `--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 2`
  * ImageNet64: `--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 1 --mode complex`
  * LSUN-Bedroom: `--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 1 --mode complex`
- FID Evaluation (DDIM)
  * CIFAR10 (LS): `--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 1 --forward_type ddim --eta 0`
  * CIFAR10 (CS): `--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 1 --forward_type ddim --eta 0 --schedule cosine_1000`
  * CelebA64: `--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 1 --forward_type ddim --eta 0`
  * ImageNet64: `-rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 1 --forward_type ddim --eta 0 --mode complex`
  * LSUN-Bedroom: `--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 1 --forward_type ddim --eta 0 --mode complex`
- NLL Evaluation
  * CIFAR10 (LS): `--rev_var_type optimal`
  * CIFAR10 (CS): `--rev_var_type optimal --schedule cosine_1000`
  * CelebA64: `--rev_var_type optimal`
  * ImageNet64: `--rev_var_type optimal --mode complex`

This [link](https://drive.google.com/drive/folders/1aqSXiJSFRqtqHBAsgUw4puZcRqrqOoHx?usp=sharing) provides precalculated FID statistics on CIFAR10, CelebA64, ImageNet64 and LSUN-Bedroom. They are computed following Appendix F.2 in [Analytic-DPM](https://arxiv.org/abs/2201.06503).

As an example, to evaluate the FID (DDPM) result of the CIFAR10 (LS) model, you can run:
```bash
python run_eval.py --pretrained_path path/to/pretrained_dpm --dataset dataset --workspace path/to/working_directory --phase sample4test --sample_steps sample_steps --batch_size batch_size --method pred_eps_hes_pretrained --rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 2
```

To evaluate the NLL result of the CIFAR10 (LS) model, you can run:
```bash
python run_eval.py --pretrained_path path/to/pretrained_dpm --dataset dataset --workspace path/to/working_directory --phase nll4test --sample_steps sample_steps --batch_size batch_size --method pred_eps_hes_pretrained --rev_var_type optimal
```

## Citation
:smile:If you find this repo is useful, please consider to cite our paper:
```
@article{ou2024improving,
  title={Improving Probabilistic Diffusion Models With Optimal Covariance Matching},
  author={Ou, Zijing and Zhang, Mingtian and Zhang, Andi and Xiao, Tim Z and Li, Yingzhen and Barber, David},
  journal={arXiv preprint arXiv:2406.10808},
  year={2024}
}
```