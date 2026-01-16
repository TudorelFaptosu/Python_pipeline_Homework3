# Python_pipeline_Homework3

**Student Name:** Caldarescu Tudor
**Course:** Advanced Topics in Neural Networks
**Homework:** 3 - PyTorch Training Pipeline

---

## 1. Setup & How to Run
*Reference: Requirement "Setup + How to run"*

This project implements a generic, device-agnostic training pipeline compatible with CIFAR-10, CIFAR-100, MNIST, and OxfordIIITPet.

### Dependencies
The pipeline requires the following libraries:
* `torch`
* `torchvision`
* `timm` (for model backbones)
* `tensorboard` (for logging)
* `tqdm` (for progress bars)
* `numpy`

### Running the Training Script
The script `train.py` is configurable via command-line arguments. Below are examples of how to run the required configurations.

**1. Basic Run (CIFAR-100 with ResNest26d):**
```bash
python train.py --dataset CIFAR100 --model resnest26d --epochs 50 --batch_size 128 --optimizer AdamW --lr 0.001 --device cuda


** Using Pretraining:**
```bash
python train.py --dataset CIFAR100 --model resnet26d --pretrained --epochs 20

** Hyperparameter Sweep Example: **
```bash
python train.py --exp_name sweep_sgd_lr01 --optimizer SGD --lr 0.01
```bash
python train.py --exp_name sweep_adam_lr001 --optimizer Adam --lr 0.001


** Runing with Batch Size Scheduling: **
```bash
python train.py --batch_schedule "10:256,30:512"

**2. Pipeline Implementation Features
