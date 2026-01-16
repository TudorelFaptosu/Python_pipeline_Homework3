# Python_pipeline_Homework3

**Student Name:** [Your Name]
**Course:** Advanced Topics in Neural Networks
**Assignment:** 3 - PyTorch Training Pipeline

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
