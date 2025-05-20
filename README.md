# Adversarial Sparse Teacher (AST)

This repository implements **Adversarial Sparse Teacher (AST)**, a defense method designed to protect deep learning models from knowledge distillation-based model stealing attacks.

AST enhances output ambiguity by training the teacher model on adversarial examples and sparse logit representations, making it significantly harder for student models to replicate the teacher’s knowledge effectively.

---

## Getting Started

Ensure you have Python 3.7+ and the required packages installed (see `requirements.txt` if available).

### 1. Train the Baseline Teacher
```bash
python train_base.py --save_path /experiments/CIFAR10/baseline/resnet18
```
### 2. Generate Adversarial Dataset
```bash

python create_adv_dataset.py --save_path data/advcifar10/resnet18
```
### 3.Train the AST Model
```bash

python train_adv_teacher.py --save_path /experiments/CIFAR10/adversarial_sparse_teacher/resnet18
```
### 4. Train Student via Knowledge Distillation (KD)
```bash

python train_kd.py /experiments/CIFAR10/kd_ast_resnet18/resnet18
```
## Acknowledgment
This work builds upon and extends the concepts introduced in:
[NastyTeacher](https://github.com/VITA-Group/Nasty-Teacher).
