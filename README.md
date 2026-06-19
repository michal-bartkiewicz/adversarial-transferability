# adversarial-transferability
This repository contains a framework for evaluating the vulnerability of deep learning models to gradient-based adversarial evasion attacks in both white-box and black-box settings. Using the Tiny ImageNet-200 dataset, the project provides pipeline implementations for training target and surrogate models (including via knowledge distillation), generating FGSM and PGD perturbations, and systematically evaluating the transferability of adversarial examples across different neural network architectures.

# Adversarial Transferability Framework

A modular PyTorch implementation for studying **Gradient-based Adversarial Evasion Attacks** and their **Transferability** (Black-box attacks) on the Tiny ImageNet-200 dataset.

## Project Structure

```text
├── data/               # Dataset storage (downloaded automatically)
├── weights/            # Model weights storage (.pth)
├── models/
│   ├── model_wrapper.py # ModelWrapper class with metadata
│   └── model_utils.py   # Factory functions for ResNet, MobileNet, etc.
├── attacks/
│   └── evasion_attacks.py # FGSM, PGD, and evaluation logic
├── utils/
│   ├── dataset_utils.py  # Data loading and Tiny ImageNet normalization
│   └── training_utils.py # Training, teacher-student labels, and distillation
├── main.py             # Main entry point for training and evaluation
├── demo.ipynb          # Interactive demonstration and visualization
└── requirements.txt    # Project dependencies
```

## Key Features

- **Standardized Model Naming**: Models are wrapped with a `.name` property identifying architecture and whether weights were from `scratch` or `pretrained`.
- **Adversarial Evasion Attacks**: Implementation of FGSM and PGD with support for both white-box and black-box (transferability) evaluations.
- **Tiny ImageNet-200**: Automated downloading and preprocessing of the Tiny ImageNet dataset.
- **Training Strategies**: Supports standard CrossEntropy training, training using teacher-generated labels, and logit distillation (MSE).

## Setup & Execution

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Experiments
To train models and run the base evaluation pipeline:
```bash
python main.py --epochs 5 --epsilon 0.1
```
The script will:
- Download Tiny ImageNet to `./data/`.
- Train or load a **Target** model (EfficientNet-V2-S).
- Train or load a **Surrogate** model (ResNet-18) using labels from the Target.
- Evaluate white-box and black-box attack success rates.

### 3. Demonstration
Open `demo.ipynb` to visualize the accuracy drop across different perturbation budgets ($\epsilon$) and see visual examples of adversarial images.

## Scientific Nomenclature
This repository uses precise taxonomy for adversarial machine learning:
- **Gradient-based Adversarial Evasion Attacks**: Direct attacks against a known model.
- **Transferability-based Black-box Attacks**: Indirect attacks where gradients from a surrogate are used against a target.

## License
MIT
