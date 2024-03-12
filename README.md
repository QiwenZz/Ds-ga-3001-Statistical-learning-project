# Plant Seedlings Classification

This repository contains the code for the Plant Seedlings Classification project, developed by Qiwen Zhang, Kristi Topollai, and Lehan Li. The project aims to classify different types of plant seedlings based on images using various deep learning techniques. The kaggle competition is: https://www.kaggle.com/c/plant-seedlings-classification/overview

## Introduction

The project utilizes computer vision techniques to detect weeds among various types of crop seedlings. The dataset used is obtained from the Kaggle competition "Plant Seedlings Classification," based on the open-source dataset from The Aarhus University Signal Processing group and the University of Southern Denmark. It contains 12 species and a total of approximately 960 unique plants' images.

## Data Preprocessing

The data preprocessing includes transformation, augmentation, image segmentation, and SMOTE for class imbalance. Preprocessing techniques such as rotation, affine transformation, flipping, noise addition, brightness adjustment, and normalization are employed to add variability to the training data.

## Model

Various deep learning models are explored, including ResNet, Inception, and Data-Efficient Image Transformers (DeiT). The models are fine-tuned on the preprocessed dataset to achieve optimal performance.

## Optimization and Ensembling

Techniques such as Snapshot Ensembling, learning rate scheduling, and hyperparameter tuning with Optuna are used to enhance the model's performance. Different optimizers like SGD with momentum and Adam are also experimented with.

## Results

The project achieved a test micro-averaged F1-score of 0.98362. Various combinations of deep learning models and image preprocessing techniques were tried to achieve the best performance.

## Prerequisites

Before running the code, ensure you have the following installed:
- Python 3.x
- PyTorch
- torchvision
- OpenCV
- numpy
- scikit-learn
- optuna

## Usage

To run the training script, use the following command:

```bash
python run.py --path data/train --model resnet50 --epochs 100 --bz 64
