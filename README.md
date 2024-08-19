# FAMF-Net: Feature Alignment Mutual Attention Fusion with Region Awareness for Breast Cancer Diagnosis via Imbalanced Data

This repository contains the code for our paper "FAMF-Net: Feature Alignment Mutual Attention Fusion with Region Awareness for Breast Cancer Diagnosis via Imbalanced Data." 

## Introduction

Breast cancer is a leading cause of cancer-related mortality among women. Accurate and early diagnosis through imaging is crucial for effective treatment. Our work proposes a novel method for the automatic classification of breast cancer using multimodal ultrasound images, addressing common challenges such as image misalignment, limited utilization of complementary information, poor interpretability, and class imbalance.

## Key Features

- **Region Awareness Alignment (RAA)**: Achieves feature alignment through class activation mapping and translation transformation.
- **Mutual Attention Fusion (MAF)**: Utilizes a mutual attention mechanism to enhance the complementarity of features from B-mode and shear wave elastography (SWE) images.
- **Reinforcement Learning-based Dynamic Optimization (RDO)**: Dynamically optimizes the weights of the loss function to handle class imbalance effectively.

## Methodology

The proposed FAMF-Net framework consists of three main components:

1. **Region Awareness Alignment (RAA)**: Aligns features from misaligned multimodal images using class activation mapping and translation transformation.
2. **Mutual Attention Fusion (MAF)**: Enhances the fusion of features from B-mode and SWE images, making the model more interpretable.
3. **Reinforcement Learning-based Dynamic Optimization (RDO)**: Adjusts the loss function weights dynamically based on the sample distribution and prediction probabilities.

## Dataset

Our dataset includes multimodal ultrasound images from 357 patients, with 312 cases of benign tumors and 45 cases of malignant tumors. The data were collected using the SIEMENS ACUSON Oxana II ABVS system. The dataset is highly imbalanced, making it a suitable case study for evaluating the effectiveness of our proposed method.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- Other dependencies as listed in `requirements.txt`

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Magnety/Multi_modal_Image.git
    cd Multi_modal_Image
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Results

Our experimental results demonstrate that FAMF-Net achieves significant improvements in accuracy, F1-score, and interpretability compared to existing methods. For detailed results and analysis, please refer to the paper.




