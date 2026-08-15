# An-Enhanced-Deep-Learning-Framework-for-Heart-Disease-Classification-using-Echocardiogram-Images

## Overview

This project presents a hybrid deep learning framework for classifying heart diseases from echocardiogram images. The proposed approach combines VGG16 and InceptionV3 to extract complementary features from cardiac images and classify them into four disease categories.

The four categories considered in this project are:

- Angina Disease
- Cardiovascular Disease (CVD)
- Coronary Artery Disease (CAD)
- Hypotension

The dataset contains 2,404 echocardiogram images distributed across the four classes.

## Objectives

- Develop an automated heart disease classification system using deep learning.
- Extract meaningful image features using pretrained CNN architectures.
- Combine VGG16 and InceptionV3 features into a hybrid model.
- Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix.
- Analyze the performance of the proposed hybrid architecture.

## Methodology

The overall workflow consists of:

1. Dataset extraction and preprocessing
2. Image format conversion
3. Dataset organization
4. Train-validation-test splitting
5. Image augmentation
6. Feature extraction using VGG16 and InceptionV3
7. Feature concatenation
8. Fully connected classification layers
9. Model training
10. Model evaluation

### Dataset Split

The dataset is divided into:

- 70% Training
- 12% Validation
- 18% Testing

### Image Processing

Input images are resized to:

224 × 224 × 3

Training images are augmented using:

- Rotation
- Zoom
- Horizontal flipping

## Model Architecture

The proposed model uses two pretrained CNN architectures:

### VGG16

VGG16 is used as one feature extraction branch.

### InceptionV3

InceptionV3 is used as the second feature extraction branch.

The feature representations obtained from both networks are combined using a concatenation layer.

The combined features are passed through:

- Batch Normalization
- Dense layer with 1024 neurons
- Dropout
- Dense layer with 256 neurons
- Dropout
- Softmax output layer

## Training Configuration

 Parameter - Value 
 Image Size - 224 × 224 
 Batch Size - 16 
 Epochs - 100 
 Optimizer - Adam 
 Learning Rate - 5 × 10⁻⁵ 
 Loss Function - Categorical Cross-Entropy 
 Data Split - 70% / 12% / 18% 
 Platform - Google Colab 
 GPU - NVIDIA T4 

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Results

The research paper reports the following test accuracies:

 Model - Accuracy 

 VGG16 - 93.17% 
 InceptionNet - 91.13% 
 Proposed Hybrid Model - 93.75% 

The proposed hybrid model achieved an average accuracy of 93.75%.
