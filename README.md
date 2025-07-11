# Alzheimer's Disease Classification Using Clinical Data: A Comparative Analysis of Deep Learning and Traditional Machine Learning Approaches

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14894609.svg)](https://doi.org/10.5281/zenodo.14894609)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Abstract

Alzheimer's disease (AD) is a progressive neurodegenerative disorder that is the leading cause of dementia worldwide, affecting over 50 million people globally. Although clinical examinations and neuroimaging are considered the diagnostic gold standard, their high cost, lengthy acquisition times, and limited accessibility underscore the need for alternative approaches. This study presents a rigorous comparative analysis of traditional Machine Learning (ML) algorithms and advanced Deep Learning (DL) architectures that rely solely on structured clinical data to enable early, scalable AD detection.

We propose a novel hybrid model that integrates Convolutional Neural Networks (CNNs), DigitCapsule-Net, and a Transformer encoder to classify four disease stages: cognitively normal (CN), early mild cognitive impairment (EMCI), late mild cognitive impairment (LMCI), and Alzheimer's disease (AD). The CNN+DigitCapsule-Net hybrid attained **90.58% accuracy** in the three-class setting, outperforming state-of-the-art baselines that rely only on clinical variables. Model interpretability was assessed with SHAP and Grad-CAM, which identified CDR-SB, LDELTOTAL, and mPACC-TrailsB as the most informative clinical features.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
- [Model Architectures](#model-architectures)
- [Usage](#usage)
- [Results](#results)
- [Interpretability Analysis](#interpretability-analysis)
- [Clinical Significance](#clinical-significance)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [Limitations](#limitations)
- [References](#references)
- [Acknowledgments](#acknowledgments)

## Project Overview

This research addresses the critical need for accessible, cost-effective, and early diagnostic tools for Alzheimer's disease. By leveraging only structured clinical data, our approach offers a scalable solution that can be deployed in resource-limited settings where advanced neuroimaging may not be available.

### Key Contributions

1. **Comprehensive Comparative Analysis**: Systematic evaluation of traditional ML algorithms versus advanced DL architectures for AD classification
2. **Novel Hybrid Architecture**: Integration of CNNs, DigitCapsule-Net, and Transformer encoders for enhanced classification performance
3. **Multi-class Classification**: Robust classification across four cognitive stages (CN, EMCI, LMCI, AD)
4. **Class Imbalance Handling**: Implementation of multiple oversampling techniques (SMOTE, ADASYN, SMOTE-Tomek)
5. **Interpretability Focus**: SHAP and Grad-CAM analysis for clinical feature importance identification
6. **Clinical Validation**: Performance validation on ADNI cohort with rigorous statistical testing

## Dataset

### Alzheimer's Disease Neuroimaging Initiative (ADNI)

The study utilizes the ADNI dataset, a landmark longitudinal study designed to track the progression of AD through various biomarkers and clinical assessments.

**Dataset Characteristics:**
- **Source**: ADNI consortium (adni.loni.usc.edu)
- **Participants**: Multi-site cohort with diverse demographic representation
- **Data Types**: Structured clinical assessments, cognitive tests, demographic information
- **Classes**: 
  - CN (Cognitively Normal)
  - EMCI (Early Mild Cognitive Impairment)
  - LMCI (Late Mild Cognitive Impairment)
  - AD (Alzheimer's Disease)

**Key Clinical Features:**
- CDR-SB (Clinical Dementia Rating Sum of Boxes)
- LDELTOTAL (Logical Memory II Delayed Total Recall)
- mPACC-TrailsB (Modified Preclinical Alzheimer Cognitive Composite)
- MMSE (Mini-Mental State Examination)
- ADAS-Cog (Alzheimer's Disease Assessment Scale-Cognitive)
- Demographic variables (age, gender, education, APOE4 status)

## Repository Structure

```
├── BalancedData/                 # Preprocessed balanced datasets
├── BalancedML/                   # ML experiments with balanced data
│   ├── ML_4clasesbalanced.ipynb              # 4-class classification
│   ├── Alzh_ML_Multiclase_AD_CN_EMCI_balanced.ipynb  # 3-class: AD vs CN vs EMCI
│   ├── Alzh_ML_Multiclase_AD_CN_LMCI_balanced.ipynb  # 3-class: AD vs CN vs LMCI
│   ├── Alzh_ML_Multiclase_AD_EMCI_LMCI_balanced.ipynb # 3-class: AD vs EMCI vs LMCI
│   └── Alz_ML_Multiclase_CN_EMCI_LMCI_balanced.ipynb  # 3-class: CN vs EMCI vs LMCI
├── UnbalancedML/                 # ML experiments with original data distribution
│   ├── Alzh_ML4clases.ipynb                  # 4-class classification
│   ├── Alzh_ML_Multiclase_AD_CN.ipynb       # Binary: AD vs CN
│   ├── Alzh_ML_Multiclase_AD_CN_EMCI.ipynb  # 3-class: AD vs CN vs EMCI
│   ├── Alzh_ML_Multiclase_AD_EMCI.ipynb     # Binary: AD vs EMCI
│   ├── Alzh_ML_Multiclase_AD_EMCI_LMCI.ipynb # 3-class: AD vs EMCI vs LMCI
│   ├── Alzh_ML_Multiclase_CN_EMCI_LMCI.ipynb # 3-class: CN vs EMCI vs LMCI
│   ├── Alz_ML_Multiclase_AD_CN_LMCI.ipynb   # 3-class: AD vs CN vs LMCI
│   ├── Azl_ML_Multiclase_AD_LMCI.ipynb      # Binary: AD vs LMCI
│   ├── cn_lmci.ipynb                         # Binary: CN vs LMCI
│   ├── ec_emci.ipynb                         # Binary: CN vs EMCI
│   └── lmci_emci.ipynb                       # Binary: LMCI vs EMCI
├── Unbalanced/                   # Original unbalanced datasets
├── interpretación/               # Model interpretability analysis
│   ├── GradCam.ipynb            # Grad-CAM visualization for DL models
│   └── Correlación/             # Feature correlation analysis
├── Tests/                        # Statistical validation
│   └── Statistical_Tests.ipynb  # Statistical significance testing
├── CITATION.cff                  # Citation information
└── README.md                     # This file
```

## Methodology

### 1. Data Preprocessing and Feature Engineering

**Feature Selection Techniques:**
- **Boruta Algorithm**: Wrapper method for relevant feature identification
- **Elastic-Net Regularization**: L1/L2 penalty combination for feature selection
- **Information Gain Ranking**: Entropy-based feature importance scoring

**Class Imbalance Handling:**
- **SMOTE (Synthetic Minority Oversampling Technique)**: Synthetic sample generation
- **ADASYN (Adaptive Synthetic Sampling)**: Density-based oversampling
- **SMOTE-Tomek**: Hybrid approach combining oversampling and undersampling

### 2. Experimental Design

**Classification Scenarios:**
- Binary classification (6 different pairs)
- 3-class classification (4 different combinations)
- 4-class classification (CN vs EMCI vs LMCI vs AD)

**Cross-Validation Strategy:**
- Stratified K-fold cross-validation (K=10)
- Nested cross-validation for hyperparameter optimization
- Statistical significance testing with paired t-tests

## Model Architectures

### Traditional Machine Learning Models

1. **Random Forest (RF)**: Ensemble of decision trees with bootstrap aggregation
2. **Gradient Boosting (GB)**: Sequential boosting with gradient descent optimization
3. **Support Vector Machine (SVM)**: Kernel-based classification with RBF kernel
4. **Logistic Regression (LR)**: Linear probabilistic classifier with regularization
5. **K-Nearest Neighbors (KNN)**: Instance-based learning with distance metrics
6. **Naive Bayes (NB)**: Probabilistic classifier with feature independence assumption

### Deep Learning Architectures

#### 1. Convolutional Neural Network (CNN)
```
Input Layer → 1D Conv → BatchNorm → ReLU → Dropout → 
1D Conv → BatchNorm → ReLU → MaxPool → Flatten → 
Dense → Dropout → Output Layer
```

#### 2. DigitCapsule-Net
- **Primary Capsules**: Low-level feature detection
- **Digit Capsules**: High-level entity representation
- **Dynamic Routing**: Iterative agreement mechanism
- **Reconstruction Network**: Regularization through input reconstruction

#### 3. Transformer Encoder
- **Multi-Head Attention**: Parallel attention mechanisms
- **Position Encoding**: Sequence order preservation
- **Feed-Forward Networks**: Non-linear transformations
- **Layer Normalization**: Training stabilization

#### 4. Hybrid CNN+DigitCapsule-Net
- **Feature Extraction**: CNN layers for initial feature learning
- **Capsule Processing**: Dynamic routing for entity relationship modeling
- **Classification Head**: Final prediction layer with softmax activation

## Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/MarioBravo12/Alzheimer-s-Disease-Classification-Using-Clinical-Data1.git
cd Alzheimer-s-Disease-Classification-Using-Clinical-Data1

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments

#### 1. Balanced Data Experiments
```bash
# 4-class classification with balanced data
jupyter notebook BalancedML/ML_4clasesbalanced.ipynb

# 3-class classification (AD vs CN vs EMCI)
jupyter notebook BalancedML/Alzh_ML_Multiclase_AD_CN_EMCI_balanced.ipynb
```

#### 2. Unbalanced Data Experiments
```bash
# Binary classification (AD vs CN)
jupyter notebook UnbalancedML/Alzh_ML_Multiclase_AD_CN.ipynb

# 4-class classification with original distribution
jupyter notebook UnbalancedML/Alzh_ML4clases.ipynb
```

#### 3. Interpretability Analysis
```bash
# Grad-CAM visualization
jupyter notebook interpretación/GradCam.ipynb

# Statistical testing
jupyter notebook Tests/Statistical_Tests.ipynb
```

### Data Requirements

- ADNI dataset access (requires registration at adni.loni.usc.edu)
- Clinical variables in CSV format
- Preprocessed feature matrices for each classification scenario

### Key Findings

1. **Hybrid Architecture Superiority**: CNN+DigitCapsule-Net consistently outperformed individual architectures
2. **Class Complexity Impact**: Performance decreased with increasing number of classes
3. **Traditional ML Competitiveness**: Gradient Boosting achieved comparable performance with lower computational cost
4. **Feature Importance**: CDR-SB, LDELTOTAL, and mPACC-TrailsB emerged as most discriminative features

### Statistical Validation

- **Friedman Tests**
- **Wilcoxon Signed-Rank Test**
- **Levene’s Test for Equality of Variances**

## Interpretability Analysis

- **SHAP**: Identified CDR-SB, LDELTOTAL, and mPACC-TrailsB as key clinical features.
- **Grad-CAM**: Used to understand DL attention mechanisms.

## Clinical Significance

### Diagnostic Support Tool

1. **Early Detection**: Identification of cognitive decline before severe symptoms
2. **Resource Efficiency**: Reduced dependency on expensive neuroimaging
3. **Accessibility**: Deployment in primary care settings
4. **Monitoring**: Longitudinal tracking of disease progression

### Clinical Workflow Integration

1. **Screening Tool**: First-line assessment in clinical practice
2. **Risk Stratification**: Patient prioritization for specialized care
3. **Treatment Planning**: Informed decision-making for interventions
4. **Research Support**: Patient recruitment for clinical trials

### Regulatory Considerations

- **FDA Guidelines**: Compliance with AI/ML-based medical device regulations
- **Clinical Validation**: Multi-site validation studies required
- **Bias Assessment**: Fairness evaluation across demographic groups
- **Interpretability Requirements**: Explainable AI for clinical adoption

## Reproducibility

### Software Environment
```
Python 3.11
TensorFlow 2.x
scikit-learn 1.x
pandas 1.x
numpy 1.x
matplotlib 3.x
seaborn 0.x
shap 0.x
```

### Experimental Protocol

1. **Data Preprocessing**: Standardized normalization and feature selection
2. **Model Training**: Fixed random seeds for reproducibility
3. **Hyperparameter Optimization**: Grid search with nested cross-validation
4. **Performance Evaluation**: Stratified k-fold cross-validation
5. **Statistical Testing**: Paired t-tests

### Code Availability

All experimental code, preprocessing scripts, and analysis notebooks are provided in this repository. The modular design allows for easy replication and extension of the methodology.

## Citation

If you use this code or dataset, please cite:

```
Mario Alejandro Bravo-Ortiz, Guevara-Navarro, E., & Holguin García, S. A. (2025). 
MarioBravo12/BI-CVT: BI-CVT First Version (v1.0.0). Zenodo. 
https://doi.org/10.5281/zenodo.15722443
```

## Limitations

### Technical Limitations

1. **Data Dependency**: Performance is contingent on data quality and completeness
2. **Generalizability**: Model validation limited to ADNI cohort characteristics
3. **Feature Engineering**: Manual feature selection may introduce bias
4. **Computational Requirements**: Deep learning models require significant resources

### Clinical Limitations

1. **Diagnostic Support Only**: Models are intended as clinical decision support tools, not standalone diagnostics
2. **Population Bias**: Training data may not represent all demographic groups equally
3. **Temporal Validation**: Limited longitudinal validation of prediction accuracy
4. **Clinical Context**: Results should be interpreted within broader clinical assessment

### Methodological Limitations

1. **Cross-sectional Analysis**: Limited temporal modeling of disease progression
2. **Feature Selection Bias**: Potential overfitting to specific feature sets
3. **Class Imbalance**: Despite mitigation strategies, imbalance effects may persist
4. **Validation Scope**: Single-dataset validation limits generalizability claims

## References

1. Alzheimer's Association. (2023). 2023 Alzheimer's disease facts and figures. Alzheimer's & Dementia, 19(4), 1598-1695.
2. Jack Jr, C. R., et al. (2018). NIA-AA Research Framework: Toward a biological definition of Alzheimer's disease. Alzheimer's & Dementia, 14(4), 535-562.
3. Petersen, R. C., et al. (2010). Alzheimer's Disease Neuroimaging Initiative (ADNI): Clinical characterization. Neurology, 74(3), 201-209.
4. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.
5. Sabour, S., Frosst, N., & Hinton, G. E. (2017). Dynamic routing between capsules. Advances in Neural Information Processing Systems, 30.
6. Vaswani, A., et al. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30.
7. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30.
8. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. Proceedings of the IEEE International Conference on Computer Vision.

## Acknowledgments

- **ADNI Consortium**: For providing the invaluable dataset that made this research possible
- **National Institute on Aging**: For supporting the ADNI initiative
- **Research Community**: For the foundational work in machine learning and Alzheimer's disease research
- **Open Source Community**: For the development of tools and libraries that enabled this research

**Disclaimer**: This research is for academic and research purposes only. The models and results presented should not be used as a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare professionals for medical decisions.
