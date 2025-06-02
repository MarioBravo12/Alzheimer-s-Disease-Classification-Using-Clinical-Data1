# Alzheimer-s-Disease-Classification-Using-Clinical-Data
Comparative Analysis of Transformer Encoder, Capsule-Net, and Traditional ML Techniques in Alzheimer's Disease Classification Using Clinical Data

Alzheimer’s disease (AD) is a progressive neuro-degenerative disorder that is the leading cause of
dementia worldwide. Although clinical examinations and neuro-imaging are considered the diagnostic
gold standard, their high cost, lengthy acquisition times, and limited accessibility underscore the need
for alternative approaches. This study presents a rigorous comparative analysis of traditional Machine
Learning (ML) algorithms and advanced Deep Learning (DL) architectures that rely solely on structured
clinical data to enable early, scalable AD detection. This research proposes a novel hybrid model that
integrates Convolutional Neural Networks (CNNs), DigitCapsule-Net, and a Transformer encoder to
classify four disease stages: cognitively normal (CN), early mild cognitive impairment (EMCI), late mild
cognitive impairment (LMCI), and AD. Feature selection was carried out on the ADNI cohort with the
Boruta algorithm, Elastic-Net regularization, and information-gain ranking. To address class imbalance,
we applied three oversampling techniques—SMOTE, ADASYN, and SMOTE-Tomek. In the three-class
setting, the CNN+DigitCapsule-Net hybrid attained 90.58% accuracy, outperforming state-of-the-art
baselines that rely only on clinical variables. A tuned Gradient-Boosting (GB) model achieved comparable
performance with substantially lower computational requirements. Model interpretability was assessed
with SHAP and Grad-CAM, which identified CDR-SB, LDELTOTAL, and mPACC-TrailsB as the most
informative clinical features. The combination of strong predictive performance, computational efficiency,
and transparent interpretation positions the proposed approach as a promising open-source tool to
facilitate early AD diagnosis in clinical settings

Mario Alejandro Bravo-Ortiz. (2025). MarioBravo12/Alzheimer-s-Disease-Classification-Using-Clinical-Data1: Alzheimers Disease Classification Using Clinical Data (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.14894609
