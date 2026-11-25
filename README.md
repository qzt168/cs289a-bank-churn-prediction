# CS289A Bank Churn Prediction

Graduate Project (Path 2) for CS289A: Introduction to Machine Learning, UC Berkeley, Fall 2026.

## Project Overview

This project compares four machine learning models for bank customer churn prediction using the [Kaggle Playground Series S4E1 dataset](https://www.kaggle.com/competitions/playground-series-s4e1):

+ Logistic Regression (LR)
+ Linear Support Vector Machine (SVM)
+ Random Forest (RF)
+ Multi-Layer Perceptron (MLP)

We evaluate predictive performance and computational cost, and investigate the effects of hyperparameter tuning, feature ablation, feature engineering, training data size, and MLP architecture and regularization.

## Main Results

| Model               | ROC-AUC (Original) | ROC-AUC (Engineered) | Accuracy (Original) | F1 (Original) |
| ------------------- | -----------------: | -------------------: | ------------------: | ------------: |
| Logistic Regression |             0.8128 |               0.8691 |              0.8320 |        0.4847 |
| Linear SVM          |             0.8134 |               0.8685 |              0.8301 |        0.4548 |
| Random Forest       |         **0.8855** |           **0.8860** |          **0.8634** |        0.6164 |
| MLP                 |             0.8853 |               0.8852 |              0.8630 |    **0.6213** |

All results are evaluated on the held-out test set.

Random Forest and MLP achieve similar ROC-AUC scores and outperform the linear models on the original features. Feature engineering substantially improves the performance of the linear models, while providing limited ROC-AUC improvements for RF and MLP.

## Project Report

See the final project report for detailed methodology, experimental results, and discussion.

## Team Members

+ Qizhi Tian ([qizhi_tian@berkeley.edu](mailto:qizhi_tian@berkeley.edu))
+ Junyu Li ([junyul@berkeley.edu](mailto:junyul@berkeley.edu))
+ Lorraine Xu ([lorraine_xu@berkeley.edu](mailto:lorraine_xu@berkeley.edu))
