# master_thesis
# ADHD Prediction Thesis – Analysis Scripts

This repository contains the R code used for the data analysis in my master's thesis on predicting adult ADHD symptom severity using cognitive task performance, chronotype and age.

## Overview

The project involves:
- Cognitive task feature aggregation and outlier removal
- Exploratory data analysis using clustering and dimensionality reduction
- Supervised learning with XGBoost (regression and classification)
- Model interpretation using SHAP and counterfactual analysis
- Generation of plots used in the thesis

## Repository Structure
scripts/
├── 02_exploratory_analysis.R # Histograms, PCA, GMM etc.
├── 03_regression_task.R # ASRS score prediction, SHAP, counterfactual analysis
├── 04_classification_task.R # ASRS screening classification, XGBoost vs k-NN
├── 05_thesis_figures.R # Plotting R-generated examples for the methods section in thesis
data/
├── cleaned_data.csv # Anonymized and preprocessed data
figures/
├── (optional) saved output figures if needed


> NOTE: The `01_data_cleaning.R` script is excluded due to participant privacy concerns. The uploaded dataset (`cleaned_data.csv`) is fully anonymized and post-cleaning.

This project was developed in R version `4.4.1`.

