# Employee Attrition Prediction Model

This project builds a binary classification model to predict whether an employee will stay or leave the company. It also provides a feature importance analysis to help HR teams understand the key drivers behind employee turnover.

## Overview

The script uses a **Random Forest Classifier** to analyze historical HR data. Random Forest was chosen for its robust performance with mixed data types and its built-in ability to calculate feature importance, making the results highly interpretable for business stakeholders.

## Dataset

The model requires the IBM HR Analytics Employee Attrition & Performance dataset:
* **Filename:** `WA_Fn-UseC_-HR-Employee-Attrition.csv`
* Ensure this file is placed in the same directory as your Python script before running.

## Prerequisites

You need Python installed along with the following libraries. You can install them via pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn

Expected Outputs Explained:
When you run the script, it will generate two main outputs to help you understand both how well the model works and why employees are leaving.

1. The Classification Report (Console Output)
Printed directly to your terminal, this report grades the model's predictive performance using the following metrics:

Accuracy: The overall percentage of employees the model correctly predicted (e.g., 84%).

Precision: When the model predicts someone will leave, how often is it correct?

Recall: Out of all the people who actually left the company, how many did the model successfully identify? (Note: Because most employees stay, predicting the minority who leave is harder, which usually results in a lower recall score for the "Leave" class).

F1-Score: A balanced average of Precision and Recall.

2. Feature Importance Chart (Saved Image)
The script automatically generates and saves a bar chart named feature_importance.png in your project folder.

What you will see in this chart:

A horizontal bar chart ranking the top 15 data points (features) that most heavily influence an employee's decision to quit.

The longer the bar, the more weight the model gave to that specific factor.

Typically, you will see factors like MonthlyIncome, Age, TotalWorkingYears, and YearsAtCompany at the top.