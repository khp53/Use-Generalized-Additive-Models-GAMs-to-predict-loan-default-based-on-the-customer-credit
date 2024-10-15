# Use Generalized Additive Models (GAMs) to Predict Loan Default Based on Customer Credit

## Overview

This project utilizes Generalized Additive Models (GAMs) to predict loan default based on customer credit data from the German Credit Data dataset. The aim is to develop an interpretable model that provides insights into the relationship between various financial and demographic features and the likelihood of loan default.

## Table of Contents

1. [Requirements](#requirements)
2. [Data](#data)
3. [Implementation](#implementation)
   - [Data Processing](#data-processing)
   - [Model Training](#model-training)
   - [Model Evaluation](#model-evaluation)
   - [Partial Dependence Plots](#partial-dependence-plots)
4. [Usage](#usage)
5. [References](#references)

## Requirements

Ensure you have the following packages installed:

- Python 3.11 or higher
- pandas
- numpy
- scikit-learn
- pygam
- matplotlib
- ucimlrepo

You can install the required packages using pip, a requirements.txt file has been provided with the project so that one can easily install the needed requirements.

```bash
pip install -r requirements.txt
```

```bash
if you are using ucimlrepo directly to fetch data, the dataset id is 144
```

## Data Description
The dataset contains 1,000 observations and 20 attributes, including financial features and a classification label indicating whether the credit is good or bad. The columns are:

    - existingchecking
    - duration
    - credithistory
    - purpose
    - creditamount
    - savings
    - employmentsince
    - installmentrate
    - personalstatus
    - otherdebtors
    - residencesince
    - property
    - age
    - otherinstallmentplans
    - housing
    - existingcredits
    - job
    - peopleliable
    - telephone
    - foreignworker
    - classification (Target variable)

For more accurate description of each column please follow the link provided in the reference section.

# Implementation
## Data Processing
Data processing involves the following steps:

1. Loading Data: Load the dataset using pandas.
2. Handling Missing Values: Impute missing values for continuous and categorical features.
3. Feature Engineering: Create new features, such as the loan-to-income ratio and credit utilization ratio.
4. Encoding Categorical Variables: Apply label encoding for binary features and one-hot encoding for other categorical features.
5. Feature Scaling: Standardize continuous features using StandardScaler.

```python
# Load dataset
data = pd.read_csv('german.data', names=column_names, delimiter=' ')

# Even though, this dataset does not have a missing values, the missing values are still handelled
# using simple imputer for continious data using median strategy and for categorical data using most frequent stretagy

# Impute continuous and categorical columns separately
continuous_cols = ['age', 'creditamount', 'duration', 'installmentrate', 'residencesince', 'existingcredits', 'peopleliable']
categorical_cols = [
    'existingchecking', 'credithistory', 'purpose', 'savings', 'employmentsince',
    'personalstatus', 'otherdebtors', 'property', 'otherinstallmentplans', 'housing', 'job', 
    'telephone', 'foreignworker'
]

self.data[continuous_cols] = imputer_cont.fit_transform(self.data[continuous_cols])
self.data[categorical_cols] = imputer_cat.fit_transform(self.data[categorical_cols])

# Ensure continuous features are numeric
self.data[continuous_cols] = self.data[continuous_cols].apply(pd.to_numeric, errors='coerce')
```
Three new columns has been added to the  continous columns, which were calculated using savings, credit amount and duration. 
These were calculated to understand borrowing behaviour of a person, risk assesment and impact on their credit score.

1. Understanding Borrowing Behavior: The credit utilization ratio calculates how much of a person's available credit they are using. It indicates how heavily an individual relies on credit usages. A higher ratio suggests greater dependency on credit, which can be a red flag for lenders.
2. Risk Assessment: Lenders often prefer borrowers with a lower credit utilization ratio, as it suggests that the borrower is managing their credit responsibly. A high utilization ratio can indicate financial distress or overextension.
3. Impact on Credit Score: Credit utilization is a significant factor in credit scoring models. Keeping this ratio low is generally favorable for maintaining a good credit score. Papers suggest 30% or below utilization is better for credit score.

```python
continuous_cols += ['creditutilizationratio', 'loantoincomeratio', 'creditamount']
```

## Model Training
The model is trained using a logistic GAM. The features are defined as smoothing splines or factors, depending on whether they are continuous or categorical.

1. Continuous Variables with s(): I used splines (s()) for the continuous variables to capture nonlinear trends. These are:

- 'duration
- 'age'
- 'creditamount'
- 'creditutilizationratio'
- 'loantoincomeratio'
- 'installmentrate'
- 'residencesince'
- 'existingcredits'
- 'peopleliable'

2. Categorical Variables with f(): Use factors (f()) for categorical variables, including the one-hot encoded fields like:

'existingchecking_A12', 'existingchecking_A13', 'existingchecking_A14'
'credithistory_A31', 'credithistory_A32', etc.
'purpose_A41', 'purpose_A42', etc.

Here we used one-hot encoding to convert categorical variables into a format that can be provided to our gam model to understand and improve predictions.

```python
# Define the logistic GAM model
gam = LogisticGAM(
    s(0) + s(1) + f(2) + ... + f(n),
    max_iter=100,
    lam=0.65,
)

# Fit the model to the training data
gam.fit(X_train_np, y_train)
```

## Model Evaluation
The model performance is evaluated using k-fold cross-validation, and the accuracy and AUC-ROC scores are computed.

```python
kf = KFold(n_splits=5, shuffle=True)
cv_scores = cross_val_score(gam, X_train, y_train, cv=kf, scoring='accuracy')
```
The Scores are following:

```
Test Accuracy: 0.75
Test ROC AUC: 0.6375466638624534
Cross-Validation Accuracy: 0.7014285714285714
```
This model's text accuracy is 0.75, the model correctly predicted whether a loan would default or not 75% of the time on the test data. This indicates that the model is fairly reliable, though there's still room for improvement, as it misclassifies 1 in 4 cases.

The ROC AUC score of 0.637 suggests that the model is moderately effective at distinguishing between those who will default and those who won’t. A score of 0.5 means random guessing, while 1.0 represents perfect classification, so 0.637 is somewhat above average but not particularly strong. The model could struggle with borderline or hard-to-classify cases.

The cross-validation accuracy of 70% indicates that the model performs reasonably consistently across different data folds, reducing the risk of overfitting. However, it also suggests that the model might not generalize perfectly to new, unseen data.

## Partial Dependence Plots
The partial dependence plots visualize the relationship between features and the target variable, providing insights into how changes in a feature impact the model's predictions.

The hyperparameater lamda's (smoothing penalty) value was choosen based on how the relationship and the line changed while observing the plot.
If lambda is too low, the model may overfit (creating too wiggly splines). If lambda is too high, the model may underfit (oversmoothing the splines).

```python
loan_to_income = model.gam.partial_dependence(term=9, X=X_train_np, width=0.95)
plt.plot(loan_to_income[0], loan_to_income[1])
plt.title('Partial Dependence of Loan-to-Income Ratio')
plt.xlabel('Loan-to-Income Ratio')
plt.ylabel('Partial Dependence')
plt.grid(True)
plt.show()
```

Here, gam.partial_dependence() is a method that computes the partial dependence of a specific feature on the target variable.

1. term=9: This specifies which term (or feature) to compute the partial dependence for. In this case, term=9 corresponds to the loan-to-income ratio in the model.
2. X=X_train_np: This is the input data for which to compute the partial dependence. It should have a 2D array or DataFrame containing the features used in the model.
3. width=0.95: This parameter defines the width of the confidence interval around the partial dependence estimate. A value of 0.95 indicates a 95% confidence interval.

In the loan_to_income there will be two arrays. The first array is the unique values of the loan-to-income ratio. The second array represents the average predicted response (partial dependency) across all other features in the model, averaged over the specified feature in this case loan to income.

The plot shows that as the loan-to-income ratio increases, the partial dependence also increases, this could indicate a higher likelihood of default. Conversely, if the partial dependence decreases as the loan-to-income ratio increases, it could suggest a lower likelihood of default.
Observing the shape of the plot can also indicate the presence of nonlinear relationships. For instance, if the plot has curves or thresholds, it suggests that the effect of the loan-to-income ratio on the prediction varies at different levels of the ratio. Which we will see if you choose age, creditamount or credit ration terms. to do that just replace the term index value from 9 to any other index values of the continious variables. By printing data.columns we can determine which index term to use.

```python
print(data.columns)
```
Output:
```
Index(['duration', 'creditamount', 'installmentrate', 'residencesince', 'age',
       'existingcredits', 'peopleliable', 'classification',
       'creditutilizationratio', 'loantoincomeratio', 'existingchecking_A12',
       'existingchecking_A13', 'existingchecking_A14', 'credithistory_A31',
       'credithistory_A32', 'credithistory_A33', 'credithistory_A34',
       'purpose_A41', 'purpose_A410', 'purpose_A42', 'purpose_A43',
       'purpose_A44', 'purpose_A45', 'purpose_A46', 'purpose_A48',
       'purpose_A49', 'employmentsince_A72', 'employmentsince_A73',
       'employmentsince_A74', 'employmentsince_A75', 'personalstatus_1',
       'personalstatus_2', 'personalstatus_3', 'otherdebtors_A102',
       'otherdebtors_A103', 'property_A122', 'property_A123', 'property_A124',
       'otherinstallmentplans_A142', 'otherinstallmentplans_A143',
       'housing_A152', 'housing_A153', 'job_A172', 'job_A173', 'job_A174',
       'telephone_1', 'foreignworker_1'])
```

## Usage
To run the project, execute the main.py file. Ensure the required libraries are installed and the dataset path is correct.

```bash
python3 main.py
```

## References

1. Hastie, T., & Tibshirani, R. (1987). Generalized Additive Models: Some Applications. Journal of the American Statistical Association, 82(398), 371–386. https://doi.org/10.1080/01621459.1987.10478440

2. Francisco Louzada, Anderson Ara, Guilherme B. Fernandes,
Classification methods applied to credit scoring: Systematic review and overall comparison,
Surveys in Operations Research and Management Science,
Volume 21, Issue 2,
2016,
Pages 117-134,
ISSN 1876-7354,
https://doi.org/10.1016/j.sorms.2016.10.001.

3. Thomas, Lyn C., Consumer Credit Models: Pricing, Profit and Portfolios (Oxford, 2009; online edn, Oxford Academic, 1 May 2009), https://doi.org/10.1093/acprof:oso/9780199232130.001.1

4. Dataset: http://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

5. Data preprocessing help was taken from: https://www.kaggle.com/code/hendraherviawan/predicting-german-credit-default

6. pygam doc: https://pygam.readthedocs.io/en/latest/

7. The test result explanation base line was googled and based on that and the actual model's performance metrics explanation was provided. 

8. Error resolving was done using stackoverflow and chatgpt