import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
    
    def load_data(self):
        names = [
            'existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount',
            'savings', 'employmentsince', 'installmentrate', 'personalstatus', 'otherdebtors',
            'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing',
            'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker', 'classification'
        ]
        self.data = pd.read_csv(self.file_path, names=names, delimiter=' ')
        print(f"Data loaded with shape: {self.data.shape}")

    def preprocess(self):
        self.data.classification.replace([1, 2], [1, 0], inplace=True)
        imputer_cont = SimpleImputer(strategy='median')
        imputer_cat = SimpleImputer(strategy='most_frequent')
        
        continuous_cols = ['age', 'creditamount', 'duration', 'installmentrate', 'residencesince', 'existingcredits', 'peopleliable']
        categorical_cols = [
            'existingchecking', 'credithistory', 'purpose', 'savings', 'employmentsince',
            'personalstatus', 'otherdebtors', 'property', 'otherinstallmentplans', 'housing', 'job', 
            'telephone', 'foreignworker'
        ]
        
        self.data[continuous_cols] = imputer_cont.fit_transform(self.data[continuous_cols])
        self.data[categorical_cols] = imputer_cat.fit_transform(self.data[categorical_cols])
        
        self.data['savings'] = pd.to_numeric(self.data['savings'], errors='coerce').fillna(0)
        self.data['creditutilizationratio'] = self.data['creditamount'] / (self.data['savings'] + 1)
        self.data['loantoincomeratio'] = self.data['creditamount'] / (self.data['duration'] + 1)
        
        self.data['creditamount'] = np.log1p(self.data['creditamount'])
        
        label_cols = ['personalstatus', 'foreignworker', 'telephone']
        for col in label_cols:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])
        
        # one-hot encoding
        self.data = pd.get_dummies(self.data, columns=categorical_cols, drop_first=True)
        
        # scaling continuous features
        scaler = StandardScaler()
        continuous_cols += ['creditutilizationratio', 'loantoincomeratio', 'creditamount']
        self.data[continuous_cols] = scaler.fit_transform(self.data[continuous_cols])

    def map_x_y(self):
        X = self.data.drop('classification', axis=1)
        y = self.data['classification']
        return X, y
    
    def get_numpy_arrays(self, X_train, X_test):
        X_train = X_train.astype({col: 'int' for col in X_train.select_dtypes(include=['bool']).columns})
        X_test = X_test.astype({col: 'int' for col in X_test.select_dtypes(include=['bool']).columns})
        return X_train.to_numpy(), X_test.to_numpy()