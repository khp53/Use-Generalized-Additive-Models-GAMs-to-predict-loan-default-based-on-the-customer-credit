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
