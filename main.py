import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_processing import DataProcessor
from src.model import CreditDefaultModel
import matplotlib.pyplot as plt

file = 'german.data'
data_processor = DataProcessor(file)
data_processor.load_data()
data_processor.preprocess()

X, y = data_processor.map_x_y()

# Had to use train test split, cause k-fold cv does not work with nparray conversion
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train_np, X_test_np = data_processor.get_numpy_arrays(X_train, X_test)

model = CreditDefaultModel()
model.create_model()
model.fit_model(X_train_np, y_train)

# Evaluate the model
accuracy, roc_auc = model.evaluate_model(X_test_np, y_test)
print(f'Test Accuracy: {accuracy}')
print(f'Test ROC AUC: {roc_auc}')

# Perform k-fold cross-validation
cv_accuracy = model.cross_validate(X_train_np, y_train)
print(f'Cross-Validation Accuracy: {cv_accuracy}')

# Plot partial dependence for loan-to-income ratio
loan_to_income = model.gam.partial_dependence(term=9, X=X_train_np, width=0.95)

plt.figure(figsize=(10, 5))
plt.plot(loan_to_income[0], loan_to_income[1])
plt.title('Partial Dependence of Loan-to-Income Ratio')
plt.xlabel('Loan-to-Income Ratio')
plt.ylabel('Partial Dependence')
plt.grid(True)
plt.show()