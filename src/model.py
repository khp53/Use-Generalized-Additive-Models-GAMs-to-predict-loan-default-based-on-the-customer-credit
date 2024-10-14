from pygam import LogisticGAM, s, f
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score

class CreditDefaultModel:
    def __init__(self):
        self.gam = None

    def create_model(self):
        # the spline terms and the factors are explained in details in the README.md file
        self.gam = LogisticGAM(
            s(0) + s(1) + s(2) + s(3) + s(4) + 
            s(5) + s(6) + s(7) + s(8) + s(9) + 
            f(10) + f(11) + f(12) + f(13) + 
            f(14) + f(15) + f(16) + f(17) + 
            f(18) + f(19) + f(20) + f(21) + 
            f(22) + f(23) + f(24) + f(25) + 
            f(26) + f(27) + f(28) + f(29) + 
            f(30) + f(31) + f(32) + f(33) + 
            f(34) + f(35) + f(36) + f(37) + 
            f(38) + f(39) + f(40) + f(41),
            max_iter=100,
            lam=0.65 # regularization parameter, why 0.65? because it was the best value found after multiple training an tuning
        )

    def fit_model(self, X_train, y_train):
        self.gam.gridsearch(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.gam.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        return accuracy, roc_auc
    
    def cross_validate(self, X_train, y_train, n_splits=5):
        k_fold = KFold(n_splits=n_splits, shuffle=True)
        cv_scores = cross_val_score(self.gam, X_train, y_train, cv=k_fold, scoring='accuracy')
        return cv_scores.mean()