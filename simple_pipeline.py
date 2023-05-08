# For this pipeline to work, you need an array named 'data' with the cleaned and imputed datapoints for each run
# This pipeline also just uses base HPs and performance is thus questionable

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

# Placeholder for the data
data = np.array([])

# Create the necessary splits to train a classifier
labeled_data = data[data['label'] != -1]
unlabeled_data = data[data['label'] == -1]

X_labeled = labeled_data.drop('label', axis=1)
y_labeled = labeled_data['label']
X_unlabeled = unlabeled_data.drop('label', axis=1)

# Base models for our stacked classifier
base_models = [
    ('xgboost', XGBClassifier()),
    ('lightgbm', LGBMClassifier()),
    ('gradient_boosting', GradientBoostingClassifier()),
    ('svm', SVC(probability=True)),
    ('knn', KNeighborsClassifier())
]

final_estimator = XGBClassifier()

# Stacking the classifiers
stacked_classifier = StackingClassifier(estimators=base_models, final_estimator=final_estimator, cv=5, stack_method='predict_proba')

# Training
k = 10  # Number of folds
kfold = KFold(n_splits=k, shuffle=True, random_state=42)
fold_scores = []

for train_index, test_index in kfold.split(X_labeled, y_labeled):
    X_train, X_test = X_labeled.iloc[train_index], X_labeled.iloc[test_index]
    y_train, y_test = y_labeled.iloc[train_index], y_labeled.iloc[test_index]

    stacked_classifier.fit(X_train, y_train)
    y_pred = stacked_classifier.predict(X_test)
    fold_accuracy = accuracy_score(y_test, y_pred)
    fold_scores.append(fold_accuracy)
    print(f'Fold accuracy: {fold_accuracy}')

average_accuracy = np.mean(fold_scores)
print(f'Average accuracy across {k} folds: {average_accuracy}')

# Predicting
stacked_classifier.fit(X_labeled, y_labeled)
y_unlabeled_pred = stacked_classifier.predict(X_unlabeled)