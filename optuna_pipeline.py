# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import optuna
from data import data_preprocessing, split_by_user
import pickle
import sys

df_trainings = pd.read_csv('/Users/luc/Documents/Coding Adventures/datathon/data/trainings.csv')
df_additional_features = pd.read_csv('/Users/luc/Documents/Coding Adventures/datathon/data/new_features.csv')
features = data_preprocessing(df_trainings, df_additional_features, remove_type=False)
with open('/Users/luc/Documents/Coding Adventures/datathon/data/runner_embeddings.pickle', 'rb') as f:
    runner_embeddings = pickle.load(f)


# Create the necessary splits to train a classifier
df = pd.read_csv('features_with_embeddings.csv')

labeled_data = df[~df['type'].isna()]
unlabeled_data = df[df['type'].isna()]

X_labeled = labeled_data.drop('type', axis=1)
y_labeled = labeled_data['type']

X_unlabeled = unlabeled_data.drop('type', axis=1)

# Define an objective function to optimize with Optuna
def objective(trial):
    # Define hyperparameter search spaces for the base models
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.001, 0.1),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 8),
    }
    lgb_params = {
        'n_estimators': trial.suggest_int('lgb_n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('lgb_learning_rate', 0.001, 0.1),
        'max_depth': trial.suggest_int('lgb_max_depth', 3, 8),
    }
    gbc_params = {
        'n_estimators': trial.suggest_int('gbc_n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('gbc_learning_rate', 0.001, 0.1),
        'max_depth': trial.suggest_int('gbc_max_depth', 3, 8),
    }
    svm_params = {
        'C': trial.suggest_float('svm_C', 1e-4, 1e4, log=True),
        'kernel': trial.suggest_categorical('svm_kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
    }
    knn_params = {
        'n_neighbors': trial.suggest_int('knn_n_neighbors', 1, 30),
    }

    # Create base models with optimized hyperparameters
    base_models = [
        ('xgboost', XGBClassifier(**xgb_params)),
        ('lightgbm', LGBMClassifier(**lgb_params)),
        ('gradient_boosting', GradientBoostingClassifier(**gbc_params)),
        ('svm', SVC(probability=True, **svm_params)),
        ('knn', KNeighborsClassifier(**knn_params))
    ]

    # Define hyperparameters for the final logistic regression model
    final_xgb_params = {
        'n_estimators': trial.suggest_int('final_xgb_n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('final_xgb_learning_rate', 0.001, 0.1),
        'max_depth': trial.suggest_int('final_xgb_max_depth', 3, 8),
    }

    final_estimator = XGBClassifier(**final_xgb_params)

    # Create the StackingClassifier with optimized hyperparameters
    stacked_classifier = StackingClassifier(estimators=base_models, final_estimator=final_estimator, cv=5, stack_method='predict_proba')

    # Evaluate the StackingClassifier using cross-validation
    cv_scores = cross_val_score(stacked_classifier, X_labeled, y_labeled, cv=5, scoring='accuracy')
    print(np.mean(cv_scores))
    return np.mean(cv_scores)

# Create the Optuna study and run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

# Retrieve the best hyperparameters
best_params = study.best_params

# Create the models with the best hyperparameters
best_base_models = [
    ('xgboost', XGBClassifier(**{key: best_params[key] for key in best_params if key.startswith('xgb_')})),
    ('lightgbm', LGBMClassifier(**{key: best_params[key] for key in best_params if key.startswith('lgb_')})),
    ('gradient_boosting', GradientBoostingClassifier(**{key: best_params[key] for key in best_params if key.startswith('gbc_')})),
    ('svm', SVC(probability=True, **{key: best_params[key] for key in best_params if key.startswith('svm_')})),
    ('knn', KNeighborsClassifier(**{key: best_params[key] for key in best_params if key.startswith('knn_')}))
]

best_final_estimator = XGBClassifier(**{key: best_params[key] for key in best_params if key.startswith('final_xgb_')})

# Stacking the classifiers with the best hyperparameters
best_stacked_classifier = StackingClassifier(estimators=best_base_models, final_estimator=best_final_estimator, cv=5, stack_method='predict_proba')

# Training
k = 5  # Number of folds
kfold = KFold(n_splits=k, shuffle=True, random_state=42)
fold_scores = []

for train_index, test_index in kfold.split(X_labeled, y_labeled):
    X_train, X_test = X_labeled.iloc[train_index], X_labeled.iloc[test_index]
    y_train, y_test = y_labeled.iloc[train_index], y_labeled.iloc[test_index]

    best_stacked_classifier.fit(X_train, y_train)
    y_pred = best_stacked_classifier.predict(X_test)
    fold_accuracy = accuracy_score(y_test, y_pred)
    fold_scores.append(fold_accuracy)
    print(f'Fold accuracy: {fold_accuracy}')

average_accuracy = np.mean(fold_scores)
print(f'Average accuracy across {k} folds: {average_accuracy}')

# Predicting
best_stacked_classifier.fit(X_labeled, y_labeled)
y_unlabeled_pred = best_stacked_classifier.predict(X_unlabeled)
