import json
import os
import argparse

import numpy as np
from utils import load_adult, load_bank
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('split_id', type=str)
parser.add_argument('model', type=str, default='xgboost')
args = parser.parse_args()

if args.dataset == 'adult':
    X_train, X_test, y_train, y_test = load_adult()
elif args.dataset == 'bank':
    X_train, X_test, y_train, y_test = load_bank()
else:
    raise ValueError(f"Unknown dataset {args.dataset}, should be adult or bank")

curr_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(curr_dir, 'saved_models', args.dataset, args.split_id)
assert os.path.exists(base_dir), f"{base_dir} does not exist"

with open(os.path.join(base_dir, "train_data_indices.json"), 'r') as f:
    train_data_indices = json.load(f)

X_train_subset = X_train.iloc[train_data_indices]
y_train_subset = y_train.iloc[train_data_indices]

if args.model != 'xgboost':
    raise ValueError(f"Unknown model {args.model}, should be xgboost")

param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.01, 0.05],
    'n_estimators': [50, 100, 200],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_weight': [1, 5, 10],
    'gamma': [0, 0.1, 0.2],
}

model = XGBClassifier(objective='binary:logistic', random_state=42, n_jobs=1, enable_categorical=True)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_subset, y_train_subset)
model = grid_search.best_estimator_

train_scores = model.predict_proba(X_train).astype(float)
test_scores = model.predict_proba(X_test).astype(float)
model.get_booster().save_model(os.path.join(base_dir, f"{args.model}.json"))
np.save(os.path.join(base_dir, f"{args.model}_train_scores.npy"), train_scores)
np.save(os.path.join(base_dir, f"{args.model}_test_scores.npy"), test_scores)
