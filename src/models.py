import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch.nn as nn

class WeightedXGB(xgb.XGBClassifier):
    def fit(self, X, y, **kwargs):
        # weighted using ratio of negative to positive instances
        count_calm = np.sum(y == 0)
        count_fragile = np.sum(y == 1)
        self.scale_pos_weight = count_calm / count_fragile
        return super().fit(X, y, **kwargs)

def get_baseline_configs():

    set_config(transform_output="pandas")

    def create_pipeline(estimator):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', estimator)
        ])
    
    models = {
        'LogisticRegression': create_pipeline(
            LogisticRegression(
                solver='liblinear',
                class_weight='balanced')
        ),
        'DecisionTree': create_pipeline(
            DecisionTreeClassifier(
                random_state=42,
                class_weight='balanced')
        ),
        'RandomForest': create_pipeline(
            RandomForestClassifier(
                n_jobs=2,
                random_state=42,
                class_weight='balanced')
        ),
        'LightGBM': create_pipeline(
            lgb.LGBMClassifier(
                n_jobs=2,
                random_state=42,
                class_weight='balanced')
        ),
        'XGBoost': create_pipeline(
            WeightedXGB(
                n_jobs=2,
                random_state=42,
                eval_metric='logloss')
        )
    }

    param_grids = {
        'LogisticRegression': {
            'model__C': [0.001, 0.01, 0.1, 1, 10]
        },
        'DecisionTree': {
            'model__max_depth': [3, 5, 10, None],
            'model__min_samples_leaf': [10, 50, 100],
            'model__criterion': ['gini', 'entropy']
        },
        'RandomForest': {
            'model__n_estimators': [100, 200], 
            'model__max_depth': [3, 5, 10], 
            'model__min_samples_leaf': [10, 50]
        },
        'LightGBM': {
            'model__n_estimators': [100, 300], 
            'model__learning_rate': [0.01, 0.05, 0.1], 
            'model__num_leaves': [30, 50]
        },
        'XGBoost': {
            'model__n_estimators': [100, 300], 
            'model__learning_rate': [0.01, 0.05, 0.1], 
            'model__max_depth': [3, 5, 7]
        }
    }
    
    return models, param_grids

class MarketRegimeGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(MarketRegimeGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class MarketRegimeCNN(nn.Module):
    def __init__(self, input_dim, seq_len, dropout, filters=32, kernel=3):
        super(MarketRegimeCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, filters, kernel_size=kernel, padding=2, dilation=2)
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(filters, 1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)
