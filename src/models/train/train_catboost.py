"""
Обучение CatBoost
Запуск: python src/models/train/train_catboost.py
"""

import os
import sys
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.utils.helpers import inverse_transform, calculate_metrics, save_metrics, plot_feature_importance


def train_catboost():
    """Обучение CatBoost"""
    print("\n=== ОБУЧЕНИЕ CATBOOST ===")
    
    # Загрузка параметров
    with open('config/params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Загрузка данных (raw, без one-hot)
    X_train_raw = pd.read_csv('data/processed/X_train_raw.csv')
    y_train = np.load('data/processed/y_train.npy')
    
    # Категориальные признаки
    cat_cols = params['data']['categorical_features']
    cat_features_idx = [X_train_raw.columns.get_loc(c) for c in cat_cols if c in X_train_raw.columns]
    
    # Параметры модели
    cb_params = params['models']['catboost']
    model = CatBoostRegressor(
        iterations=cb_params['iterations'],
        learning_rate=cb_params['learning_rate'],
        depth=cb_params['depth'],
        l2_leaf_reg=cb_params['l2_leaf_reg'],
        cat_features=cat_features_idx,
        verbose=False,
        random_seed=params['base']['random_state']
    )
    
    # Обучение
    model.fit(X_train_raw, y_train)
    
    # Предсказания на обучающей выборке
    transform_type = params['data']['transform_type']
    y_pred = inverse_transform(model.predict(X_train_raw), 0)
    y_true = inverse_transform(y_train, 0)
    
    # Метрики
    metrics = calculate_metrics(y_true, y_pred)
    print(f"Train: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.4f}")
    
    # Сохранение метрик (stage='train')
    save_metrics(metrics, 'catboost', stage='train')
    
    # Сохранение модели
    os.makedirs('models', exist_ok=True)
    model.save_model('models/catboost.cbm')
    print("Модель сохранена в models/catboost.cbm")
    
    # Feature Importance
    importance = pd.DataFrame({
        'feature': X_train_raw.columns,
        'importance': model.get_feature_importance()
    }).sort_values('importance', ascending=False)
    importance.to_csv('models/catboost_feature_importance.csv', index=False)
    plot_feature_importance(importance, 'CatBoost - Feature Importance', 'catboost_importance.png')
    
    print("\nТоп-5 важных признаков:")
    for _, row in importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return model, metrics


if __name__ == '__main__':
    train_catboost()