"""
Валидация CatBoost
Запуск: python src/models/validate/validate_catboost.py
"""

import os
import sys
import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostRegressor

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.utils.helpers import inverse_transform, calculate_metrics, save_metrics


def validate_catboost():
    """Валидация CatBoost"""
    print("\n=== ВАЛИДАЦИЯ CATBOOST ===")
    
    # Загрузка параметров
    with open('config/params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Загрузка данных (raw, без one-hot) - валидационная выборка
    X_val_raw = pd.read_csv('data/processed/X_val_raw.csv')
    y_val = np.load('data/processed/y_val.npy')
    
    # Загрузка модели
    model = CatBoostRegressor()
    model.load_model('models/catboost.cbm')
    
    # Предсказания
    transform_type = params['data']['transform_type']
    y_pred = inverse_transform(model.predict(X_val_raw), 0)
    y_true = inverse_transform(y_val, 0)
    
    # Метрики
    metrics = calculate_metrics(y_true, y_pred)
    print(f"Validation: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.4f}")
    
    # Сохранение метрик (stage='val')
    save_metrics(metrics, 'catboost', stage='val')
    
    return metrics


if __name__ == '__main__':
    validate_catboost()