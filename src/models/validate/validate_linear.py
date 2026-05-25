"""
Валидация линейной регрессии
Запуск: python src/models/validate/validate_linear.py
"""

import os
import sys
import numpy as np
import yaml
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.utils.helpers import inverse_transform, calculate_metrics, save_metrics


def validate_linear_regression():
    """Валидация линейной регрессии"""
    print("\n=== ВАЛИДАЦИЯ ЛИНЕЙНОЙ РЕГРЕССИИ ===")
    
    # Загрузка параметров
    with open('config/params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Загрузка данных (валидационная выборка)
    X_val = np.load('data/processed/X_val.npy')
    y_val = np.load('data/processed/y_val.npy')
    
    # Загрузка модели
    model = joblib.load('models/linear_regression.pkl')
    
    # Предсказания
    transform_type = params['data']['transform_type']
    y_pred = inverse_transform(model.predict(X_val), 0)
    y_true = inverse_transform(y_val, 0)
    
    # Метрики
    metrics = calculate_metrics(y_true, y_pred)
    print(f"Validation: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.4f}")
    
    # Сохранение метрик (stage='val')
    save_metrics(metrics, 'linear_regression', stage='val')
    
    return metrics


if __name__ == '__main__':
    validate_linear_regression()