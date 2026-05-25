"""
Тестирование CatBoost
Запуск: python src/models/test/test_catboost.py
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


def test_catboost():
    """Тестирование CatBoost"""
    print("\n=== ТЕСТИРОВАНИЕ CATBOOST ===")
    
    # Загрузка параметров
    with open('config/params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Загрузка данных (raw, без one-hot) - тестовая выборка
    X_test_raw = pd.read_csv('data/processed/X_test_raw.csv')
    y_test = np.load('data/processed/y_test.npy')
    
    # Загрузка модели
    model = CatBoostRegressor()
    model.load_model('models/catboost.cbm')
    
    # Предсказания
    transform_type = params['data']['transform_type']
    y_pred = inverse_transform(model.predict(X_test_raw), 0)
    y_true = inverse_transform(y_test, 0)
    
    # Метрики
    metrics = calculate_metrics(y_true, y_pred)
    print(f"Test: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.4f}")
    
    # Сохранение метрик (stage='test')
    save_metrics(metrics, 'catboost', stage='test')
    
    # Визуализация
    os.makedirs('reports/figures', exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='k')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Реальные значения')
    plt.ylabel('Предсказанные значения')
    plt.title(f'CatBoost (Test)\nR² = {metrics["r2"]:.4f}')
    plt.tight_layout()
    plt.savefig('reports/figures/catboost_predictions.png', dpi=150)
    plt.close()
    
    # Feature Importance
    importance = pd.DataFrame({
        'feature': X_test_raw.columns,
        'importance': model.get_feature_importance()
    }).sort_values('importance', ascending=False)
    plot_feature_importance(importance, 'CatBoost - Feature Importance (Test)', 'catboost_importance_test.png')
    
    return metrics


if __name__ == '__main__':
    test_catboost()