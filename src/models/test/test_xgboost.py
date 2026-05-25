"""
Тестирование XGBoost
Запуск: python src/models/test/test_xgboost.py
"""

import os
import sys
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.utils.helpers import inverse_transform, calculate_metrics, save_metrics, plot_feature_importance


def test_xgboost():
    """Тестирование XGBoost"""
    print("\n=== ТЕСТИРОВАНИЕ XGBOOST ===")
    
    # Загрузка параметров
    with open('config/params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Загрузка данных (тестовая выборка)
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    # Загрузка имён признаков
    with open('models/feature_names.txt', 'r') as f:
        feature_names = f.read().strip().split('\n')
    
    # Загрузка модели
    model = xgb.XGBRegressor()
    model.load_model('models/xgboost.json')
    
    # Предсказания
    transform_type = params['data']['transform_type']
    y_pred = inverse_transform(model.predict(X_test), 0)
    y_true = inverse_transform(y_test, 0)
    
    # Метрики
    metrics = calculate_metrics(y_true, y_pred)
    print(f"Test: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.4f}")
    
    # Сохранение метрик (stage='test')
    save_metrics(metrics, 'xgboost', stage='test')
    
    # Визуализация
    os.makedirs('reports/figures', exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='k')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Реальные значения')
    plt.ylabel('Предсказанные значения')
    plt.title(f'XGBoost (Test)\nR² = {metrics["r2"]:.4f}')
    plt.tight_layout()
    plt.savefig('reports/figures/xgboost_predictions.png', dpi=150)
    plt.close()
    
    # Feature Importance
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    plot_feature_importance(importance, 'XGBoost - Feature Importance (Test)', 'xgboost_importance_test.png')
    
    return metrics


if __name__ == '__main__':
    test_xgboost()