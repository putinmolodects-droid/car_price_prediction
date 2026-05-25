"""
Тестирование дерева решений
Запуск: python src/models/test/test_tree.py
"""

import os
import sys
import numpy as np
import yaml
import joblib
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.utils.helpers import inverse_transform, calculate_metrics, save_metrics


def test_decision_tree():
    """Тестирование дерева решений"""
    print("\n=== ТЕСТИРОВАНИЕ ДЕРЕВА РЕШЕНИЙ ===")
    
    # Загрузка параметров
    with open('config/params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Загрузка данных (тестовая выборка)
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    # Загрузка модели
    model = joblib.load('models/decision_tree.pkl')
    
    # Предсказания
    transform_type = params['data']['transform_type']
    y_pred = inverse_transform(model.predict(X_test), 0)
    y_true = inverse_transform(y_test, 0)
    
    # Метрики
    metrics = calculate_metrics(y_true, y_pred)
    print(f"Test: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.4f}")
    
    # Сохранение метрик (stage='test')
    save_metrics(metrics, 'decision_tree', stage='test')
    
    # Визуализация
    os.makedirs('reports/figures', exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='k')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Реальные значения')
    plt.ylabel('Предсказанные значения')
    plt.title(f'Decision Tree (Test)\nR² = {metrics["r2"]:.4f}')
    plt.tight_layout()
    plt.savefig('reports/figures/decision_tree_predictions.png', dpi=150)
    plt.close()
    
    return metrics


if __name__ == '__main__':
    test_decision_tree()