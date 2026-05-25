"""
Тестирование нейронной сети
Запуск: python src/models/test/test_nn.py
"""

import os
import sys
import numpy as np
import yaml
import matplotlib.pyplot as plt
from tensorflow import keras

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.utils.helpers import inverse_transform, calculate_metrics, save_metrics


def test_neural_network():
    """Тестирование нейронной сети"""
    print("\n=== ТЕСТИРОВАНИЕ НЕЙРОННОЙ СЕТИ ===")
    
    # Загрузка параметров
    with open('config/params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Загрузка данных (тестовая выборка)
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    # Загрузка модели
    model = keras.models.load_model('models/neural_network.keras')
    
    # Предсказания
    transform_type = params['data']['transform_type']
    y_pred = inverse_transform(model.predict(X_test).flatten(), 0)
    y_true = inverse_transform(y_test, 0)
    
    # Метрики
    metrics = calculate_metrics(y_true, y_pred)
    print(f"Test: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.4f}")
    
    # Сохранение метрик (stage='test')
    save_metrics(metrics, 'neural_network', stage='test')
    
    # Визуализация
    os.makedirs('reports/figures', exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='k')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Реальные значения')
    plt.ylabel('Предсказанные значения')
    plt.title(f'Neural Network (Test)\nR² = {metrics["r2"]:.4f}')
    plt.tight_layout()
    plt.savefig('reports/figures/neural_network_predictions.png', dpi=150)
    plt.close()
    
    return metrics


if __name__ == '__main__':
    test_neural_network()