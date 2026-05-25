"""
Обучение дерева решений
Запуск: python src/models/train/train_tree.py
"""

import os
import sys
import numpy as np
import yaml
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.utils.helpers import inverse_transform, calculate_metrics, save_metrics


def train_decision_tree():
    """Обучение дерева решений"""
    print("\n=== ОБУЧЕНИЕ ДЕРЕВА РЕШЕНИЙ ===")
    
    # Загрузка параметров
    with open('config/params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Загрузка данных
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    
    # Загрузка имён признаков
    with open('models/feature_names.txt', 'r') as f:
        feature_names = f.read().strip().split('\n')
    
    # Параметры модели
    dt_params = params['models']['decision_tree']
    model = DecisionTreeRegressor(
        max_depth=dt_params['max_depth'],
        min_samples_split=dt_params['min_samples_split'],
        min_samples_leaf=dt_params['min_samples_leaf'],
        random_state=params['base']['random_state']
    )
    
    # Обучение
    model.fit(X_train, y_train)
    
    # Предсказания на обучающей выборке
    transform_type = params['data']['transform_type']
    y_pred = inverse_transform(model.predict(X_train), 0)
    y_true = inverse_transform(y_train, 0)
    
    # Метрики
    metrics = calculate_metrics(y_true, y_pred)
    print(f"Train: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.4f}")
    
    # Сохранение метрик (stage='train')
    save_metrics(metrics, 'decision_tree', stage='train')
    
    # Сохранение модели
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/decision_tree.pkl')
    print("Модель сохранена в models/decision_tree.pkl")
    
    # Визуализация дерева
    os.makedirs('reports/figures', exist_ok=True)
    plt.figure(figsize=(20, 12))
    plot_tree(model, feature_names=feature_names, filled=True, rounded=True, max_depth=3, fontsize=8)
    plt.title('Decision Tree - первые 3 уровня')
    plt.tight_layout()
    plt.savefig('reports/figures/decision_tree.png', dpi=150)
    plt.close()
    
    return model, metrics


if __name__ == '__main__':
    train_decision_tree()