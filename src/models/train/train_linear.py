"""
Обучение линейной регрессии
Запуск: python src/models/train/train_linear.py
"""

import os
import sys
import numpy as np
import pandas as pd
import yaml
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.utils.helpers import inverse_transform, calculate_metrics, save_metrics


def train_linear_regression():
    """Обучение линейной регрессии"""
    print("\n=== ОБУЧЕНИЕ ЛИНЕЙНОЙ РЕГРЕССИИ ===")
    
    # Загрузка параметров
    with open('config/params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Загрузка данных
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    
    # Загрузка имён признаков
    with open('models/feature_names.txt', 'r') as f:
        feature_names = f.read().strip().split('\n')
    
    # Обучение модели
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Предсказания на обучающей выборке
    transform_type = params['data']['transform_type']
    y_pred = inverse_transform(model.predict(X_train), 0)
    y_true = inverse_transform(y_train, 0)
    
    # Метрики
    metrics = calculate_metrics(y_true, y_pred)
    print(f"Train: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.4f}")
    
    # Сохранение метрик (stage='train')
    save_metrics(metrics, 'linear_regression', stage='train')
    
    # Сохранение модели
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/linear_regression.pkl')
    print("Модель сохранена в models/linear_regression.pkl")
    
    # Коэффициенты
    coefs = pd.DataFrame({'feature': feature_names, 'coefficient': model.coef_})
    coefs = coefs.reindex(coefs.coefficient.abs().sort_values(ascending=False).index)
    coefs.to_csv('models/linear_coefficients.csv', index=False)
    
    print("\nТоп-5 коэффициентов:")
    for _, row in coefs.head(5).iterrows():
        print(f"  {row['feature']}: {row['coefficient']:.4f}")
    
    # Визуализация
    os.makedirs('reports/figures', exist_ok=True)
    plt.figure(figsize=(10, 6))
    colors = ['green' if c > 0 else 'red' for c in coefs['coefficient'].head(15)]
    sns.barplot(x='coefficient', y='feature', data=coefs.head(15), palette=colors)
    plt.title('Linear Regression - Коэффициенты признаков')
    plt.tight_layout()
    plt.savefig('reports/figures/linear_coefficients.png', dpi=150)
    plt.close()
    
    return model, metrics


if __name__ == '__main__':
    train_linear_regression()