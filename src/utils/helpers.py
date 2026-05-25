"""
Вспомогательные функции для проекта
"""

import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dvclive import Live


def inverse_transform(y_transformed, transform_type='sqrt'):
    """
    Обратное преобразование целевой переменной
    transform_type: 'none', 'log', 'sqrt'
    """
    if transform_type == 'log':
        return np.expm1(y_transformed)
    elif transform_type == 'sqrt':
        return np.square(y_transformed)
    else:
        return y_transformed


def apply_transform(y, transform_type='sqrt'):
    """
    Применение преобразования к целевой переменной
    """
    if transform_type == 'log':
        return np.log1p(y)
    elif transform_type == 'sqrt':
        return np.sqrt(y)
    else:
        return y


def calculate_metrics(y_true, y_pred):
    """Вычисление метрик"""
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }


def save_metrics(metrics_dict, model_name, stage='test'):
    """Сохраняет метрики через dvclive и JSON"""
    os.makedirs(f'reports/metrics/{stage}/{model_name}', exist_ok=True)
    
    # dvclive
    with Live(f'dvclive/{stage}/{model_name}', resume=True) as live:
        for key, value in metrics_dict.items():
            live.log_metric(f'{stage}.{key}', value)
    
    # JSON
    with open(f'reports/metrics/{stage}/{model_name}.json', 'w') as f:
        json.dump(metrics_dict, f, indent=4)


def extract_number(value):
    """Извлечение числа из строки"""
    if pd.isna(value):
        return np.nan
    match = re.search(r'(\d+(?:\.\d+)?)', str(value))
    return float(match.group(1)) if match else np.nan


def extract_max_torque(value):
    """Извлечение максимального числа из диапазона torque"""
    if pd.isna(value):
        return np.nan
    numbers = re.findall(r'(\d+(?:\.\d+)?)', str(value))
    if not numbers:
        return np.nan
    return max(float(n) for n in numbers)


def plot_feature_importance(feature_importance, title, filename, top_n=15):
    """Визуализация важности признаков"""
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(top_n)
    sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
    plt.title(title)
    plt.xlabel('Важность')
    plt.ylabel('Признак')
    plt.tight_layout()
    plt.savefig(f'reports/figures/{filename}', dpi=150)
    plt.close()


def plot_learning_curves(history, filename='nn_learning_curves.png'):
    """Кривые обучения нейросети"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE')
    axes[0].legend()
    
    axes[1].plot(history.history['mae'], label='Train MAE')
    axes[1].plot(history.history['val_mae'], label='Val MAE')
    axes[1].set_title('MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f'reports/figures/{filename}', dpi=150)
    plt.close()


def plot_weights_histogram(model, filename='nn_weights_hist.png'):
    """Гистограмма весов нейросети"""
    plt.figure(figsize=(12, 5))
    layer_num = 1
    for i, layer in enumerate(model.layers):
        if len(layer.get_weights()) > 0:
            weights = layer.get_weights()[0].flatten()
            plt.subplot(1, min(4, len(model.layers)), layer_num)
            plt.hist(weights, bins=50, alpha=0.7)
            plt.title(f'Layer {layer_num}')
            layer_num += 1
    plt.tight_layout()
    plt.savefig(f'reports/figures/{filename}', dpi=150)
    plt.close()

# def save_metrics(metrics_dict, model_name, stage='test'):
#     """
#     Сохраняет метрики в соответствующую директорию
#     stage: 'train', 'val', 'test'
#     """
#     # Создаём директорию для соответствующего этапа
#     os.makedirs(f'reports/metrics/{stage}', exist_ok=True)
    
#     # Сохраняем файл
#     metrics_file = f'reports/metrics/{stage}/{model_name}.json'
    
#     with open(metrics_file, 'w') as f:
#         json.dump(metrics_dict, f, indent=4)
    
#     print(f"Метрики ({stage}) сохранены в {metrics_file}")
    
# def save_metrics(metrics_dict, model_name, stage='test'):
#     """
#     Сохраняет метрики в JSON файл для DVC
#     """
#     os.makedirs('reports/metrics', exist_ok=True)
#     dvc_metrics={}
#     # Формат для DVC metrics
#     dvc_metrics[stage] = {
#         'mae': metrics_dict['mae'],
#         'mse': metrics_dict['mse'],
#         'rmse': metrics_dict['rmse'],
#         'r2': metrics_dict['r2']
#     }
    
#     # Сохраняем JSON файл
#     with open(f'reports/metrics/{model_name}.json', 'w') as f:
#         json.dump(dvc_metrics, f, indent=4)
    
#     print(f"Метрики сохранены в reports/metrics/{model_name}.json")
    
# def save_all_metrics(metrics_dict, model_name, stage='test'):
#     """
#     Сохраняет метрики в JSON файл для DVC
#     stage: 'train', 'val', 'test'
#     """
#     os.makedirs('reports/metrics', exist_ok=True)
    
#     # Загружаем существующие метрики, если есть
#     metrics_file = f'reports/metrics/{model_name}.json'
#     all_metrics = {}
    
#     if os.path.exists(metrics_file):
#         with open(metrics_file, 'r') as f:
#             all_metrics = json.load(f)
    
#     # Добавляем новые метрики
#     all_metrics[stage] = {
#         'mae': metrics_dict['mae'],
#         'mse': metrics_dict['mse'],
#         'rmse': metrics_dict['rmse'],
#         'r2': metrics_dict['r2']
#     }
    
#     # Сохраняем
#     with open(metrics_file, 'w') as f:
#         json.dump(all_metrics, f, indent=4)
    
#     print(f"Метрики ({stage}) сохранены в {metrics_file}")