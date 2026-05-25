"""
Сводная таблица метрик и сравнение моделей
Запуск: python src/models/evaluate.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.helpers import inverse_transform, calculate_metrics


def load_metrics(model_name, stage='test'):
    """
    Загрузка метрик из JSON файла
    """
    metrics_file = f'reports/metrics/{stage}/{model_name}.json'
    
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    else:
        print(f"Файл не найден: {metrics_file}")
        return None


def create_comparison_table():
    
    models = ['linear_regression', 'decision_tree', 'catboost', 'xgboost', 'neural_network']
    model_names = {
        'linear_regression': 'Linear Regression',
        'decision_tree': 'Decision Tree',
        'catboost': 'CatBoost',
        'xgboost': 'XGBoost',
        'neural_network': 'Neural Network (MLP)'
    }
    stages = ['train', 'val', 'test']
    
    # Сбор данных
    results = []
    
    for model in models:
        for stage in stages:
            metrics = load_metrics(model, stage)
            if metrics:
                results.append({
                    'Model': model_names[model],
                    'Stage': stage.upper(),
                    'MAE': metrics['mae'],
                    'MSE': metrics['mse'],
                    'RMSE': metrics['rmse'],
                    'R2': metrics['r2']
                })
    
    # Создание DataFrame
    df_results = pd.DataFrame(results)
    
    # Сводная таблица
    pivot_mae = df_results.pivot(index='Model', columns='Stage', values='MAE')
    pivot_r2 = df_results.pivot(index='Model', columns='Stage', values='R2')
    
    print("\n--- Средняя абсолютная ошибка (MAE) ---")
    print(pivot_mae.round(2).to_string())
    
    print("\n--- Коэффициент детерминации (R²) ---")
    print(pivot_r2.round(4).to_string())
    
    # Сохранение в CSV
    df_results.to_csv('model_comparison.csv', index=False)
    print("\nТаблица сохранена в 'model_comparison.csv'")
    
    return df_results, pivot_mae, pivot_r2


def plot_metrics_comparison(df_results, pivot_mae, pivot_r2):

    os.makedirs('reports/figures', exist_ok=True)
    
    # 1. Сравнение R² по выборкам
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # График R²
    ax = axes[0]
    pivot_r2.plot(kind='bar', ax=ax, color=['#2ecc71', '#3498db', '#e74c3c'])
    ax.set_title('Сравнение моделей по R²', fontsize=14, fontweight='bold')
    ax.set_xlabel('Модель')
    ax.set_ylabel('R²')
    ax.set_ylim(0, 1)
    ax.legend(title='Выборка')
    ax.grid(True, alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # График MAE
    ax = axes[1]
    pivot_mae.plot(kind='bar', ax=ax, color=['#2ecc71', '#3498db', '#e74c3c'])
    ax.set_title('Сравнение моделей по MAE', fontsize=14, fontweight='bold')
    ax.set_xlabel('Модель')
    ax.set_ylabel('MAE')
    ax.legend(title='Выборка')
    ax.grid(True, alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # График RMSE
    ax = axes[2]
    for model in df_results['Model'].unique():
        model_data = df_results[df_results['Model'] == model]
        ax.plot(model_data['Stage'], model_data['RMSE'], 'o-', label=model, linewidth=2, markersize=8)
    ax.set_title('Сравнение моделей по RMSE', fontsize=14, fontweight='bold')
    ax.set_xlabel('Выборка')
    ax.set_ylabel('RMSE')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/figures/metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("График сохранён в 'reports/figures/metrics_comparison.png'")
    
    # 2. Детальный анализ по моделям (train/val/test)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = {'TRAIN': '#2ecc71', 'VAL': '#3498db', 'TEST': '#e74c3c'}
    
    for idx, model in enumerate(df_results['Model'].unique()):
        ax = axes[idx]
        model_data = df_results[df_results['Model'] == model]
        
        x = np.arange(len(model_data))
        width = 0.25
        
        bars1 = ax.bar(x - width, model_data['MAE'], width, label='MAE', color='#3498db')
        bars2 = ax.bar(x, model_data['RMSE'], width, label='RMSE', color='#e74c3c')
        bars3 = ax.bar(x + width, model_data['R2'] * 10000, width, label='R²×10000', color='#2ecc71')
        
        ax.set_title(f'{model}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Выборка')
        ax.set_ylabel('Значение')
        ax.set_xticks(x)
        ax.set_xticklabels(model_data['Stage'])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    # Убираем пустые подграфики
    for idx in range(len(df_results['Model'].unique()), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Детальное сравнение метрик по моделям', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figures/metrics_detailed.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("График сохранён в 'reports/figures/metrics_detailed.png'")


def print_best_model_analysis(pivot_r2, pivot_mae):
    
    # Лучшая модель по R² на тестовой выборке
    if 'TEST' in pivot_r2.columns:
        best_r2_model = pivot_r2['TEST'].idxmax()
        best_r2_value = pivot_r2['TEST'].max()
        print(f"\nЛучшая модель по R² на тестовой выборке: {best_r2_model} (R² = {best_r2_value:.4f})")
    
    # Лучшая модель по MAE на тестовой выборке
    if 'TEST' in pivot_mae.columns:
        best_mae_model = pivot_mae['TEST'].idxmin()
        best_mae_value = pivot_mae['TEST'].min()
        print(f"Лучшая модель по MAE на тестовой выборке: {best_mae_model} (MAE = {best_mae_value:.2f})")
    
    # Проверка переобучения
    print("\n--- Проверка переобучения ---")
    for model in pivot_r2.index:
        if 'TRAIN' in pivot_r2.columns and 'TEST' in pivot_r2.columns:
            train_r2 = pivot_r2.loc[model, 'TRAIN']
            test_r2 = pivot_r2.loc[model, 'TEST']
            diff = (train_r2 - test_r2) * 100
            if diff > 10:
                print(f"  {model}: Высокое переобучение. R² train/test = {diff:.1f}%")
            elif diff > 5:
                print(f"  {model}:  Среднее переобучение. R² train/test = {diff:.1f}%")
            else:
                print(f"  {model}:  Низкое переобучение. Разница R² train/test = {diff:.1f}%")


def create_summary_report():
    
    # Загрузка всех метрик
    models = ['linear_regression', 'decision_tree', 'catboost', 'xgboost', 'neural_network']
    model_names = {
        'linear_regression': 'Линейная регрессия',
        'decision_tree': 'Дерево решений',
        'catboost': 'CatBoost',
        'xgboost': 'XGBoost',
        'neural_network': 'Нейронная сеть (MLP)'
    }
    
    summary_data = []
    
    for model in models:
        test_metrics = load_metrics(model, 'test')
        if test_metrics:
            summary_data.append({
                'Модель': model_names[model],
                'MAE': test_metrics['mae'],
                'RMSE': test_metrics['rmse'],
                'R²': test_metrics['r2']
            })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values('R²', ascending=False)
    
    print("\nРезультаты на тестовой выборке:")
    print(df_summary.to_string(index=False))
    
    # Определение лучшей модели
    best_model = df_summary.iloc[0]['Модель']
    best_r2 = df_summary.iloc[0]['R²']
    
    print("\n" + "=" * 80)
    print("ВЫВОД")
    print("=" * 80)
    print(f"Лучшая модель: {best_model}")
    print(f"R² = {best_r2:.4f}")
    print(f"MAE = {df_summary.iloc[0]['MAE']:.2f}")
    
    return df_summary


def main():
    # Создание сводной таблицы
    df_results, pivot_mae, pivot_r2 = create_comparison_table()
    
    # Визуализация
    plot_metrics_comparison(df_results, pivot_mae, pivot_r2)
    
    # Анализ лучшей модели
    print_best_model_analysis(pivot_r2, pivot_mae)
    
    df_summary = create_summary_report()
    
    df_summary.to_csv('reports/final_report.csv', index=False)
    print("\Итог сохранён в 'reports/final_report.csv'")


if __name__ == '__main__':
    main()