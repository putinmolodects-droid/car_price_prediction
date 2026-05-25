"""
Обучение XGBoost
Запуск: python src/models/train/train_xgboost.py
"""

import os
import sys
import numpy as np
import pandas as pd
import yaml
import xgboost as xgb

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.utils.helpers import inverse_transform, calculate_metrics, save_metrics, plot_feature_importance


def train_xgboost():
    """Обучение XGBoost"""
    print("\n=== ОБУЧЕНИЕ XGBOOST ===")
    
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
    xgb_params = params['models']['xgboost']
    model = xgb.XGBRegressor(
        n_estimators=xgb_params['n_estimators'],
        max_depth=xgb_params['max_depth'],
        learning_rate=xgb_params['learning_rate'],
        reg_lambda=xgb_params['reg_lambda'],
        random_state=params['base']['random_state'],
        verbosity=0
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
    save_metrics(metrics, 'xgboost', stage='train')
    
    # Сохранение модели
    os.makedirs('models', exist_ok=True)
    model.save_model('models/xgboost.json')
    print("Модель сохранена в models/xgboost.json")
    
    # Feature Importance
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    importance.to_csv('models/xgboost_feature_importance.csv', index=False)
    plot_feature_importance(importance, 'XGBoost - Feature Importance', 'xgboost_importance.png')
    
    print("\nТоп-5 важных признаков:")
    for _, row in importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return model, metrics


if __name__ == '__main__':
    train_xgboost()