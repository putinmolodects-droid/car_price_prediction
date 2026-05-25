"""
Главный скрипт обработки данных для DVC пайплайна
Запуск: python src/data/make_dataset.py
"""

import os
import sys
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.data.load_data import load_and_merge_data
from src.data.preprocess import preprocess_data
from src.utils.helpers import apply_transform


def prepare_for_models(df, numeric_cols, cat_cols, target='price', transform_type='sqrt'):
    """
    Подготовка данных для обучения моделей с разделением на 3 выборки
    """
    print("\n=== ПОДГОТОВКА ДАННЫХ ДЛЯ МОДЕЛЕЙ ===")
    
    X = df[numeric_cols + cat_cols]
    y = apply_transform(df[target], transform_type)
    
    # Разделение на train/val/test (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"Train: {X_train.shape[0]} строк ({X_train.shape[0]/len(X)*100:.0f}%)")
    print(f"Validation: {X_val.shape[0]} строк ({X_val.shape[0]/len(X)*100:.0f}%)")
    print(f"Test: {X_test.shape[0]} строк ({X_test.shape[0]/len(X)*100:.0f}%)")
    
    # Препроцессор
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])
    
    # Преобразование
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Сохраняем raw для CatBoost
    X_train_raw = X_train.copy()
    X_val_raw = X_val.copy()
    X_test_raw = X_test.copy()
    
    # Имена признаков
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
    feature_names = numeric_cols + list(cat_feature_names)
    
    return (X_train_processed, X_val_processed, X_test_processed,
            y_train, y_val, y_test,
            X_train_raw, X_val_raw, X_test_raw,
            preprocessor, feature_names)


def save_processed_data(data, preprocessor, feature_names):
    """Сохранение обработанных данных"""
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    (X_train, X_val, X_test, y_train, y_val, y_test,
     X_train_raw, X_val_raw, X_test_raw, preprocessor, feature_names) = data
    
    # Сохранение numpy массивов
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/X_val.npy', X_val)
    np.save('data/processed/X_test.npy', X_test)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_val.npy', y_val)
    np.save('data/processed/y_test.npy', y_test)
    
    # Сохранение raw данных для CatBoost
    X_train_raw.to_csv('data/processed/X_train_raw.csv', index=False)
    X_val_raw.to_csv('data/processed/X_val_raw.csv', index=False)
    X_test_raw.to_csv('data/processed/X_test_raw.csv', index=False)
    
    # Сохранение препроцессора и имён признаков
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    with open('models/feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_names))
    
    print("\nДанные сохранены в data/processed/")


def main():
    # Загрузка параметров
    with open('config/params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Загрузка и предобработка
    df, _, _, _ = load_and_merge_data()
    df, numeric_cols, cat_cols = preprocess_data(df)
    
    # Подготовка данных для моделей
    data = prepare_for_models(
        df, numeric_cols, cat_cols,
        transform_type=params['data']['transform_type']
    )
    
    # Сохранение
    save_processed_data(data, data[9], data[10])  # preprocessor и feature_names
    
    print("\nОбработка данных завершена!")


if __name__ == '__main__':
    main()