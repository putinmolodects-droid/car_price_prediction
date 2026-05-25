"""
Предобработка данных
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from src.utils.helpers import extract_number, extract_max_torque
from src.features.brand_processor import extract_brand_from_model, extract_brand_from_name, categorize_brand


def preprocess_data(df):
    """Полная предобработка данных"""
    print("\n=== ПРЕДОБРАБОТКА ДАННЫХ ===")
    
    # 1. Удаление выбросов по цене
    q_low = df['price'].quantile(0.01)
    q_high = df['price'].quantile(0.99)
    df = df[(df['price'] >= q_low) & (df['price'] <= q_high)]
    print(f"После удаления выбросов по цене: {len(df)} строк")
    
    # 2. Удаление выбросов по пробегу
    q_km = df['km_driven'].quantile(0.99)
    df = df[df['km_driven'] <= q_km]
    print(f"После удаления выбросов по пробегу: {len(df)} строк")
    
    # 3. Извлечение числовых значений из строковых признаков
    if 'mileage' in df.columns:
        df['mileage_num'] = df['mileage'].apply(extract_number)
    if 'engine' in df.columns:
        df['engine_num'] = df['engine'].apply(extract_number)
    if 'max_power' in df.columns:
        df['max_power_num'] = df['max_power'].apply(extract_number)
    if 'torque' in df.columns:
        df['torque_num'] = df['torque'].apply(extract_max_torque)
    
    # 4. Обработка owner
    df['owner'] = df['owner'].astype(str)
    owner_mapping = {
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
        'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3,
        'Fourth & Above Owner': 4, 'Test Drive Car': 0
    }
    df['owner_encoded'] = df['owner'].map(owner_mapping).fillna(0).astype(int)
    
    # 5. Обработка брендов
    df['brand'] = None
    
    mask_car_data = df['source'] == 'car_data'
    if mask_car_data.any():
        df.loc[mask_car_data, 'brand'] = df.loc[mask_car_data, 'name'].apply(extract_brand_from_model)
    
    mask_other = ~mask_car_data & df['name'].notna()
    if mask_other.any():
        df.loc[mask_other, 'brand'] = df.loc[mask_other, 'name'].apply(extract_brand_from_name)
    
    df['brand'] = df['brand'].fillna('unknown')
    df['brand_category'] = df['brand'].apply(categorize_brand)
    
    # 6. Удаление ненужных колонок
    drop_cols = ['name', 'year', 'owner', 'mileage', 'engine', 'max_power', 'torque', 
                 'present_price', 'source']
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    # 7. Заполнение пропусков
    numeric_cols = ['km_driven', 'car_age', 'owner_encoded', 'mileage_num', 
                    'engine_num', 'max_power_num', 'torque_num']
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    if len(numeric_cols) > 1 and df[numeric_cols].dropna().shape[0] > 0:
        knn_imputer = KNNImputer(n_neighbors=5)
        df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])
    else:
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
    
    cat_cols = ['fuel', 'seller_type', 'transmission', 'brand_category']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('missing')
    
    # 8. Переиндексация
    df = df.reset_index(drop=True)
    
    print(f"После предобработки: {len(df)} строк, {len(df.columns)} колонок")
    
    return df, numeric_cols, cat_cols