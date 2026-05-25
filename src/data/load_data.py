"""
Загрузка и объединение данных
"""

import pandas as pd


def load_and_merge_data():
    """Загрузка и объединение трёх файлов"""
    print("\n=== ЗАГРУЗКА ДАННЫХ ===")
    
    # Загрузка car_data.csv
    df1 = pd.read_csv('data/raw/car_data.csv')
    df1 = df1.drop_duplicates().reset_index(drop=True)
    df1.columns = df1.columns.str.lower()
    df1 = df1.rename(columns={
        'car_name': 'name', 'selling_price': 'price', 'present_price': 'present_price',
        'kms_driven': 'km_driven', 'fuel_type': 'fuel', 'seller_type': 'seller_type',
        'transmission': 'transmission', 'owner': 'owner'
    })
    df1['source'] = 'car_data'
    print(f"car_data.csv: {len(df1)} строк")
    
    # Загрузка CAR_DETAILS_FROM_CAR_DEKHO.csv
    df2 = pd.read_csv('data/raw/CAR_DETAILS_FROM_CAR_DEKHO.csv')
    df2 = df2.drop_duplicates().reset_index(drop=True)
    df2.columns = df2.columns.str.lower()
    df2 = df2.rename(columns={'selling_price': 'price', 'km_driven': 'km_driven'})
    df2['source'] = 'car_dekho'
    print(f"CAR_DETAILS_FROM_CAR_DEKHO.csv: {len(df2)} строк")
    
    # Загрузка Car_details_v3.csv
    df3 = pd.read_csv('data/raw/Car_details_v3.csv')
    df3 = df3.drop_duplicates().reset_index(drop=True)
    df3.columns = df3.columns.str.lower()
    df3 = df3.rename(columns={'selling_price': 'price', 'km_driven': 'km_driven'})
    df3['source'] = 'car_details_v3'
    print(f"Car_details_v3.csv: {len(df3)} строк")
    
    # Добавление возраста автомобиля
    current_year = 2026
    for df in [df1, df2, df3]:
        if 'year' in df.columns:
            df['car_age'] = current_year - df['year']
    
    # Объединение
    df = pd.concat([df1, df2, df3], ignore_index=True)
    print(f"\nОбъединённый датасет: {len(df)} строк")
    
    return df, df1, df2, df3