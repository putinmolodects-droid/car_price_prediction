"""
Обработка брендов автомобилей
"""

import re
import pandas as pd


def extract_brand_from_name(name):
    """Извлечение бренда из полного названия"""
    if pd.isna(name):
        return 'unknown'
    name_clean = str(name).strip().lower()
    
    two_word_brands = ['land rover', 'range rover', 'mercedes benz', 'royal enfield']
    for brand in two_word_brands:
        if name_clean.startswith(brand):
            return brand.replace(' ', '_')
    
    first_word = name_clean.split()[0]
    return re.sub(r'[^\w\-]', '', first_word)


def extract_brand_from_model(model_name):
    """Определение бренда по названию модели (для car_data.csv)"""
    if pd.isna(model_name):
        return 'unknown'
    model_name = str(model_name).lower().strip()
    
    model_to_brand = {
        'swift': 'maruti', 'dzire': 'maruti', 'alto': 'maruti', 'wagon r': 'maruti',
        'ciaz': 'maruti', 'ritz': 'maruti', 'ertiga': 'maruti', 'baleno': 'maruti',
        'vitara brezza': 'maruti', 's-cross': 'maruti', 'ignis': 'maruti', 'celerio': 'maruti',
        'i20': 'hyundai', 'i10': 'hyundai', 'grand i10': 'hyundai', 'creta': 'hyundai',
        'verna': 'hyundai', 'xcent': 'hyundai', 'eon': 'hyundai', 'elantra': 'hyundai',
        'city': 'honda', 'amaze': 'honda', 'brio': 'honda', 'jazz': 'honda', 'wrv': 'honda',
        'fortuner': 'toyota', 'innova': 'toyota', 'corolla altis': 'toyota', 'etios': 'toyota',
        'figo': 'ford', 'ecosport': 'ford', 'endeavour': 'ford', 'fiesta': 'ford',
        'scorpio': 'mahindra', 'xuv500': 'mahindra', 'bolero': 'mahindra', 'thar': 'mahindra',
        'indica': 'tata', 'indigo': 'tata', 'nano': 'tata', 'tiago': 'tata', 'nexon': 'tata',
        'kwid': 'renault', 'duster': 'renault', 'lodgy': 'renault',
        'micra': 'nissan', 'sunny': 'nissan', 'terrano': 'nissan',
        'polo': 'volkswagen', 'vento': 'volkswagen'
    }
    
    if model_name in model_to_brand:
        return model_to_brand[model_name]
    
    for model, brand in model_to_brand.items():
        if model in model_name or model_name in model:
            return brand
    
    return 'unknown'


def categorize_brand(brand):
    """Смысловое группирование брендов"""
    if pd.isna(brand) or brand == 'unknown':
        return 'other'
    
    brand = str(brand).lower()
    
    premium = ['bmw', 'mercedes', 'audi', 'lexus', 'jaguar', 'land_rover', 'range_rover', 
               'volvo', 'porsche', 'ferrari', 'lamborghini', 'bentley', 'maserati', 'tesla']
    mass = ['maruti', 'hyundai', 'honda', 'toyota', 'ford', 'nissan', 'kia', 'volkswagen',
            'chevrolet', 'renault', 'skoda', 'mitsubishi', 'mazda', 'subaru', 'suzuki']
    budget = ['datsun', 'tata', 'mahindra', 'lada', 'chery', 'great_wall']
    sports = ['royal_enfield', 'ktm', 'yamaha', 'kawasaki', 'ducati', 'bajaj', 'hero']
    
    if brand in premium:
        return 'premium'
    elif brand in mass:
        return 'mass'
    elif brand in budget:
        return 'budget'
    elif brand in sports:
        return 'sports'
    return 'other'