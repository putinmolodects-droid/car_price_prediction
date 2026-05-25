"""
Обучение нейронной сети (MLP)
Запуск: python src/models/train/train_nn.py

Создаёт:
1. Граф модели (архитектура) - model.png
2. Кривые обучения (Loss и MAE) - learning_curves.png
3. Гистограммы весов слоёв - weights_histograms.png
4. TensorBoard логи
"""

import os
import sys
import numpy as np
import yaml
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.utils import plot_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.utils.helpers import inverse_transform, calculate_metrics, save_metrics


def create_model(input_dim, dropout_rate=0.3, learning_rate=0.001):
    """
    Создание модели нейронной сети
    """
    model = keras.Sequential([
        keras.Input(shape=(input_dim,), name='input_layer'),
        layers.Dense(128, activation='relu', name='hidden_1'),
        layers.Dropout(dropout_rate, name='dropout_1'),
        layers.Dense(64, activation='relu', name='hidden_2'),
        layers.Dropout(dropout_rate - 0.1, name='dropout_2'),
        layers.Dense(32, activation='relu', name='hidden_3'),
        layers.Dense(1, name='output_layer')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model


def plot_model_architecture(model, filename='nn_architecture.png', show_shapes=True, show_layer_names=True):
    """
    Визуализация графа модели нейронной сети
    
    Параметры:
    - model: модель Keras
    - filename: имя выходного файла
    - show_shapes: показывать размерности (None, 128) и т.д.
    - show_layer_names: показывать имена слоёв
    """
    os.makedirs('reports/figures', exist_ok=True)
    
    # Полный путь к файлу
    filepath = f'reports/figures/{filename}'
    
    # Сохраняем изображение архитектуры модели
    plot_model(
        model, 
        to_file=filepath,
        show_shapes=show_shapes,
        show_layer_names=show_layer_names,
        expand_nested=True,
        dpi=96
    )
    
    print(f"Граф модели сохранён в {filepath}")
    
    return filepath


def plot_learning_curves(history, filename='nn_learning_curves.png'):
    """
    Визуализация кривых обучения (Loss и MAE)
    """
    os.makedirs('reports/figures', exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # График Loss (MSE)
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2, color='blue')
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='red')
    axes[0].set_title('Кривые обучения (MSE)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Эпоха')
    axes[0].set_ylabel('MSE')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # График MAE
    axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2, color='blue')
    axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2, color='red')
    axes[1].set_title('Кривые обучения (MAE)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Эпоха')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'reports/figures/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Кривые обучения сохранены в reports/figures/{filename}")


def plot_weights_histograms(model, filename='nn_weights_histograms.png'):
    """
    Визуализация гистограмм весов для каждого слоя
    """
    os.makedirs('reports/figures', exist_ok=True)
    
    # Собираем слои, у которых есть веса
    weight_layers = []
    layer_names = []
    
    for i, layer in enumerate(model.layers):
        if len(layer.get_weights()) > 0:
            weight_layers.append(layer)
            layer_names.append(f"{i+1}. {layer.name}")
    
    if not weight_layers:
        print("Нет слоёв с весами для визуализации")
        return
    
    # Определяем количество подграфиков
    n_layers = len(weight_layers)
    cols = min(3, n_layers)
    rows = (n_layers + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_layers == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, (layer, name) in enumerate(zip(weight_layers, layer_names)):
        weights = layer.get_weights()[0].flatten()  # веса
        biases = layer.get_weights()[1].flatten() if len(layer.get_weights()) > 1 else None
        
        ax = axes[i]
        ax.hist(weights, bins=50, alpha=0.7, color='steelblue', edgecolor='black', label='Weights')
        if biases is not None:
            ax.hist(biases, bins=30, alpha=0.5, color='coral', edgecolor='black', label='Biases')
        
        ax.set_title(f'Слой: {name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Значение')
        ax.set_ylabel('Частота')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Добавляем статистику
        stats_text = f"μ={weights.mean():.3f}, σ={weights.std():.3f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Скрываем пустые подграфики
    for i in range(n_layers, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Гистограммы весов и смещений слоёв нейронной сети', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'reports/figures/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Гистограммы весов сохранены в reports/figures/{filename}")


def train_neural_network():
    """Обучение нейронной сети со всеми визуализациями"""
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ (MLP)")
    print("=" * 60)
    
    # Загрузка параметров
    with open('config/params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Загрузка данных
    X_train = np.load('data/processed/X_train.npy')
    X_val = np.load('data/processed/X_val.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_val = np.load('data/processed/y_val.npy')
    
    print(f"\nРазмер данных:")
    print(f"  Train: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    
    # Создание модели
    model = create_model(
        input_dim=X_train.shape[1],
        dropout_rate=0.3,
        learning_rate=0.001
    )
    
        
    os.makedirs('logs', exist_ok=True)
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=20, 
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=10,
        verbose=1
    )
    tensorboard = callbacks.TensorBoard(
        log_dir='logs/nn', 
        histogram_freq=0,
        write_graph=True,
        write_images=True,
        update_freq="epoch",
        profile_batch=0
    )
    
    print("\nНачало обучения нейронной сети...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=params['models']['neural_network']['epochs'],
        batch_size=params['models']['neural_network']['batch_size'],
        callbacks=[early_stop, reduce_lr, tensorboard],
        verbose=1
    )
    ###
    # with tf.summary.create_file_writer('logs').as_default():
    #     tf.summary.trace_export(
    #         name="DNN_graph_trace",
    #         step=0,
    #         profiler_outdir='logs'
    #     )
    
    # Предсказания на обучающей выборке
    transform_type = params['data']['transform_type']
    y_train_pred = inverse_transform(model.predict(X_train).flatten(), transform_type)
    y_train_true = inverse_transform(y_train, transform_type)
    
    # Предсказания на валидационной выборке
    y_val_pred = inverse_transform(model.predict(X_val).flatten(), transform_type)
    y_val_true = inverse_transform(y_val, transform_type)
    
    # Метрики
    train_metrics = calculate_metrics(y_train_true, y_train_pred)
    val_metrics = calculate_metrics(y_val_true, y_val_pred)
    
    print(f"\nРезультаты:")
    print(f"  Train: MAE={train_metrics['mae']:.2f}, R2={train_metrics['r2']:.4f}")
    print(f"  Validation: MAE={val_metrics['mae']:.2f}, R2={val_metrics['r2']:.4f}")
    
    # Сохранение метрик
    save_metrics(train_metrics, 'neural_network', stage='train')
    save_metrics(val_metrics, 'neural_network', stage='val')
    
    
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    for idx, layer in enumerate(dense_layers[:-1]):
        weights = layer.kernel.numpy().flatten()
        plt.figure(figsize=(12, 6))
        plt.hist(weights, bins=100)
        plt.title(f"Dense layer {idx+1} Weight Distribution (final epoch)")
        plt.xlabel("Weight value")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(f'reports/figures/', f"DenseNeuralNetwork_weights_layer_{idx+1}.png"))
        plt.close()
        
    plot_model_architecture(model, filename='nn_architecture.png', show_shapes=True, show_layer_names=True)
    
    plot_learning_curves(history, 'nn_learning_curves.png')
    
    plot_weights_histograms(model, 'nn_weights_histograms.png')
    
    # Сохранение модели
    os.makedirs('models', exist_ok=True)
    model.save('models/neural_network.keras')
    print(f"\nМодель сохранена в models/neural_network.keras")
    
    print("\n" + "=" * 60)
    print("TENSORBOARD")
    print("=" * 60)
    print("Для просмотра логов TensorBoard выполните команду:")
    print("  tensorboard --logdir=logs")
    print("\nВ браузере откроется http://localhost:6006")
    print("Там вы увидите:")
    print("  - Граф модели (Graphs)")
    print("  - Гистограммы весов (Histograms)")
    print("  - Кривые обучения (Scalars)")
    print("  - Распределения активаций (Distributions)")
    
    return model, history, train_metrics, val_metrics


if __name__ == '__main__':
    train_neural_network()