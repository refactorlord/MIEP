import numpy as np
import pandas as pd
import os

def load_data(filename='data.csv'):
    """Загружает данные без интерполяции"""
    # Чтение данных с учетом заголовков
    data = pd.read_csv(filename, header=0)
    time_col = data.columns[0]
    
    # Разделение на эталон и патологии
    reference = data.iloc[:, 1].values
    pathologies = [data.iloc[:, i].values for i in range(2, data.shape[1])]
    pathology_names = data.columns[2:].tolist()
    
    # Получаем временные точки из первого столбца
    time_points = data[time_col].values
    
    # Сохраняем исходные данные
    os.makedirs('results', exist_ok=True)
    
    return time_points, reference, pathologies, pathology_names

def generate_noisy_signals(reference, num_signals=50, noise_level=0.1):
    """Генерирует зашумленные версии эталонного сигнала"""
    valid_values = reference[~np.isnan(reference)]
    if len(valid_values) == 0:
        return [np.zeros_like(reference) for _ in range(num_signals)]
    
    signal_std = np.std(valid_values)
    noisy_signals = []
    
    for _ in range(num_signals):
        noise = np.random.normal(0, noise_level * signal_std, size=len(reference))
        noisy_signal = reference + noise
        noisy_signals.append(noisy_signal)
    
    return noisy_signals