from data_loader import load_data, generate_noisy_signals
from corridor_builder import build_corridors
from validator import validate
import os
import pandas as pd
from datetime import datetime

def save_summary_table(validation_results, filename="results/summary.csv"):
    """Сохраняет сводную таблицу в CSV формате с улучшенным форматированием"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Создаем DataFrame с результатами
    df = pd.DataFrame([{
        'Тип сигнала': res['pathology_type'],
        'Уровень шума (%)': validation_results['noise_level'],
        '1σ мин': res['мин_1σ'],
        '1σ сред': res['ср_1σ'],
        '1σ макс': res['макс_1σ'],
        '2σ мин': res['мин_2σ'],
        '2σ сред': res['ср_2σ'],
        '2σ макс': res['макс_2σ'],
        '3σ мин': res['мин_3σ'],
        '3σ сред': res['ср_3σ'],
        '3σ макс': res['макс_3σ']
    } for res in validation_results['results']])
    
    # Сохраняем в CSV с кодировкой UTF-8 и метаданными
    with open(filename, 'w', encoding='utf-8-sig') as f:
        df.to_csv(f, index=False, encoding='utf-8-sig')
    
    print(f"Сводная таблица сохранена: {filename}")

def main():
    print("=== Анализатор электроретинограмм ===")
    print("1. Загрузка данных...")
    time_points, reference, pathologies, pathology_names = load_data('_data.csv')
    
    # Добавляем название для эталонного сигнала
    signal_names = ['Эталон'] + pathology_names
    all_signals = [reference] + pathologies
    
    print("2. Генерация зашумленных сигналов...")
    noisy_signals = generate_noisy_signals(reference, num_signals=50)
    
    print("3. Построение коридоров...")
    corridor_data = build_corridors(noisy_signals)
    
    print("4. Валидация сигналов...")
    validation_results = validate(all_signals, corridor_data, signal_names)
    
    print("5. Сохранение результатов...")
    save_summary_table(validation_results)
    
    print("\nАнализ завершен успешно. Результаты сохранены в папке results")

if __name__ == "__main__":
    main()