from simulation import simulate_system
from build_corridors import build_corridors
from validate_in_corridors import validate
from cross_validate import cross_validate
import os
import numpy as np
import pandas as pd
from generate_summary_table import generate_summary_table

SYSTEM_PAIRS = [
    {
        'type': 'aperiodic',
        'name': 'Апериодическое',
        'norm_params': {'T': 1},
    },
    {
        'type': 'oscillatory',
        'name': 'Колебательное',
        'norm_params': {'T': 1, 'xi': 0.5},
    },
    {
        'type': 'integrating',
        'name': 'Интегрирующее',
        'norm_params': {'k': 1},
    }
]

# Ввод параметров пользователем
base_T = float(input("Введите базовое значение T: "))
start_T = float(input("Начальное значение T: "))
end_T = float(input("Конечное значение T: "))
step_T = float(input("Шаг изменения T: "))

noise_levels_input = input("Уровни шума через пробел (например, 10 20 30): ")
NOISE_LEVELS = list(map(int, noise_levels_input.split()))

# Формирование диапазона T
T_VALUES = []
t = start_T
while t <= end_T + 1e-6:
    T_VALUES.append(round(t, 2))
    t += step_T

all_results = []

for system_info in SYSTEM_PAIRS:
    system_type = system_info['type']
    system_name = system_info['name']
    base_dir = os.path.join("results", system_type)
    os.makedirs(base_dir, exist_ok=True)

    for noise_level in NOISE_LEVELS:
        print(f"\n{'='*60}\nОбработка: {system_name}, уровень шума={noise_level}%\n{'='*60}")

        # Моделирование 100 симуляций
        simulations = simulate_system(
            num_simulations=100,
            system_type=system_type,
            params=system_info['norm_params'],
            noise_level=noise_level / 100
        )

        # Построение сигма-коридоров на основе первых 50
        corridor_path = os.path.join(base_dir, f"noise_{noise_level}")
        corridor_data = build_corridors(simulations[:50], save_path=corridor_path, system_info=system_info)

        # Валидация на основе оставшихся 50
        val_result = validate(simulations[50:], corridor_data, save_path=os.path.join(corridor_path, "validation"))

        # Добавляем в all_results данные валидации (T = 1)
        all_results.append({
            'тип_звена': system_name,
            'параметр': base_T,
            'уровень_шума': noise_level,
            'мин_1σ': round(val_result['мин_1σ'], 2),
            'ср_1σ': round(val_result['ср_1σ'], 2),
            'макс_1σ': round(val_result['макс_1σ'], 2),
            'мин_2σ': round(val_result['мин_2σ'], 2),
            'ср_2σ': round(val_result['ср_2σ'], 2),
            'макс_2σ': round(val_result['макс_2σ'], 2),
            'мин_3σ': round(val_result['мин_3σ'], 2),
            'ср_3σ': round(val_result['ср_3σ'], 2),
            'макс_3σ': round(val_result['макс_3σ'], 2)
        })

        # Кросс-валидация на основе тех же 50 тестовых данных
        cv_results = cross_validate(simulations[50:], corridor_data, T_VALUES, save_path=os.path.join(corridor_path, "cross_validation"))

        # Добавление всех кросс-валидационных результатов
        all_results.extend([
            {
                'тип_звена': res['system_name'],
                'параметр': res['T'],
                'уровень_шума': res['noise_level'],
                'мин_1σ': round(res['min_1sigma'], 2),
                'ср_1σ': round(res['avg_1sigma'], 2),
                'макс_1σ': round(res['max_1sigma'], 2),
                'мин_2σ': round(res['min_2sigma'], 2),
                'ср_2σ': round(res['avg_2sigma'], 2),
                'макс_2σ': round(res['max_2sigma'], 2),
                'мин_3σ': round(res['min_3sigma'], 2),
                'ср_3σ': round(res['avg_3sigma'], 2),
                'макс_3σ': round(res['max_3sigma'], 2)
            } for res in cv_results
        ])

df_summary = pd.DataFrame(all_results)
generate_summary_table(all_results, filename="results/cross_validation_summary.csv")