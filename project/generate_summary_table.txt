import pandas as pd
import os

def generate_summary_table(results, filename="results/cross_validation_summary.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Группировка по типу звена, уровню шума и параметру T
    grouped = {}

    for result in results:
        key = (result['тип_звена'], result['параметр'], result['уровень_шума'])
        if key not in grouped:
            grouped[key] = {
                'мин_1σ': [],
                'ср_1σ': [],
                'макс_1σ': [],
                'мин_2σ': [],
                'ср_2σ': [],
                'макс_2σ': [],
                'мин_3σ': [],
                'ср_3σ': [],
                'макс_3σ': []
            }

        grouped[key]['мин_1σ'].append(result['мин_1σ'])
        grouped[key]['ср_1σ'].append(result['ср_1σ'])
        grouped[key]['макс_1σ'].append(result['макс_1σ'])

        grouped[key]['мин_2σ'].append(result['мин_2σ'])
        grouped[key]['ср_2σ'].append(result['ср_2σ'])
        grouped[key]['макс_2σ'].append(result['макс_2σ'])

        grouped[key]['мин_3σ'].append(result['мин_3σ'])
        grouped[key]['ср_3σ'].append(result['ср_3σ'])
        grouped[key]['макс_3σ'].append(result['макс_3σ'])

    summary_data = []

    for key, values in grouped.items():
        system_name, T, noise_level = key

        summary_data.append({
            'тип_звена': system_name,
            'параметр': T,
            'уровень_шума': noise_level,
            'мин_1σ': round(min(values['мин_1σ']), 2),
            'ср_1σ': round(sum(values['ср_1σ']) / len(values['ср_1σ']), 2),
            'макс_1σ': round(max(values['макс_1σ']), 2),

            'мин_2σ': round(min(values['мин_2σ']), 2),
            'ср_2σ': round(sum(values['ср_2σ']) / len(values['ср_2σ']), 2),
            'макс_2σ': round(max(values['макс_2σ']), 2),

            'мин_3σ': round(min(values['мин_3σ']), 2),
            'ср_3σ': round(sum(values['ср_3σ']) / len(values['ср_3σ']), 2),
            'макс_3σ': round(max(values['макс_3σ']), 2)
        })

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"Сводная таблица обновлена: {filename}")