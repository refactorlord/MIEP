import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def validate(signals, corridor_data, signal_names, save_path="results/validation"):
    """Проверяет сигналы на соответствие коридорам с обработкой крайних случаев"""
    os.makedirs(save_path, exist_ok=True)
    
    mean = corridor_data["mean"]
    std = corridor_data["std"]
    time_points = corridor_data["time"]

    # Границы коридоров с защитой от NaN
    one_sigma_low = np.nan_to_num(mean - std, nan=0)
    one_sigma_high = np.nan_to_num(mean + std, nan=0)
    two_sigma_low = np.nan_to_num(mean - 2*std, nan=0)
    two_sigma_high = np.nan_to_num(mean + 2*std, nan=0)
    three_sigma_low = np.nan_to_num(mean - 3*std, nan=0)
    three_sigma_high = np.nan_to_num(mean + 3*std, nan=0)

    results = []
    
    for idx, (signal, name) in tqdm(enumerate(zip(signals, signal_names)), total=len(signals)):
        signal = np.nan_to_num(np.array(signal), nan=0)
        length = min(len(signal), len(mean))

        # Расчет попадания в коридоры с защитой от деления на 0
        in_1sigma = np.logical_and(
            signal[:length] > one_sigma_low[:length],
            signal[:length] < one_sigma_high[:length]
        )
        in_2sigma = np.logical_and(
            signal[:length] > two_sigma_low[:length],
            signal[:length] < two_sigma_high[:length]
        )
        in_3sigma = np.logical_and(
            signal[:length] > three_sigma_low[:length],
            signal[:length] < three_sigma_high[:length]
        )

        # Разбиваем сигнал на 10 сегментов для анализа
        segments = 10
        segment_len = max(1, length // segments)  # Защита от нулевой длины
        
        segment_stats = {'1sigma': [], '2sigma': [], '3sigma': []}

        for i in range(segments):
            start = i * segment_len
            end = min((i + 1) * segment_len, length)  # Защита от выхода за границы
            
            # Расчет процента попадания с обработкой пустых сегментов
            seg_len = end - start
            if seg_len == 0:
                segment_stats['1sigma'].append(0)
                segment_stats['2sigma'].append(0)
                segment_stats['3sigma'].append(0)
                continue
                
            seg_1sigma = 100 * np.sum(in_1sigma[start:end]) / seg_len
            seg_2sigma = 100 * np.sum(in_2sigma[start:end]) / seg_len
            seg_3sigma = 100 * np.sum(in_3sigma[start:end]) / seg_len
            
            segment_stats['1sigma'].append(seg_1sigma)
            segment_stats['2sigma'].append(seg_2sigma)
            segment_stats['3sigma'].append(seg_3sigma)

        # Расчет средних значений
        avg_1sigma = round(np.nanmean(segment_stats['1sigma']), 2)
        avg_2sigma = round(np.nanmean(segment_stats['2sigma']), 2)
        avg_3sigma = round(np.nanmean(segment_stats['3sigma']), 2)

        # Сохранение результатов
        results.append({
            'signal_num': idx + 1,
            'pathology_type': name,
            'мин_1σ': round(np.nanmin(segment_stats['1sigma']), 2),
            'ср_1σ': avg_1sigma,
            'макс_1σ': round(np.nanmax(segment_stats['1sigma']), 2),
            'мин_2σ': round(np.nanmin(segment_stats['2sigma']), 2),
            'ср_2σ': avg_2sigma,
            'макс_2σ': round(np.nanmax(segment_stats['2sigma']), 2),
            'мин_3σ': round(np.nanmin(segment_stats['3sigma']), 2),
            'ср_3σ': avg_3sigma,
            'макс_3σ': round(np.nanmax(segment_stats['3sigma']), 2)
        })

        plt.figure(figsize=(12, 7), dpi=300)
        plt.plot(time_points[:length], signal[:length], label=f'Сигнал: {name}', linewidth=2)
        plt.plot(time_points[:length], mean[:length], '--', color='red', linewidth=1.5, label='Среднее')
        
        plt.fill_between(time_points[:length], three_sigma_low[:length], three_sigma_high[:length],
                        color='yellow', alpha=0.1, label=f'3σ ({avg_3sigma}%)')
        plt.fill_between(time_points[:length], two_sigma_low[:length], two_sigma_high[:length],
                        color='orange', alpha=0.2, label=f'2σ ({avg_2sigma}%)')
        plt.fill_between(time_points[:length], one_sigma_low[:length], one_sigma_high[:length],
                        color='green', alpha=0.3, label=f'1σ ({avg_1sigma}%)')

        plt.title(f"Валидация: {name}", fontsize=14)
        plt.xlabel("Время, с", fontsize=12)
        plt.ylabel("Амплитуда, мкВ", fontsize=12)
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(save_path, f"validation_{idx+1}.png"), bbox_inches='tight')
        plt.close()

    return {
        'noise_level': corridor_data.get('noise_level', 10),
        'results': results
    }