import numpy as np
import matplotlib.pyplot as plt
import os

def build_corridors(signals, save_path="results/corridors"):
    """Строит сигма-коридоры на основе сигналов с улучшенной визуализацией"""
    os.makedirs(save_path, exist_ok=True)
    data_array = np.array(signals)
    
    # Расчет статистик
    mean_values = np.nanmean(data_array, axis=0)
    std_values = np.nanstd(data_array, axis=0)
    time_points = np.linspace(0, 2, len(mean_values))

    # Расчет теоретических процентов попадания в коридоры
    # Для нормального распределения:
    # 1σ ≈ 68.27%, 2σ ≈ 95.45%, 3σ ≈ 99.73%
    theory_1sigma = 68.27
    theory_2sigma = 95.45
    theory_3sigma = 99.73

    # Границы коридоров
    one_sigma = (mean_values - std_values, mean_values + std_values)
    two_sigma = (mean_values - 2*std_values, mean_values + 2*std_values)
    three_sigma = (mean_values - 3*std_values, mean_values + 3*std_values)

    # Визуализация с улучшениями
    plt.figure(figsize=(12, 7), dpi=300)
    plt.plot(time_points, mean_values, '--', color='red', linewidth=2, label='Среднее')
    
    # Заполнение областей с прозрачностью
    plt.fill_between(time_points, *three_sigma, color='yellow', alpha=0.1, 
                    label=f'3σ (теор. {theory_3sigma}%)')
    plt.fill_between(time_points, *two_sigma, color='orange', alpha=0.2, 
                    label=f'2σ (теор. {theory_2sigma}%)')
    plt.fill_between(time_points, *one_sigma, color='green', alpha=0.3, 
                    label=f'1σ (теор. {theory_1sigma}%)')

    # Настройки графика
    plt.title("Сигма-коридоры электроретинограммы", fontsize=14, pad=20)
    plt.xlabel("Время, с", fontsize=12)
    plt.ylabel("Амплитуда, мкВ", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Легенда
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        fontsize=10,
        framealpha=1.0
    )
    
    plt.tight_layout()
    
    # Сохранение с высоким качеством
    plt.savefig(
        os.path.join(save_path, "sigma_corridors.png"),
        dpi=300,
        bbox_inches='tight',
        metadata={
            'Title': 'Сигма-коридоры электроретинограммы',
            'Author': 'ERG Analysis System',
            'Description': 'График сигма-коридоров для анализа ЭРГ'
        }
    )
    plt.close()

    return {
        "mean": mean_values,
        "std": std_values,
        "time": time_points,
        "noise_level": 10
    }