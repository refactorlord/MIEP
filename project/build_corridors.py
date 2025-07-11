import numpy as np
import matplotlib.pyplot as plt
import os

def build_corridors(data, save_path="results/corridors", system_info=None):
    os.makedirs(save_path, exist_ok=True)
    data_array = np.array(data)
    mean_values = np.nanmean(data_array, axis=0)
    std_values = np.nanstd(data_array, axis=0)
    time_points = np.linspace(0, 10, len(mean_values))

    one_sigma_upper = mean_values + std_values
    one_sigma_lower = mean_values - std_values
    two_sigma_upper = mean_values + 2 * std_values
    two_sigma_lower = mean_values - 2 * std_values
    three_sigma_upper = mean_values + 3 * std_values
    three_sigma_lower = mean_values - 3 * std_values

    plt.figure(figsize=(10, 6))
    plt.plot(time_points, mean_values, '--', color='red', label='Среднее')
    plt.fill_between(time_points, one_sigma_lower, one_sigma_upper, color='green', alpha=0.2, label='1σ')
    plt.fill_between(time_points, two_sigma_lower, two_sigma_upper, color='orange', alpha=0.2, label='2σ')
    plt.fill_between(time_points, three_sigma_lower, three_sigma_upper, color='yellow', alpha=0.2, label='3σ')

    plt.title(f"Сигма-коридоры ({system_info['name']})")
    plt.xlabel("Время (с)")
    plt.ylabel("Отклик")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"sigma_corridors_{system_info['type']}.png"), dpi=150)
    plt.close()

    corridor_data = {
        "mean": mean_values,
        "std": std_values,
        "time": time_points,
        "system_type": system_info['type'],
        "system_name": system_info['name'],
        "noise_level": int(save_path.split('_')[-1])
    }

    return corridor_data