import numpy as np
import matplotlib.pyplot as plt
import os

GRAPHS_TO_SAVE = 1

def validate(data, corridor_data, save_path="results/validation"):
    os.makedirs(save_path, exist_ok=True)

    mean_values = corridor_data["mean"]
    std_values = corridor_data["std"]
    time_points = corridor_data["time"]

    one_sigma_upper = mean_values + std_values
    one_sigma_lower = mean_values - std_values
    two_sigma_upper = mean_values + 2 * std_values
    two_sigma_lower = mean_values - 2 * std_values
    three_sigma_upper = mean_values + 3 * std_values
    three_sigma_lower = mean_values - 3 * std_values

    percent_1sigma_list = []
    percent_2sigma_list = []
    percent_3sigma_list = []

    for idx, series in enumerate(data):
        series = np.array(series)
        length = min(len(series), len(mean_values))

        # Вычисление процента попадания в коридоры
        in_1sigma = ((series > one_sigma_lower[:length]) & (series < one_sigma_upper[:length])).sum() / length * 100
        in_2sigma = ((series > two_sigma_lower[:length]) & (series < two_sigma_upper[:length])).sum() / length * 100
        in_3sigma = ((series > three_sigma_lower[:length]) & (series < three_sigma_upper[:length])).sum() / length * 100

        percent_1sigma_list.append(in_1sigma)
        percent_2sigma_list.append(in_2sigma)
        percent_3sigma_list.append(in_3sigma)

        # Построение графиков только для нескольких симуляций
        if idx < GRAPHS_TO_SAVE:
            plt.figure(figsize=(10, 6))
            plt.plot(time_points[:length], series, label='Сигнал')
            plt.plot(time_points[:length], mean_values[:length], '--', color='red', label='Среднее')
            plt.fill_between(time_points[:length], one_sigma_lower[:length], one_sigma_upper[:length],
                             color='green', alpha=0.2, label='1σ')
            plt.fill_between(time_points[:length], two_sigma_lower[:length], two_sigma_upper[:length],
                             color='orange', alpha=0.2, label='2σ')
            plt.fill_between(time_points[:length], three_sigma_lower[:length], three_sigma_upper[:length],
                             color='yellow', alpha=0.2, label='3σ')

            plt.text(0.5, 0.5,
                     f"В 1σ: {in_1sigma:.1f}%\n"
                     f"В 2σ: {in_2sigma:.1f}%\n"
                     f"В 3σ: {in_3sigma:.1f}%",
                     ha='center', va='top', transform=plt.gca().transAxes, fontsize=10)

            plt.title(f"График валидации: {corridor_data['system_name']}")
            plt.xlabel("Время (с)")
            plt.ylabel("Отклик")
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4, fancybox=True, shadow=True)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"validation_{idx+1}.png"))
            plt.close()

    return {
        'system_name': corridor_data['system_name'],
        'noise_level': corridor_data['noise_level'],
        'мин_1σ': round(min(percent_1sigma_list), 2),
        'ср_1σ': round(sum(percent_1sigma_list) / len(percent_1sigma_list), 2),
        'макс_1σ': round(max(percent_1sigma_list), 2),
        'мин_2σ': round(min(percent_2sigma_list), 2),
        'ср_2σ': round(sum(percent_2sigma_list) / len(percent_2sigma_list), 2),
        'макс_2σ': round(max(percent_2sigma_list), 2),
        'мин_3σ': round(min(percent_3sigma_list), 2),
        'ср_3σ': round(sum(percent_3sigma_list) / len(percent_3sigma_list), 2),
        'макс_3σ': round(max(percent_3sigma_list), 2)
    }