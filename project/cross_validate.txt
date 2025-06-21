import numpy as np
import matplotlib.pyplot as plt
import os
from simulation import simulate_system

def cross_validate(data, corridor_data, test_T_values, save_path="results/cross_validation"):
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

    results = []

    for T in test_T_values:
        print(f"Кросс-валидация: T={T:.2f}")
        # Моделируем 50 сигналов для статистики
        params = None
        if corridor_data['system_type'] == 'aperiodic':
            params = {'T': T}
        elif corridor_data['system_type'] == 'oscillatory':
            params = {'T': T, 'xi': 0.5}
        elif corridor_data['system_type'] == 'integrating':
            params = {'k': round(T, 2)}

        test_simulations = simulate_system(num_simulations=20, system_type=corridor_data['system_type'], params=params,
                                           noise_level=corridor_data['noise_level'] / 100)

        percent_1sigma_list = []
        percent_2sigma_list = []
        percent_3sigma_list = []

        for idx, series in enumerate(test_simulations):
            series = np.array(series)
            length = min(len(series), len(mean_values))

            in_1sigma = ((series > one_sigma_lower[:length]) & (series < one_sigma_upper[:length])).sum() / length * 100
            in_2sigma = ((series > two_sigma_lower[:length]) & (series < two_sigma_upper[:length])).sum() / length * 100
            in_3sigma = ((series > three_sigma_lower[:length]) & (series < three_sigma_upper[:length])).sum() / length * 100

            percent_1sigma_list.append(in_1sigma)
            percent_2sigma_list.append(in_2sigma)
            percent_3sigma_list.append(in_3sigma)

            # Построение графика только для первой симуляции
            if idx == 0:
                plt.figure(figsize=(10, 6))
                plt.plot(time_points[:length], series, label='Сигнал')
                plt.plot(time_points[:length], mean_values[:length], '--', color='red', label='Среднее')
                plt.fill_between(time_points[:length], one_sigma_lower[:length], one_sigma_upper[:length],
                                 color='green', alpha=0.2, label='1σ')
                plt.fill_between(time_points[:length], two_sigma_lower[:length], two_sigma_upper[:length],
                                 color='orange', alpha=0.2, label='2σ')
                plt.fill_between(time_points[:length], three_sigma_lower[:length], three_sigma_upper[:length],
                                 color='yellow', alpha=0.2, label='3σ')

                avg_1sigma = sum(percent_1sigma_list) / len(percent_1sigma_list)
                avg_2sigma = sum(percent_2sigma_list) / len(percent_2sigma_list)
                avg_3sigma = sum(percent_3sigma_list) / len(percent_3sigma_list)

                plt.text(0.5, 0.5,
                         f"В 1σ: {avg_1sigma:.1f}%\n"
                         f"В 2σ: {avg_2sigma:.1f}%\n"
                         f"В 3σ: {avg_3sigma:.1f}%",
                         ha='center', va='top', transform=plt.gca().transAxes, fontsize=10)

                plt.title(f"Кросс-валидация: {corridor_data['system_name']} → T={T}")
                plt.xlabel("Время (с)")
                plt.ylabel("Отклик")
                plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4, fancybox=True, shadow=True)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, f"T_{T}_{idx+1}.png"))
                plt.close()

        result = {
            'system_name': corridor_data['system_name'],
            'system_type': corridor_data['system_type'],
            'T': T,
            'noise_level': corridor_data['noise_level'],
            'min_1sigma': min(percent_1sigma_list),
            'avg_1sigma': sum(percent_1sigma_list) / len(percent_1sigma_list),
            'max_1sigma': max(percent_1sigma_list),
            'min_2sigma': min(percent_2sigma_list),
            'avg_2sigma': sum(percent_2sigma_list) / len(percent_2sigma_list),
            'max_2sigma': max(percent_2sigma_list),
            'min_3sigma': min(percent_3sigma_list),
            'avg_3sigma': sum(percent_3sigma_list) / len(percent_3sigma_list),
            'max_3sigma': max(percent_3sigma_list)
        }

        results.append(result)

    return results