import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === ПАРАМЕТРЫ ===
NUM_SIMULATIONS = 50     # количество симуляций для построения коридоров
NOISE_LEVELS = [10]       # уровни шума (%) - можно расширить при необходимости
CSV_FILENAME = "data.csv"
RESULTS_DIR = "results"

# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===

def load_data(filename):
    """
    Загружает данные из CSV.
    Первый столбец — временные точки, остальные — сигналы.
    """
    data = pd.read_csv(filename, header=None)
    time_points = data.iloc[:, 0].values
    signals = [data.iloc[:, i].values for i in range(1, data.shape[1])]
    return time_points, signals


def add_white_noise(signal, noise_level=(NOISE_LEVELS[0] / 100)):
    """
    Добавляет нормальный белый шум к сигналу.
    noise_level: процент от стандартного отклонения сигнала
    """
    signal_std = np.std(signal)
    noise = np.random.normal(0, noise_level * signal_std, size=len(signal))
    return signal + noise


def build_corridors(simulations):
    """
    Строит сигма-коридоры на основе массива симуляций.
    """
    simulations_array = np.array(simulations)
    mean_signal = np.mean(simulations_array, axis=0)
    std_signal = np.std(simulations_array, axis=0)

    one_sigma_upper = mean_signal + std_signal
    one_sigma_lower = mean_signal - std_signal
    two_sigma_upper = mean_signal + 2 * std_signal
    two_sigma_lower = mean_signal - 2 * std_signal
    three_sigma_upper = mean_signal + 3 * std_signal
    three_sigma_lower = mean_signal - 3 * std_signal

    return {
        "mean": mean_signal,
        "std": std_signal,
        "one_sigma_upper": one_sigma_upper,
        "one_sigma_lower": one_sigma_lower,
        "two_sigma_upper": two_sigma_upper,
        "two_sigma_lower": two_sigma_lower,
        "three_sigma_upper": three_sigma_upper,
        "three_sigma_lower": three_sigma_lower
    }


def validate(signal, corridor_data):
    """
    Проверяет, сколько точек попадает в сигма-коридоры
    """
    length = min(len(signal), len(corridor_data["mean"]))

    in_1sigma = ((signal > corridor_data["one_sigma_lower"][:length]) &
                 (signal < corridor_data["one_sigma_upper"][:length])).sum() / length * 100
    in_2sigma = ((signal > corridor_data["two_sigma_lower"][:length]) &
                 (signal < corridor_data["two_sigma_upper"][:length])).sum() / length * 100
    in_3sigma = ((signal > corridor_data["three_sigma_lower"][:length]) &
                 (signal < corridor_data["three_sigma_upper"][:length])).sum() / length * 100

    return {
        "avg_1sigma": round(in_1sigma, 2),
        "avg_2sigma": round(in_2sigma, 2),
        "avg_3sigma": round(in_3sigma, 2)
    }


def plot_validation(time_points, test_signal, corridor_data, filename):
    """
    Сохраняет график тестового сигнала с коридорами
    """
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, test_signal, label='Тестовый сигнал')
    plt.plot(time_points, corridor_data["mean"], '--', color='red', label='Среднее значение')
    plt.fill_between(time_points, corridor_data["one_sigma_lower"], corridor_data["one_sigma_upper"],
                     color='green', alpha=0.2, label='±1σ')
    plt.fill_between(time_points, corridor_data["two_sigma_lower"], corridor_data["two_sigma_upper"],
                     color='orange', alpha=0.2, label='±2σ')
    plt.fill_between(time_points, corridor_data["three_sigma_lower"], corridor_data["three_sigma_upper"],
                     color='yellow', alpha=0.2, label='±3σ')

    val_result = validate(test_signal, corridor_data)
    text = "\n".join([
        f"В 1σ: {val_result['avg_1sigma']}%",
        f"В 2σ: {val_result['avg_2sigma']}%",
        f"В 3σ: {val_result['avg_3sigma']}%"
    ])

    plt.text(0.5, 0.95, text, ha='center', va='top', transform=plt.gca().transAxes, fontsize=10)
    plt.title('Сигнал и сигма-коридоры')
    plt.xlabel("Время (с)")
    plt.ylabel("Отклик")
    plt.grid(True)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# === ОСНОВНОЙ ЦИКЛ ===
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "corridors"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "validation"), exist_ok=True)

    # --- ШАГ 1: Чтение данных ---
    time_points, signals = load_data(CSV_FILENAME)
    print(f"✅ Прочитано {len(signals)} сигналов")

    # --- ШАГ 2: Построение коридоров для первого сигнала ---
    first_signal = signals[0]
    print("\n🧪 Построение сигма-коридоров: сигнал 1")

    all_simulations = []
    for _ in range(NUM_SIMULATIONS):
        noisy = add_white_noise(first_signal, NOISE_LEVELS[0] / 100)
        all_simulations.append(noisy)

    corridor_data = build_corridors(all_simulations)
    corridor_data["time"] = time_points
    corridor_data["system_name"] = "Signal_1"
    corridor_data["noise_level"] = NOISE_LEVELS[0]

    # Сохраняем график коридоров
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, corridor_data["mean"], '--', color='red', label='Среднее')
    plt.fill_between(time_points, corridor_data["one_sigma_lower"], corridor_data["one_sigma_upper"],
                     color='green', alpha=0.2, label='±1σ')
    plt.fill_between(time_points, corridor_data["two_sigma_lower"], corridor_data["two_sigma_upper"],
                     color='orange', alpha=0.2, label='±2σ')
    plt.fill_between(time_points, corridor_data["three_sigma_lower"], corridor_data["three_sigma_upper"],
                     color='yellow', alpha=0.2, label='±3σ')
    plt.title('Сигма-коридоры (Signal_1)')
    plt.xlabel('Время (с)')
    plt.ylabel('Отклик')
    plt.grid(True)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "corridors", "sigma_corridors_Signal_1.png"))
    plt.close()

    # --- ШАГ 3: Валидация первого сигнала на самом себе ---
    print("\n🔍 Валидация первого сигнала на самом себе:")
    self_validation = validate(first_signal, corridor_data)
    print(self_validation)

    # --- ШАГ 4: Кросс-валидация для всех других сигналов ---
    other_signals = signals[1:]
    print(f"\n🔁 Кросс-валидация: {len(other_signals)} сигналов")

    cv_results = []

    for idx, signal in enumerate(other_signals):
        print(f"🔬 Кросс-валидация: сигнал {idx + 2}")

        percent_1sigma_list = []
        percent_2sigma_list = []
        percent_3sigma_list = []

        # Создаем 50 версий сигнала с шумом
        for _ in range(NUM_SIMULATIONS):
            noisy_signal = add_white_noise(signal, NOISE_LEVELS[0] / 100)
            val_result = validate(noisy_signal, corridor_data)
            percent_1sigma_list.append(val_result["avg_1sigma"])
            percent_2sigma_list.append(val_result["avg_2sigma"])
            percent_3sigma_list.append(val_result["avg_3sigma"])

        avg_1sigma = round(sum(percent_1sigma_list) / len(percent_1sigma_list), 2)
        avg_2sigma = round(sum(percent_2sigma_list) / len(percent_2sigma_list), 2)
        avg_3sigma = round(sum(percent_3sigma_list) / len(percent_3sigma_list), 2)

        min_1sigma = round(min(percent_1sigma_list), 2)
        max_1sigma = round(max(percent_1sigma_list), 2)
        min_2sigma = round(min(percent_2sigma_list), 2)
        max_2sigma = round(max(percent_2sigma_list), 2)
        min_3sigma = round(min(percent_3sigma_list), 2)
        max_3sigma = round(max(percent_3sigma_list), 2)

        cv_results.append({
            'тип_звена': f'signal_{idx + 2}',
            'мин_1σ': min_1sigma,
            'ср_1σ': avg_1sigma,
            'макс_1σ': max_1sigma,
            'мин_2σ': min_2sigma,
            'ср_2σ': avg_2sigma,
            'макс_2σ': max_2sigma,
            'мин_3σ': min_3sigma,
            'ср_3σ': avg_3sigma,
            'макс_3σ': max_3sigma
        })

        # Строим график кросс-валидации
        plot_validation(
            time_points,
            signal,
            corridor_data,
            os.path.join(RESULTS_DIR, "validation", f"cross_validation_signal_{idx + 2}.png")
        )
    print(percent_1sigma_list, percent_2sigma_list, percent_3sigma_list)