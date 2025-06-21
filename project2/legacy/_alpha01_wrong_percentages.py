import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# === ПАРАМЕТРЫ СИМУЛЯЦИЙ ===
NUM_SIMULATIONS = 50     # количество симуляций
NOISE_LEVEL = 0.10       # уровень белого шума (%)

# === ЧТЕНИЕ ДАННЫХ ИЗ CSV ===
def load_data(filename):
    data = pd.read_csv(filename, header=None)
    time_points = data.iloc[:, 0].values
    signals = [data.iloc[:, i].values for i in range(1, data.shape[1])]
    return time_points, signals

# === ДОБАВЛЕНИЕ БЕЛОГО ШУМА ===
def add_white_noise(signal, noise_level=0.1):
    signal_std = np.std(signal)
    noise = np.random.normal(0, noise_level * signal_std, size=len(signal))
    return signal + noise

# === ПОСТРОЕНИЕ КОРРИДОРОВ ===
def build_corridors(simulations):
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

# === ВАЛИДАЦИЯ ОДНОГО СИГНАЛА ===
def validate(test_signal, corridor_data):
    length = min(len(test_signal), len(corridor_data["mean"]))
    in_1sigma = ((test_signal > corridor_data["one_sigma_lower"][:length]) & (test_signal < corridor_data["one_sigma_upper"][:length])).sum() / length * 100
    in_2sigma = ((test_signal > corridor_data["two_sigma_lower"][:length]) & (test_signal < corridor_data["two_sigma_upper"][:length])).sum() / length * 100
    in_3sigma = ((test_signal > corridor_data["three_sigma_lower"][:length]) & (test_signal < corridor_data["three_sigma_upper"][:length])).sum() / length * 100

    return {
        "avg_1sigma": round(in_1sigma, 2),
        "avg_2sigma": round(in_2sigma, 2),
        "avg_3sigma": round(in_3sigma, 2)
    }

# === КРОСС-ВАЛИДАЦИЯ ===
def cross_validate(signals_list, corridor_data):
    results = []
    for idx, signal in enumerate(signals_list):
        print(f"Кросс-валидация: сигнал {idx + 1}")
        validations = []

        for _ in range(NUM_SIMULATIONS):
            noisy_signal = add_white_noise(signal, NOISE_LEVEL)
            val_result = validate(noisy_signal, corridor_data)
            validations.append(val_result)

        avg_1sigma = np.mean([v['avg_1sigma'] for v in validations])
        avg_2sigma = np.mean([v['avg_2sigma'] for v in validations])
        avg_3sigma = np.mean([v['avg_3sigma'] for v in validations])

        results.append({
            'signal_index': idx + 1,
            'avg_1sigma': round(avg_1sigma, 2),
            'avg_2sigma': round(avg_2sigma, 2),
            'avg_3sigma': round(avg_3sigma, 2)
        })

    return results

# === ГРАФИК ОДНОГО СИГНАЛА С КОРИДОРАМИ ===
def plot_validation(time_points, test_signal, corridor_data, filename="results/validation_plot.png"):
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, test_signal, label='Тестовый сигнал')
    plt.plot(time_points, corridor_data["mean"], '--', color='red', label='Среднее')
    plt.fill_between(time_points, corridor_data["one_sigma_lower"], corridor_data["one_sigma_upper"], color='green', alpha=0.2, label='±1σ')
    plt.fill_between(time_points, corridor_data["two_sigma_lower"], corridor_data["two_sigma_upper"], color='orange', alpha=0.2, label='±2σ')
    plt.fill_between(time_points, corridor_data["three_sigma_lower"], corridor_data["three_sigma_upper"], color='yellow', alpha=0.2, label='±3σ')

    plt.title('Валидация: Тестовый сигнал и сигма-коридоры')
    plt.xlabel('Время (с)')
    plt.ylabel('Отклик')
    plt.grid(True)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# === ОСНОВНОЙ ЦИКЛ ===
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # Загрузка данных
    time_points, signals = load_data("data.csv")
    print(f"✅ Прочитано {len(signals)} сигналов")

    # === ВАЛИДАЦИЯ ПЕРВОГО СИГНАЛА ===
    first_signal = signals[0]
    print("\n🧪 Обычная валидация: сигнал 1")
    all_simulations = []

    # 50 симуляций с шумом для построения коридоров
    for _ in range(NUM_SIMULATIONS):
        noisy = add_white_noise(first_signal, NOISE_LEVEL)
        all_simulations.append(noisy)

    # Построение коридоров
    corridor_data = build_corridors(all_simulations)

    # График валидации
    plot_validation(time_points, first_signal, corridor_data, "results/validation_signal_1.png")
    print("✅ Коридоры построены на основе 50 симуляций для первого сигнала")

    # === КРОСС-ВАЛИДАЦИЯ ОСТАЛЬНЫХ СИГНАЛОВ ===
    other_signals = signals[1:]
    print(f"\n🔁 Кросс-валидация: {len(other_signals)} сигналов")

    cv_results = []
    for idx, signal in enumerate(other_signals):
        print(f"🔬 Кросс-валидация: сигнал {idx + 2}")

        validations = []
        for _ in range(NUM_SIMULATIONS):
            noisy = add_white_noise(signal, NOISE_LEVEL)
            val_result = validate(noisy, corridor_data)
            validations.append(val_result)

        avg_1sigma = np.mean([v['avg_1sigma'] for v in validations])
        avg_2sigma = np.mean([v['avg_2sigma'] for v in validations])
        avg_3sigma = np.mean([v['avg_3sigma'] for v in validations])

        cv_results.append({
            'signal_index': idx + 2,
            'avg_1sigma': round(avg_1sigma, 2),
            'avg_2sigma': round(avg_2sigma, 2),
            'avg_3sigma': round(avg_3sigma, 2)
        })

        # График кросс-валидации
        plt.figure(figsize=(12, 6))
        plt.plot(time_points, signal, label=f'Сигнал {idx + 2}')
        plt.plot(time_points, corridor_data["mean"], '--', color='red', label='Среднее')
        plt.fill_between(time_points, corridor_data["one_sigma_lower"], corridor_data["one_sigma_upper"], color='green', alpha=0.2, label='±1σ')
        plt.fill_between(time_points, corridor_data["two_sigma_lower"], corridor_data["two_sigma_upper"], color='orange', alpha=0.2, label='±2σ')
        plt.fill_between(time_points, corridor_data["three_sigma_lower"], corridor_data["three_sigma_upper"], color='yellow', alpha=0.2, label='±3σ')

        plt.text(0.5, 0.95,
                 f"Signal {idx + 2}\n"
                 f"В 1σ: {avg_1sigma:.2f}%\n"
                 f"В 2σ: {avg_2sigma:.2f}%\n"
                 f"В 3σ: {avg_3sigma:.2f}%",
                 ha='center', va='top', transform=plt.gca().transAxes, fontsize=10)

        plt.title(f'Кросс-валидация: сигнал {idx + 2}')
        plt.xlabel('Время (с)')
        plt.ylabel('Отклик')
        plt.grid(True)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)
        plt.tight_layout()
        plt.savefig(f"results/CV_signal_{idx + 2}.png")
        plt.close()

    # === СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ===
    df = pd.DataFrame(cv_results)
    df.to_csv("results/cross_validation_summary.csv", index=False, encoding="utf-8-sig")
    print("\n📊 Результаты кросс-валидации:")
    print(df.to_string(index=False))

    print("\n✅ Все графики и таблица сохранены в папке results/")