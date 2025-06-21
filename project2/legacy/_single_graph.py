import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# === ПАРАМЕТРЫ СИМУЛЯЦИЙ ===
NUM_SIMULATIONS = 50
NOISE_LEVEL = 0.10  # 10% белого шума

# === ЧТЕНИЕ ДАННЫХ ИЗ ФАЙЛА ===
def load_data(filename):
    data = pd.read_csv(filename, header=None)
    time_points = data.iloc[:, 0].values
    signal = data.iloc[:, 1].values
    return time_points, signal

# === ДОБАВЛЕНИЕ БЕЛОГО ШУМА ===
def add_white_noise(signal, noise_level=0.1):
    signal_std = np.std(signal)
    noise = np.random.normal(0, noise_level * signal_std, size=len(signal))
    noisy_signal = signal + noise
    return noisy_signal

# === ПОСТРОЕНИЕ КОРРИДОРОВ ===
def build_corridors(signals):
    mean_signal = np.mean(signals, axis=0)
    std_signal = np.std(signals, axis=0)

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

# === ОСНОВНОЙ ЦИКЛ ===
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # Чтение данных
    time_points, original_signal = load_data("data.csv")
    print("Входной сигнал:", original_signal[:10])

    # Создание массива для хранения симуляций
    all_simulations = []

    # Выполнение 50 симуляций с белым шумом
    for _ in range(NUM_SIMULATIONS):
        noisy_signal = add_white_noise(original_signal, NOISE_LEVEL)
        all_simulations.append(noisy_signal)

    # Построение коридоров
    corridor_data = build_corridors(all_simulations)

    # Сохранение графика оригинального сигнала
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, original_signal, label='Оригинальный сигнал')
    plt.title('Выходной сигнал модели сетчатки глаза')
    plt.xlabel('Время (с)')
    plt.ylabel('Отклик')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/signal.png")

    # Сохранение графика среднего сигнала с коридорами
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, corridor_data["mean"], '--', color='red', label='Среднее')
    plt.fill_between(time_points, corridor_data["one_sigma_lower"], corridor_data["one_sigma_upper"], color='green', alpha=0.2, label='±1σ')
    plt.fill_between(time_points, corridor_data["two_sigma_lower"], corridor_data["two_sigma_upper"], color='orange', alpha=0.2, label='±2σ')
    plt.fill_between(time_points, corridor_data["three_sigma_lower"], corridor_data["three_sigma_upper"], color='yellow', alpha=0.2, label='±3σ')

    plt.title('Сигма-коридоры')
    plt.xlabel('Время (с)')
    plt.ylabel('Отклик')
    plt.grid(True)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    plt.tight_layout()
    plt.savefig("results/signal_corridors.png")

    print("✅ Графики сохранены в папке results/")