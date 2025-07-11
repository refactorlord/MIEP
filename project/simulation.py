import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def simulate_system(num_simulations=100, system_type="aperiodic", params=None, noise_level=0.1):
    if params is None:
        params = {}

    t = np.linspace(0, 10, 500)
    u = np.ones_like(t)

    all_signals = []

    for _ in range(num_simulations):
        if system_type == "aperiodic":
            T = params.get('T', 1)
            y_prev = 0
            dt = t[1] - t[0]
            output = []
            for i in range(len(t)):
                dy = (u[i] - y_prev) * dt / T
                y_prev += dy
                output.append(y_prev)

        elif system_type == "oscillatory":
            T = params.get('T', 1)
            xi = params.get('xi', 0.5)
            y_prev = 0
            v_prev = 0
            dt = t[1] - t[0]
            output = []
            for i in range(len(t)):
                a = (u[i] - 2 * xi * T * v_prev - y_prev) / (T ** 2)
                v = v_prev + a * dt
                y = y_prev + v * dt
                y_prev, v_prev = y, v
                output.append(y)

        elif system_type == "integrating":
            k = params.get('k', 1)
            y_prev = 0
            dt = t[1] - t[0]
            output = []
            for i in range(len(t)):
                y_prev += u[i] * dt * k
                output.append(y_prev)

        else:
            raise ValueError(f"Неизвестный тип звена: {system_type}")

        signal_std = np.std(output)
        noise = np.random.normal(0, noise_level * signal_std, size=len(output))
        noisy_output = np.array(output) + noise
        all_signals.append(noisy_output.tolist())

    return all_signals


def save_to_file(data, filename):
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)