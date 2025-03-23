import numpy as np
import matplotlib.pyplot as plt


# Task 1
# Вихідні дані
a = 0.13e-6  # м²/с
L = 0.05  # м
T = 720  # сек
N = 100  # Кількість вузлів
h_t = 0.015  # сек
alpha = 9  # °C (ліва межа)
beta = 28  # °C (права межа)

# Просторова сітка
h_y = L / N
y = np.linspace(0, L, N + 1)

# Початковий розподіл температур
u = np.zeros(N + 1)
u[0] = alpha
u[-1] = beta


# Метод Рунге-Кутта 4-го порядку
def f(u):
    dudt = np.zeros_like(u)
    for i in range(1, N):
        dudt[i] = a * (u[i + 1] - 2 * u[i] + u[i - 1]) / h_y ** 2
    return dudt


# Часова сітка
time_steps = int(T / h_t)

# Зберігаємо рішення
results = [u.copy()]

# Основний цикл Рунге-Кутта
for n in range(time_steps):
    k1 = f(u)
    k2 = f(u + h_t / 2 * k1)
    k3 = f(u + h_t / 2 * k2)
    k4 = f(u + h_t * k3)
    u = u + (h_t / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Граничні умови
    u[0] = alpha
    u[-1] = beta
    results.append(u.copy())

results = np.array(results)

# Побудова температурного профілю при t = T
plt.figure(figsize=(8, 5))
plt.plot(y, results[-1], label=f't = {T} сек')
plt.xlabel('Координата y (м)')
plt.ylabel('Температура (°C)')
plt.title('Температурний профіль у стінці при t = T')
plt.legend()
plt.grid(True)
plt.show()

# Побудова еволюції температури у часі
plt.figure(figsize=(8, 5))
for n in [0, int(0.25 * time_steps), int(0.5 * time_steps), int(0.75 * time_steps), -1]:
    plt.plot(y, results[n], label=f't = {n * h_t:.1f} сек')

plt.xlabel('Координата y (м)')
plt.ylabel('Температура (°C)')
plt.title('Еволюція температурного профілю у часі')
plt.legend()
plt.grid(True)
plt.show()



