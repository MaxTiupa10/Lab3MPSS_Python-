import numpy as np
import matplotlib.pyplot as plt

# Вихідні дані
# Теплопровідність матеріалу (м²/с)
a = 0.13e-6
# Товщина стінки (м)
L = 0.05
# Час моделювання (сек)
T = 720
# Кількість вузлів розбиття
N = 100
# Часовий крок (сек)
h_t = 0.015
# Температура на лівій та правій межах (°C)
alpha = 9
beta = 28

# Просторова сітка
h_y = L / N  # Крок дискретизації по простору
y = np.linspace(0, L, N + 1)  # Координати вузлів

# Початковий розподіл температури (задаємо внутрішні точки нулями)
u = np.zeros(N + 1)
u[0] = alpha  # Гранична умова (ліва межа)
u[-1] = beta  # Гранична умова (права межа)


# Функція для розрахунку похідної температури (теплопровідність)
def f(u):
    dudt = np.zeros_like(u)
    for i in range(1, N):  # Обчислюємо тільки для внутрішніх точок
        dudt[i] = a * (u[i + 1] - 2 * u[i] + u[i - 1]) / h_y ** 2
    return dudt


# Кількість часових кроків
time_steps = int(T / h_t)

# Масив для збереження рішень у кожен момент часу
results = [u.copy()]

# Основний цикл методу Рунге-Кутта 4-го порядку
for n in range(time_steps):
    k1 = f(u)
    k2 = f(u + h_t / 2 * k1)
    k3 = f(u + h_t / 2 * k2)
    k4 = f(u + h_t * k3)

    # Оновлюємо значення температури у вузлах
    u[1:N] = u[1:N] + (h_t / 6) * (k1[1:N] + 2 * k2[1:N] + 2 * k3[1:N] + k4[1:N])

    # Відновлюємо граничні умови
    u[0] = alpha
    u[-1] = beta

    # Зберігаємо поточний розподіл температури
    results.append(u.copy())

# Перетворюємо список у масив для зручності аналізу
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
time_points = [0, int(0.25 * time_steps), int(0.5 * time_steps), int(0.75 * time_steps), -1]
for n in time_points:
    plt.plot(y, results[n], label=f't = {n * h_t:.1f} сек')

plt.xlabel('Координата y (м)')
plt.ylabel('Температура (°C)')
plt.title('Еволюція температурного профілю у часі')
plt.legend()
plt.grid(True)
plt.show()
