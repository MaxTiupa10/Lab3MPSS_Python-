import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Вихідні дані
a = 0.13e-6  # м²/с (коефіцієнт теплопровідності для гуми)
L = 0.05  # м (довжина)
T = 720  # сек (час)
N = 100  # кількість вузлів
h_t = 0.015  # сек (крок по часу)
alpha = 9  # °C (ліва межа температури)
beta = 28  # °C (права межа температури)

# Кроки по простору та часу
dx = L / (N - 1)  # крок по простору
dt = h_t  # крок по часу
M = int(T / dt)  # кількість кроків по часу

# Ініціалізація температури в просторі
u_numeric = np.zeros((N, M))  # числове рішення
x = np.linspace(0, L, N)  # просторові координати
t = np.linspace(0, T, M)  # часові координати

# Початкові умови
u_numeric[:, 0] = np.linspace(alpha, beta, N)  # Початкова температура

# Числове рішення за допомогою методу скінченних різниць
for n in range(0, M - 1):
    for i in range(1, N - 1):
        u_numeric[i, n + 1] = u_numeric[i, n] + (a * dt / dx ** 2) * (
                u_numeric[i + 1, n] - 2 * u_numeric[i, n] + u_numeric[i - 1, n])

    # Граничні умови залишаються незмінними
    u_numeric[0, n + 1] = alpha
    u_numeric[-1, n + 1] = beta


# Аналітичне рішення (обмежене 30 доданками)
def analytical_solution(x, t, a, L, alpha, beta, N_terms=30):
    u_analytic = np.zeros((len(x), len(t)))
    for n in range(1, N_terms + 1):
        lambda_n = (n * np.pi) / L
        B_n = (2 * (beta - alpha) / (n * np.pi)) * (-1) ** n  # Формула для коефіцієнтів
        for i, xi in enumerate(x):
            u_analytic[i, :] += B_n * np.sin(lambda_n * xi) * np.exp(-a * (lambda_n ** 2) * t)
    u_analytic += alpha + (beta - alpha) * x[:, None] / L  # Додаємо лінійну частину
    return u_analytic


# Обчислення аналітичного рішення
u_analytic = analytical_solution(x, t, a, L, alpha, beta)

# Візуалізація числового та аналітичного розв'язків
X, Y = np.meshgrid(x, t)
fig = plt.figure(figsize=(12, 8))

# Числове рішення
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, u_numeric.T, cmap='viridis')
ax1.set_title('Числове рішення')
ax1.set_xlabel('Простір (x)')
ax1.set_ylabel('Час (t)')
ax1.set_zlabel('Температура (u)')

# Аналітичне рішення
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, u_analytic.T, cmap='viridis')
ax2.set_title('Аналітичне рішення')
ax2.set_xlabel('Простір (x)')
ax2.set_ylabel('Час (t)')
ax2.set_zlabel('Температура (u)')

plt.show()


# Обчислення похибок
def calculate_errors(u_numeric, u_analytic):
    # MAE (Max Absolute Error)
    mae = np.max(np.abs(u_numeric - u_analytic))

    # MSE (Mean Squared Error)
    mse = np.mean((u_numeric - u_analytic) ** 2)

    return mae, mse


# Обчислення MAE та MSE
mae, mse = calculate_errors(u_numeric, u_analytic)

print(f"Максимальна абсолютна похибка (MAE): {mae:.6f}")
print(f"Середньостатистична похибка (MSE): {mse:.6f}")
