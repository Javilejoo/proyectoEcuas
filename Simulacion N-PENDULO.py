import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation

def n_pendulum(y, t, lengths, masses, g):
    n = len(lengths)
    dydt = np.zeros_like(y)
    dydt[:n] = y[n:]
    for i in range(n):
        dydt[n + i] = -g / lengths[i] * np.sin(y[i])
        for j in range(n):
            if i != j:
                dydt[n + i] += (g / lengths[j]) * np.sin(y[j] - y[i])
    return dydt

def simulate_n_pendulum(n, lengths, masses, initial_conditions, total_time, num_frames):
    g = 9.8
    t = np.linspace(0, total_time, num_frames)
    y0 = np.zeros(2 * n)
    y0[:n] = initial_conditions

    sol = odeint(n_pendulum, y0, t, args=(lengths, masses, g))
    return sol, t

def update(frame, sol, lines, lengths, time_line, time_ax):
    n = len(lengths)
    artists = []

    # Actualizar las líneas de los péndulos
    for i in range(n):
        lines[i].set_data([0, lengths[i] * np.sin(sol[frame, i])], [0, -lengths[i] * np.cos(sol[frame, i])])
        artists.append(lines[i])

    # Actualizar las líneas que conectan los péndulos
    for i in range(n - 1):
        j = i + 1
        if isinstance(lines[n + i], plt.Line2D):
            lines[n + i].set_data([lengths[i] * np.sin(sol[frame, i]), lengths[j] * np.sin(sol[frame, j])],
                                  [-lengths[i] * np.cos(sol[frame, i]), -lengths[j] * np.cos(sol[frame, j])])
            artists.append(lines[n + i])

    # Actualizar la línea de tiempo
    time_line.set_data(sol[:frame+1, 0], sol[:frame+1, 1])
    time_ax.relim()
    time_ax.autoscale_view()

    artists.extend([time_line])
    return artists

def plot_n_pendulum_real_time(sol, lengths):
    n = len(lengths)
    fig, (pendulum_ax, time_ax) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    
    lines = [pendulum_ax.plot([], [], 'o-', label=f'Péndulo {i + 1} (L={lengths[i]})')[0] for i in range(n)]
    lines.extend(pendulum_ax.plot([], [], 'k-') for _ in range(n - 1))
    pendulum_ax.set_title('Movimiento de un n-péndulo acoplado')
    pendulum_ax.set_xlim(-sum(lengths), sum(lengths))
    pendulum_ax.set_ylim(-max(lengths), 0)
    pendulum_ax.set_xlabel('Posición X')
    pendulum_ax.set_ylabel('Posición Y')
    pendulum_ax.legend()

    time_line, = time_ax.plot([], [], label='Trayectoria')
    time_ax.set_title('Movimiento respecto al tiempo')
    time_ax.set_xlabel('Ángulo (radianes)')
    time_ax.set_ylabel('Velocidad angular')
    time_ax.legend()

    ani = FuncAnimation(fig, update, frames=len(sol), fargs=(sol, lines, lengths, time_line, time_ax), interval=50, blit=True)
    plt.show()
    
def plot_n_pendulum(sol, lengths):
    """
    Grafica el movimiento de un sistema de n péndulos acoplados.

    Parámetros:
    sol : array
        Resultados de la simulación.
    lengths : array
        Longitudes de los péndulos.
    """
    n = len(lengths)
    for i in range(n):
        plt.plot(sol[:, i], label=f'Péndulo {i + 1} (L={lengths[i]})')

    plt.title('Movimiento de un n-péndulo acoplado')
    plt.xlabel('Tiempo')
    plt.ylabel('Ángulo')
    plt.legend()
    plt.show()

# Solicita al usuario las condiciones iniciales
n = int(input("Ingrese el número de péndulos (n): "))
lengths = [float(input(f"Ingrese la longitud del péndulo {i + 1}: ")) for i in range(n)]
masses = [float(input(f"Ingrese la masa del péndulo {i + 1}: ")) for i in range(n)]
initial_conditions = np.deg2rad([float(input(f"Ingrese el ángulo inicial del péndulo {i + 1} en grados: ")) for i in range(n)])
total_time = float(input("Ingrese el tiempo total de simulación: "))
num_frames = 500

# Simula y grafica el movimiento del péndulo en tiempo real sin historial
sol, t = simulate_n_pendulum(n, lengths, masses, initial_conditions, total_time, num_frames)
plot_n_pendulum_real_time(sol, lengths)
plot_n_pendulum(sol, lengths)
