import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

# Constants
g = 9.81
dt = 0.02

# Initial state [theta1, omega1, theta2, omega2]
state = np.array([np.pi/2, 0.0, np.pi/2 + 0.01, 0.0])
params = {'L1': 1.0, 'L2': 1.0, 'm1': 1.0, 'm2': 1.0}
trail_x, trail_y = [], []
paused = False

# Equations of motion (Lagrangian mechanics)
def derivatives(state, L1, L2, m1, m2):
    t1, w1, t2, w2 = state
    d = t1 - t2
    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(d)**2
    den2 = (L2 / L1) * den1

    dw1 = (m2 * L1 * w1**2 * np.sin(d) * np.cos(d) +
           m2 * g * np.sin(t2) * np.cos(d) +
           m2 * L2 * w2**2 * np.sin(d) -
           (m1 + m2) * g * np.sin(t1)) / den1

    dw2 = (-m2 * L2 * w2**2 * np.sin(d) * np.cos(d) +
            (m1 + m2) * g * np.sin(t1) * np.cos(d) -
            (m1 + m2) * L1 * w1**2 * np.sin(d) -
            (m1 + m2) * g * np.sin(t2)) / den2

    return np.array([w1, dw1, w2, dw2])

# RK4 integrator
def rk4_step(state, L1, L2, m1, m2):
    k1 = derivatives(state,                  L1, L2, m1, m2)
    k2 = derivatives(state + 0.5 * dt * k1, L1, L2, m1, m2)
    k3 = derivatives(state + 0.5 * dt * k2, L1, L2, m1, m2)
    k4 = derivatives(state + dt * k3,        L1, L2, m1, m2)
    return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

# Figure setup
fig = plt.figure(figsize=(8, 8))
fig.patch.set_facecolor('black')

ax = fig.add_axes([0.05, 0.30, 0.90, 0.65])
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.set_facecolor('black')
ax.set_title('Double Pendulum  |  Click to reposition', color='white', fontsize=10)
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('gray')

# Plot elements
line, = ax.plot([], [], 'o-', color='white', lw=2, markersize=10, markerfacecolor='white')
trail, = ax.plot([], [], '-', color='cyan', lw=0.8, alpha=0.7)
drag_dot, = ax.plot([], [], 'o', color='red', markersize=12, alpha=0.6)

# Sliders
ax_L1 = fig.add_axes([0.15, 0.20, 0.70, 0.025])
ax_L2 = fig.add_axes([0.15, 0.15, 0.70, 0.025])
ax_m1 = fig.add_axes([0.15, 0.10, 0.70, 0.025])
ax_m2 = fig.add_axes([0.15, 0.05, 0.70, 0.025])

slider_L1 = Slider(ax_L1, 'Length 1', 0.2, 2.0, valinit=1.0, color='deepskyblue')
slider_L2 = Slider(ax_L2, 'Length 2', 0.2, 2.0, valinit=1.0, color='deepskyblue')
slider_m1 = Slider(ax_m1, 'Mass 1',   0.1, 3.0, valinit=1.0, color='magenta')
slider_m2 = Slider(ax_m2, 'Mass 2',   0.1, 3.0, valinit=1.0, color='magenta')

for sl in [slider_L1, slider_L2, slider_m1, slider_m2]:
    sl.label.set_color('white')
    sl.valtext.set_color('white')

def update_params(val):
    params['L1'] = slider_L1.val
    params['L2'] = slider_L2.val
    params['m1'] = slider_m1.val
    params['m2'] = slider_m2.val
    trail_x.clear()
    trail_y.clear()

slider_L1.on_changed(update_params)
slider_L2.on_changed(update_params)
slider_m1.on_changed(update_params)
slider_m2.on_changed(update_params)

# Mouse interaction
def on_press(event):
    global state, paused
    if event.inaxes == ax:
        paused = True
        trail_x.clear()
        trail_y.clear()
        t1 = np.arctan2(event.xdata, -event.ydata)
        t2 = t1 + 0.01
        state = np.array([t1, 0.0, t2, 0.0])
        drag_dot.set_data([event.xdata], [event.ydata])

def on_release(event):
    global paused
    if event.inaxes == ax:
        paused = False
        drag_dot.set_data([], [])

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)

# Animation loop
def animate(i):
    global state
    if not paused:
        state = rk4_step(state, params['L1'], params['L2'],
                                params['m1'], params['m2'])

    L1, L2 = params['L1'], params['L2']
    reach = L1 + L2 + 0.2
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    x1 = L1 * np.sin(state[0])
    y1 = -L1 * np.cos(state[0])
    x2 = x1 + L2 * np.sin(state[2])
    y2 = y1 - L2 * np.cos(state[2])

    if not paused:
        trail_x.append(x2)
        trail_y.append(y2)
        if len(trail_x) > 500:
            trail_x.pop(0)
            trail_y.pop(0)

    line.set_data([0, x1, x2], [0, y1, y2])
    trail.set_data(trail_x, trail_y)
    return line, trail, drag_dot

ani = animation.FuncAnimation(fig, animate, frames=10000, interval=20, blit=True)
plt.show()
