import numpy as np
from matplotlib.widgets import Slider

g = 9.81
v0 = 20        # initial speed
angle = 45     # launch angle in degrees

angle_rad = np.radians(angle)   # convert to radians for numpy
vx = v0 * np.cos(angle_rad)     # horizontal component
vy = v0 * np.sin(angle_rad)     # vertical component

t_flight = 2 * vy / g           # total time in the air
t = np.linspace(0, t_flight, 500)

x = vx * t
y = vy * t - 0.5 * g * t**2

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.set_xlim(0, 1100)
ax.set_ylim(0, 550)
plt.subplots_adjust(bottom=0.25)

line, = ax.plot(x, y, color='cyan')
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Height (m)')
ax.set_title('Projectile Motion')

ax_v0    = fig.add_axes([0.15, 0.13, 0.70, 0.03])
ax_angle = fig.add_axes([0.15, 0.06, 0.70, 0.03])

slider_v0    = Slider(ax_v0,    'Speed',  5, 100, valinit=20)
slider_angle = Slider(ax_angle, 'Angle',  1,  89, valinit=45)
def update(val):
    v0 = slider_v0.val
    angle = slider_angle.val
    
    angle_rad = np.radians(angle)
    vx = v0 * np.cos(angle_rad)
    vy = v0 * np.sin(angle_rad)
    
    t_flight = 2 * vy / g
    t = np.linspace(0, t_flight, 500)
    
    x = vx * t
    y = vy * t - 0.5 * g * t**2
    
    line.set_data(x, y)
    fig.canvas.draw_idle()
slider_v0.on_changed(update)
slider_angle.on_changed(update)

plt.show()