#!/usr/bin/python


import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['figure.dpi'] = 120

data = np.loadtxt('cores.dat')
orbit_data_0 = np.loadtxt('../orbit0.dat')
orbit_data_1 = np.loadtxt('orbit1.dat')

core_col = data[:,0]
secs_col = data[:,1]

zpos_1 = orbit_data_1[:,0]
x_env_1 = orbit_data_1[:,1]

zpos_0 = orbit_data_0[:,0]
x_env_0 = orbit_data_0[:,1]

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
line1, = ax1.plot(core_col, secs_col)
line1.set_lw(2)
line1.set_ls('--')
line1.set_marker('o')
line1.set_ms(10)
line1.set_mec('r')
line1.set_mfc('w')
line1.set_mew(2)
ax1.set_xlabel('CPU Cores(Threads) #')
ax1.set_ylabel('Time [sec]')
ax1.set_title('Dakota/CG/30iters/120vars')

text_str = 'Intel(R) Xeon E5-2640 v2 (2.00GHz)' \
         + '\n' + '2CPUs x 8Cores x 2'
ax1.text(10, 700, text_str, bbox=dict(facecolor='red', alpha=0.2))

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
line2, = ax2.plot(zpos_1, x_env_1)
line2.set_color('r')
line2.set_ls('--')
line2.set_lw(2)
line2.set_marker('s')
line2.set_ms(8)
line2.set_mec('b')
line2.set_mfc('w')
line2.set_mew(1.5)
line2.set_label('After OC')

line3, = ax2.plot(zpos_0, x_env_0)
line3.set_color('g')
line3.set_ls('--')
line3.set_lw(1.5)
line3.set_marker('o')
line3.set_ms(8)
line3.set_mec('m')
line3.set_mfc('w')
line3.set_mew(1.5)
line3.set_label('Before OC')

ax2.set_xlabel('$z\,\mathrm{[m]}$')
ax2.set_ylabel('$x_0\,\mathrm{[mm]}$')
ax2.set_title('Beam Orbit')
#ax2.set_ylim([-1,1])
ax2.legend(loc=3, fontsize=14)

plt.show()



