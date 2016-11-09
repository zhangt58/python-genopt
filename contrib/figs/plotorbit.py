#!/usr/bin/python


import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['figure.dpi'] = 100

orbit_data_0 = np.loadtxt('orbit0.dat')
orbit_data_1 = np.loadtxt('orbit7.dat')

zpos_0 = orbit_data_0[:,0]
x_env_0 = orbit_data_0[:,1]
y_env_0 = orbit_data_0[:,2]

zpos_1 = orbit_data_1[:,0]
x_env_1 = orbit_data_1[:,1]
y_env_1 = orbit_data_1[:,2]


fig1 = plt.figure(1)
# x
ax1 = fig1.add_subplot(211)
line1, = ax1.plot(zpos_1, x_env_1)
line1.set_color('r')
line1.set_ls('--')
line1.set_lw(2)
line1.set_marker('s')
line1.set_ms(8)
line1.set_mec('b')
line1.set_mfc('w')
line1.set_mew(1.5)
line1.set_label('After OC')

line2, = ax1.plot(zpos_0, x_env_0)
line2.set_color('g')
line2.set_ls('--')
line2.set_lw(1.5)
line2.set_marker('o')
line2.set_ms(8)
line2.set_mec('m')
line2.set_mfc('w')
line2.set_mew(1.5)
line2.set_label('Before OC')

ax1.set_ylabel('$x_0\,\mathrm{[mm]}$')
#ax1.set_ylim([-10,10])
ax1.set_title('Beam Orbit')
ax1.legend(loc=3, fontsize=14)

# y
ax2 = fig1.add_subplot(212)
line3, = ax2.plot(zpos_1, y_env_1)
line3.set_color('r')
line3.set_ls('--')
line3.set_lw(2)
line3.set_marker('s')
line3.set_ms(8)
line3.set_mec('b')
line3.set_mfc('w')
line3.set_mew(1.5)
line3.set_label('After OC')

line4, = ax2.plot(zpos_0, y_env_0)
line4.set_color('g')
line4.set_ls('--')
line4.set_lw(1.5)
line4.set_marker('o')
line4.set_ms(8)
line4.set_mec('m')
line4.set_mfc('w')
line4.set_mew(1.5)
line4.set_label('Before OC')

ax2.set_xlabel('$z\,\mathrm{[m]}$')
ax2.set_ylabel('$y_0\,\mathrm{[mm]}$')
#ax2.set_ylim([-10,10])
ax2.legend(loc=3, fontsize=14)



plt.show()



