#!/usr/bin/python


import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['lines.linewidth'] = 2.0

orbit_data_0 = np.loadtxt('orbit0.dat')
orbit_data_1 = np.loadtxt('../orbit1_0.dat')
orbit_data_2 = np.loadtxt('../orbit1_1.dat')
orbit_data_3 = np.loadtxt('../orbit1_5.dat')

zpos_0 = orbit_data_0[:,0]
x_env_0 = orbit_data_0[:,1]
y_env_0 = orbit_data_0[:,2]

zpos_1 = orbit_data_1[:,0]
x_env_1 = orbit_data_1[:,1]
y_env_1 = orbit_data_1[:,2]

zpos_2 = orbit_data_2[:,0]
x_env_2 = orbit_data_2[:,1]
y_env_2 = orbit_data_2[:,2]

zpos_3 = orbit_data_3[:,0]
x_env_3 = orbit_data_3[:,1]
y_env_3 = orbit_data_3[:,2]

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
line0, = ax1.plot(zpos_0, x_env_0, '--', label='original')
line1, = ax1.plot(zpos_1, x_env_1, label='oc-0mm')
line2, = ax1.plot(zpos_2, x_env_2, label='oc-1mm')
line3, = ax1.plot(zpos_3, x_env_3, label='oc-5mm')
ax1.set_xlabel('$z\,\mathrm{[m]}$')
ax1.set_ylabel('$x_0\,\mathrm{[mm]}$')
ax1.legend(loc=3, fontsize=14)
ax1.grid()

plt.show()



