#!/usr/bin/python

#
# just show env
#
# Tong Zhang <zhangt@frib.msu.edu>
# 2016-10-16 20:23:35 PM EDT
#

import numpy as np
import time
from flame import Machine
import matplotlib.pyplot as plt


lat_fid = open('../lattice/test_392.lat', 'r')
m = Machine(lat_fid)
s = m.allocState({})
r = m.propagate(s, 0, len(m), observe=range(len(m)))
bpms = m.find(type='bpm')
x, y = np.array([[r[i][1].moment0_env[j] for i in bpms] 
                    for j in [0,2]])
pos = np.array([r[i][1].pos for i in bpms])

np.savetxt('orbit0.dat',
           np.vstack((pos, x, y)).T,
           fmt="%22.14e",
           comments='# orbit data saved at ' + time.ctime() + '\n',
           header="#{0:^22s} {1:^22s} {2:^22s}".format(
               "zpos [m]", "x [mm]", "y [mm]"),
           delimiter=' ')

fig = plt.figure()
ax1 = fig.add_subplot(211)
linex, = ax1.plot(pos, x, 'r-', label='$x$')
#ax2 = fig.add_subplot(212)
#lineryx, = ax2.plot(pos, y, 'b-', label='$y$')

ax1.legend()
#ax2.legend()
plt.show()

