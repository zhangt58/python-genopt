#!/usr/bin/python


from flame import Machine
import matplotlib.pyplot as plt


f1 = '../lattice/test_392.lat'
#f2 = 'optout.lat'
f1 = 'opt6_1.lat'
f2 = 'opt6_2.lat'

m1 = Machine(open(f1))
s1 = m1.allocState({})
r1 = m1.propagate(s1, 0, len(m1), range(len(m1)))
z1 = [r1[i][1].pos for i in range(len(m1))]
x1 = [r1[i][1].moment0_env[0] for i in range(len(m1))]
y1 = [r1[i][1].moment0_env[2] for i in range(len(m1))]

m2 = Machine(open(f2))
s2 = m2.allocState({})
r2 = m2.propagate(s2, 0, len(m2), range(len(m2)))
z2 = [r2[i][1].pos for i in range(len(m2))]
x2 = [r2[i][1].moment0_env[0] for i in range(len(m2))]
y2 = [r2[i][1].moment0_env[2] for i in range(len(m2))]

fig = plt.figure()
ax = fig.add_subplot(111)
lx1, = ax.plot(z1, x1, 'r--', lw=2, label='x0')
lx2, = ax.plot(z2, x2, 'b-', lw=2, label='x')
ly1, = ax.plot(z1, y1, 'm--', lw=2, label='y0')
ly2, = ax.plot(z2, y2, 'g-', lw=2, label='y')

ax.legend()

plt.show()
