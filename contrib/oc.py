#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# Tong Zhang <zhangt@frib.msu.edu>
# 2016-10-16 20:20:57 PM EDT
#

from flame import Machine
import numpy as np
import matplotlib.pyplot as plt

lat_fid = open('test.lat', 'r')
m = Machine(lat_fid)

## all BPMs and Correctors (both horizontal and vertical)
bpm_ids, cor_ids = m.find(type='bpm'), m.find(type='orbtrim')
corh_ids = cor_ids[0::2]
corv_ids = cor_ids[1::2]

observe_ids = bpm_ids
## before distortion

s = m.allocState({})
r = m.propagate(s, 0, len(m), observe=range(len(m)))
x, y = np.array([[r[i][1].moment0_env[0] for i in range(len(m))] for j in [0,2]])
pos = np.array([r[i][1].pos for i in range(len(m))])

fig1 = plt.figure(figsize=(10, 8), dpi=120)
ax1 = fig1.add_subplot(111)
linex, = ax1.plot(pos[observe_ids], x[observe_ids], 'r-', 
                  alpha=0.6, 
                  label='$\mathrm{ref\;orbit}$')
linex.set_lw(2)


## apply random kicks
N = 1
#corh_ids_enabled = np.random.choice(corh_ids, size=N)
#corh_val_enabled = 5e-3 * (np.random.random(size=N) * (2 - 1) + 1)
corh_ids_enabled = np.array([392])
corh_val_enabled = np.array([0.005])
for i, v in zip(corh_ids_enabled, corh_val_enabled):
    m.reconfigure(i, {'theta_x': v})

"""
for i, v in zip(corh_sel, corh_val):
    m.reconfigure(i, {'theta_x': v})
for i, v in zip(corv_sel, corv_val):
    m.reconfigure(i, {'theta_y': v})
"""

s_tmp = m.allocState({})
r_tmp = m.propagate(s_tmp, 0, len(m), observe=range(len(m)))
x_tmp, y_tmp = np.array([[r_tmp[i][1].moment0_env[0] for i in range(len(m))] for j in [0,2]])
pos = np.array([r_tmp[i][1].pos for i in range(len(m))])


# data plot
linex_tmp, = ax1.plot(pos[observe_ids], x_tmp[observe_ids], 'b--', 
                      alpha=0.8,
                      label='$\mathrm{kicked\;orbit}$')
linex_tmp.set_lw(2)

## mark the enabled kickers
corr = ax1.scatter(pos[corh_ids_enabled], x_tmp[corh_ids_enabled],
                   c='m', alpha=0.8, s=100,
                   label=r"$\mathrm{Kicker}$")


#plt.show()


## correct orbit
# define objective function to minimize
#def obj_func(cor_val, cor_ids):
#    """ Objective function for `minimize`, calculate the distance
#    to the ideal trajectory
#
#    :param cor_val: corrector strength/values, list/array, [rad]
#    :param cor_ids: corrector id numbers, list/array
#    :return: sum of the square of the deviation between present
#             ideal trajectory
#    """
#    for i, v in zip(cor_ids, cor_val):
#        m.reconfigure(i, {'theta_x': v})  # horizontal only
#
#    s_tmp = m.allocState({})
#    r_tmp = m.propagate(s_tmp, 0, len(m), observe=range(len(m)))
#    x_tmp, y_tmp = np.array([[r_tmp[i][1].moment0_env[j] 
#                                for i in observe_ids] 
#                                    for j in [0, 2]])
#    #return np.sum((x_tmp - x)**2)
#    #return np.sum((x_tmp)**2)
#    xsq = x_tmp * x_tmp
#    return xsq.mean() * xsq.max()
#    #return np.sum(xsq)
#
#print obj_func(corh_ids_enabled, corh_val_enabled)

def obj_func(cor_val, cor_ids):
    """ Objective function for `minimize`, calculate the distance
    to the ideal trajectory

    :param cor_val: corrector strength/values, list/array, [rad]
    :param cor_ids: corrector id numbers, list/array
    :return: sum of the square of the deviation between present
             ideal trajectory
    """
    corh_val, corv_val = cor_val[0::2], cor_val[1::2]
    corh_ids, corv_ids = cor_ids[0::2], cor_ids[1::2]
    for i, v in zip(corh_ids, corh_val):
        m.reconfigure(i, {'theta_x': v})
    for i, v in zip(corv_ids, corv_val):
        m.reconfigure(i, {'theta_y': v})

    s_tmp = m.allocState({})
    r_tmp = m.propagate(s_tmp, 0, len(m), observe=range(len(m)))
    x_tmp, y_tmp = np.array([[r_tmp[i][1].moment0_env[j] 
                                for i in observe_ids] 
                                    for j in [0, 2]])
    #return np.sum((x_tmp - x)**2)
    #return np.sum((x_tmp)**2)
    xsq = x_tmp * x_tmp
    return np.sum(xsq)
    #return xsq.mean() * xsq.max()


# select correctors, H
#NC = 20
#corh_ids_se = np.random.choice(corh_ids, size=NC)
#corh_val_se = 0. * (np.random.random(size=NC) * (2 - 1) + 1)

#corh_ids_se = corh_ids_enabled
#corh_val_se = [0.005]

cor_ids_se = m.find(type='orbtrim')[45:61]
#cor_ids_se = m.find(type='orbtrim')[34:50]
#cor_ids_se = m.find(type='orbtrim')[44:50]
#print cor_ids_se
#import sys
#sys.exit(1)

#corh_ids_se = cor_ids_se[0::2]
#corv_ids_se = cor_ids_se[1::2]
cor_val_se = [1e-4]*len(cor_ids_se)
#corh_val_se = [1e-4] * len(corh_ids_se)
#corv_val_se = [1e-4] * len(corv_ids_se)


from scipy.optimize import minimize

res = minimize(obj_func, cor_val_se, args=(cor_ids_se,),
                #method='Nelder-Mead',
    method='L-BFGS-B', options={'disp':True}
    #method='SLSQP', options={'maxiter':200, 'disp':True}
    )

print res.x
cor_val = res.x

# show corrected result
corh_val, corv_val = cor_val[0::2], cor_val[1::2]
corh_ids, corv_ids = cor_ids_se[0::2], cor_ids_se[1::2]
for i, v in zip(corh_ids, corh_val):
    m.reconfigure(i, {'theta_x': v})
for i, v in zip(corv_ids, corv_val):
    m.reconfigure(i, {'theta_y': v})

s_oc = m.allocState({})
r_oc = m.propagate(s_oc, 0, len(m), observe=range(len(m)))
x_oc, y_oc = np.array([[r_oc[i][1].moment0_env[j] 
                            for i in observe_ids] 
                                for j in [0, 2]])
x_oc_all, y_oc_all = np.array([[r_oc[i][1].moment0_env[j] 
                            for i in range(len(m))] 
                                for j in [0, 2]])
pos_oc = np.array([r_oc[i][1].pos for i in observe_ids])
pos_oc_all = np.array([r_oc[i][1].pos for i in range(len(m))])

linex_oc, = ax1.plot(pos_oc, x_oc, 'g-', 
                     alpha=0.9, 
                     label='$\mathrm{corrected\;orbit}$')
linex_oc.set_lw(2)

# setup ax1
ax1.set_xlim([0,160])
ax1.set_title(r"$\mathrm{kick\;of}\;\theta_x = %.3f\;\mathrm{is\;applied\;at\;corrector\;id:}\;%d$" % (corh_val_enabled[0], corh_ids_enabled[0]), fontsize=18)
ax1.set_xlabel('$z\,\mathrm{[m]}$', fontsize=20)
ax1.set_ylabel('$x_{env}\,\mathrm{[mm]}$', fontsize=20)
ax1.legend(loc=3)
#ax1.text(20, 16, 
#         r'$\mathrm{Orbit\;is\;corrected\;back\;by\;applying}\;\theta_x=%.4f$' % (corh_val),
#         fontdict={'fontsize':18})


corr1 = ax1.scatter(pos_oc_all[cor_ids_se], x_oc_all[cor_ids_se],
                   c='m', alpha=0.8, s=100,
                   label=r"$\mathrm{Kicker}$")

np.savetxt('zxy_scipy_3.dat', np.vstack((pos_oc, x_oc, y_oc)).T)
# show
plt.show()

import sys
sys.exit(1)

#corr1 = ax1.scatter(pos[corh_ids_se], x[corh_ids_se], 
#        c='k', alpha=0.8, s=80)

# show with x-rms
xrms_tmp, yrms_tmp = np.array([[r_tmp[i][1].moment0_rms[j] 
                                for i in range(len(observe_ids))] 
                                    for j in [0, 2]])

fig_tmp = plt.figure()
ax_tmp = fig_tmp.add_subplot(111)
linex_tmp, = ax_tmp.plot(pos, x_tmp, 'r', lw=2)
fillx_tmp = ax_tmp.fill_between(pos, x_tmp - xrms_tmp, 
        x_tmp + xrms_tmp, alpha=0.2, color='b')

plt.show()
