# set x correctors
hcors = oc_ins.get_all_cors(type='h')[0:40]

# set initial, lower, upper values for each variables
n_h = len(hcors)
xinit_vals = (np.random.random(size=n_h) - 0.5) * 1.0e-4
xlower_vals = np.ones(n_h) * (-0.01)
xupper_vals = np.ones(n_h) * 0.01
xlbls = ['X{0:03d}'.format(i) for i in range(1, n_h+1)]

# create parameters
plist_x = [genopt.DakotaParam(lbl, val_i, val_l, val_u) 
            for (lbl, val_i, val_l, val_u) in 
                zip(xlbls, xinit_vals, xlower_vals, xupper_vals)]
