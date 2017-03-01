import genopt

# lattice file
latfile = 'test_392.lat'
oc_ins = genopt.DakotaOC(lat_file=latfile)

# select BPMs
bpms = oc_ins.get_elem_by_type('bpm')
oc_ins.set_bpms(bpm=bpms)

# select correctors
hcors = oc_ins.get_all_cors(type='h')[0:40]
vcors = oc_ins.get_all_cors(type='v')[0:40]
oc_ins.set_cors(hcor=hcors, vcor=vcors)

# setup objective function type
oc_ins.ref_flag = "xy"

# setup reference orbit in x and y
bpms_size = len(oc_ins.bpms)
oc_ins.set_ref_x0(np.ones(bpms_size)*0.0)
oc_ins.set_ref_y0(np.ones(bpms_size)*0.0)

# run optimizaiton
oc_ins.simple_run(method='cg', mpi=True, np=4, iternum=30)

# get output
oc_ins.get_orbit(outfile='orbit.dat')

# plot
oc_ins.plot()

