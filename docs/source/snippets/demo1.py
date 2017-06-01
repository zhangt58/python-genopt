import genopt

latfile = 'test_392.lat'
oc_ins = genopt.DakotaOC(lat_file=latfile)

oc_ins.simple_run(method='cg', mpi=True, np=4, iternum=20)

# get output
oc_ins.get_orbit(outfile='orbit.dat')

# plot
oc_ins.plot()
