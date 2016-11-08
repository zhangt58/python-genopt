from genopt import DakotaOC

latfile = '../lattice/test_392.lat'
oc_ins = DakotaOC(lat_file=latfile)
bpms = oc_ins.get_elem_by_type('bpm')
hcors = oc_ins.get_all_cors(type='h')
vcors = oc_ins.get_all_cors(type='v')
oc_ins.set_bpms(bpm=bpms)
oc_ins.set_cors(hcor=hcors, vcor=vcors)

#oc_ins.set_variables()
#oc_ins.gen_dakota_input()

oc_ins.plot(outfile='./oc_tmp/dakota.out')
