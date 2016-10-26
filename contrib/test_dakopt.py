#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" demos to use ``genopt`` package
"""

import os

import genopt


def demo1():
    """ simple rosenbrock test
    """
    latfile = '../lattice/test.lat'
    oc_ins = genopt.DakotaOC(lat_file=latfile)
    oc_ins.gen_dakota_input(debug=True)
    oc_ins.run(mpi=True, np=2)
    #oc_ins.run()
    print oc_ins.get_opt_results()

def demo2():
    """ orbit correction test
    """
    lattice_dir = "../lattice"
    latname = 'test_392.lat'
    latfile = os.path.join(lattice_dir, latname)
    oc_ins = genopt.DakotaOC(lat_file=latfile)
    #names = ('LS1_CA01:BPM_D1144', 'LS1_WA01:BPM_D1155')
    #names = 'LS1_CA01:BPM_D1144'
    #names = 'LS1_CB06:DCH_D1574'
    #idx = oc_ins.get_elem_by_name(names)
    #print idx

    # set BPMs and correctors
    bpms = oc_ins.get_elem_by_type('bpm')
    cors = oc_ins.get_all_cors()[45:61]
    #cors = oc_ins.get_all_cors()[34:50]
    #print oc_ins.get_all_cors(type='v')
    oc_ins.set_bpms(bpm=bpms)
    oc_ins.set_cors(cor=cors)
    
    # set parameters
    oc_ins.set_variables()
    
    oc_ins.gen_dakota_input()
    oc_ins.run(mpi=True, np=4)
    #oc_ins.run()
    #print oc_ins.get_opt_results()
    oc_ins.get_orbit((oc_ins.hcor, oc_ins.vcor), oc_ins.get_opt_results(), 
            outfile='orbit.dat')
    oc_ins.plot()


if __name__ == '__main__':
    #demo1()
    demo2()
