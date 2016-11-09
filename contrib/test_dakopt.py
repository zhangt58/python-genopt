#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" demos to use ``genopt`` package
"""

import os
import numpy as np

import genopt


def demo1():
    """ simple rosenbrock test
    """
    latfile = '../lattice/test.lat'
    oc_ins = genopt.DakotaOC(lat_file=latfile)
    oc_ins.gen_dakota_input(debug=True)
    oc_ins.run(mpi=True, np=4)
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
    #cors = oc_ins.get_all_cors()[45:61]
    cors = oc_ins.get_all_cors()[:]
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

def demo3():
    """ orbit correction test
    
    build variables manually
    """
    lattice_dir = "../lattice"
    latname = 'test_392.lat'
    latfile = os.path.join(lattice_dir, latname)
    oc_ins = genopt.DakotaOC(lat_file=latfile, 
            workdir='./oc_tmp2', keep=True)

    # set BPMs and correctors
    bpms = oc_ins.get_elem_by_type('bpm')
    hcors = oc_ins.get_all_cors(type='h')[0:40]
    vcors = oc_ins.get_all_cors(type='v')[0:40]
    oc_ins.set_bpms(bpm=bpms)
    oc_ins.set_cors(hcor=hcors, vcor=vcors)
    
    # set parameters
    n_h = len(hcors)
    xinit_vals = (np.random.random(size=n_h) - 0.5) * 1.0e-4
    xlower_vals = np.ones(n_h) * (-0.01)
    xupper_vals = np.ones(n_h) * 0.01
    xlbls = ['X{0:03d}'.format(i) for i in range(1, n_h+1)]

    n_v = len(vcors)
    yinit_vals = (np.random.random(size=n_v) - 0.5) * 1.0e-4
    ylower_vals = np.ones(n_v) * (-0.01)
    yupper_vals = np.ones(n_v) * 0.01
    ylbls = ['Y{0:03d}'.format(i) for i in range(1, n_v+1)]
    
    plist_x = [genopt.DakotaParam(lbl, val_i, val_l, val_u) 
            for (lbl, val_i, val_l, val_u) in 
                zip(xlbls, xinit_vals, xlower_vals, xupper_vals)]
    plist_y = [genopt.DakotaParam(lbl, val_i, val_l, val_u) 
            for (lbl, val_i, val_l, val_u) in 
                zip(ylbls, yinit_vals, ylower_vals, yupper_vals)]

    #oc_ins.set_variables(plist=plist_x+plist_y)
    oc_ins.set_variables()
    oc_ins.set_interface()

    #r = genopt.dakutils.DakotaResponses(gradient='numerical',step=2.0e-5)
    oc_ins.set_responses()

    #m = genopt.dakutils.DakotaModel()
    oc_ins.set_model()

    #
    md = genopt.dakutils.DakotaMethod(method='ps', 
            max_function_evaluations=1000)
    oc_ins.set_method(method=md)

    #
    tabfile = os.path.abspath('./dakota1.dat')
    e = genopt.dakutils.DakotaEnviron(tabfile=tabfile)
    oc_ins.set_environ(e)
    
    oc_ins.gen_dakota_input()
    #oc_ins.run(mpi=True, np=4)
    #oc_ins.run()
    #print oc_ins.get_opt_results()
    #oc_ins.get_orbit((oc_ins.hcor, oc_ins.vcor), oc_ins.get_opt_results(), 
    #        outfile='orbit.dat')
    #oc_ins.plot()

def demo4():
    """ orbit correction test
    """
    lattice_dir = "../lattice"
    latname = 'test_392.lat'
    latfile = os.path.join(lattice_dir, latname)
    oc_ins = genopt.DakotaOC(lat_file=latfile, 
                             workdir='./oc_tmp4', 
                             keep=True)

    # set BPMs and correctors
    bpms = oc_ins.get_elem_by_type('bpm')
    hcors = oc_ins.get_all_cors(type='h')[0:40]
    vcors = oc_ins.get_all_cors(type='v')[0:40]
    oc_ins.set_bpms(bpm=bpms)
    oc_ins.set_cors(hcor=hcors, vcor=vcors)
    
    # set parameters
    oc_ins.set_variables()

    # set interface
    oc_ins.set_interface()

    # set responses
    #r = genopt.dakutils.DakotaResponses(gradient='numerical',step=2.0e-5)
    r = genopt.dakutils.DakotaResponses()
    oc_ins.set_responses(r)

    # set model
    #m = genopt.dakutils.DakotaModel()
    oc_ins.set_model()

    # set method
    md = genopt.dakutils.DakotaMethod(method='ps', 
            max_function_evaluations=1000)
    oc_ins.set_method(method=md)

    # set environment
    tabfile = os.path.abspath('./oc_tmp4/dakota1.dat')
    e = genopt.dakutils.DakotaEnviron(tabfile=tabfile)
    oc_ins.set_environ(e)
    
    # generate input
    oc_ins.gen_dakota_input()

    # run
    oc_ins.run(mpi=True, np=4)
    #print oc_ins.get_opt_results()
    
    # get output
    oc_ins.get_orbit((oc_ins.hcor, oc_ins.vcor), oc_ins.get_opt_results(), 
                      outfile='orbit.dat')

    # plot
    #oc_ins.plot()

def demo5():
    """ orbit correction test
    """
    lattice_dir = "../lattice"
    latname = 'test_392.lat'
    latfile = os.path.join(lattice_dir, latname)
    oc_ins = genopt.DakotaOC(lat_file=latfile, 
                             workdir='./oc_tmp6', 
                             keep=True)

    # set BPMs and correctors
    bpms = oc_ins.get_elem_by_type('bpm')
    hcors = oc_ins.get_all_cors(type='h')[0:40]
    vcors = oc_ins.get_all_cors(type='v')[0:40]
    oc_ins.set_bpms(bpm=bpms)
    oc_ins.set_cors(hcor=hcors, vcor=vcors)
    
    oc_ins.simple_run(method='cg', mpi=True, np=4, iternum=10)
    
    # get output
    oc_ins.get_orbit(outfile='orbit1.dat')
    # plot
    #oc_ins.plot()

def demo6():
    """ orbit correction test
    """
    lattice_dir = "../lattice"
    latname = 'test_392.lat'
    latfile = os.path.join(lattice_dir, latname)
    oc_ins = genopt.DakotaOC(lat_file=latfile, 
                             workdir='./oc_tmp7', 
                             keep=True)

    hcors = oc_ins.get_all_cors(type='h')[0:40]
    vcors = oc_ins.get_all_cors(type='v')[0:40]
    oc_ins.set_cors(hcor=hcors, vcor=vcors)

    oc_ins.ref_flag = "xy"
    bpms_size = len(oc_ins.bpms)
    oc_ins.set_ref_x0(np.ones(bpms_size)*0.0)
    oc_ins.set_ref_y0(np.ones(bpms_size)*0.0)
    oc_ins.simple_run(method='cg', mpi=True, np=4, iternum=30, evalnum=2000)
    
    # get output
    oc_ins.get_orbit(outfile='orbit_x0y0_cg.dat')
    # plot
    oc_ins.plot()

def demo7():
    """ orbit correction test
    """
    lattice_dir = "../lattice"
    latname = 'test_392.lat'
    latfile = os.path.join(lattice_dir, latname)
    oc_ins = genopt.DakotaOC(lat_file=latfile, 
                             workdir='./oc_tmp7', 
                             keep=True)

    oc_ins.simple_run(method='cg', mpi=True, np=4, iternum=20, evalnum=2000)
    
    # get output
    oc_ins.get_orbit(outfile='orbit7.dat')
    # plot
    oc_ins.plot()


if __name__ == '__main__':
    #demo1()
    #demo2()
    #demo3()
    #demo4()
    #demo5()
    #demo6()
    demo7()
