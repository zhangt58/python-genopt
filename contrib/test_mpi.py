#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" dakota mpi test

Tong Zhang <zhangt@frib.msu.edu>
2016-10-24 14:47:12 PM EDT
"""

import dakopt

def test_dakotaoc_mpi():
    latfile = 'test/test.lat'
    oc_ins = dakopt.DakotaOC(lat_file=latfile)
    #bpms = oc_ins.get_elem_by_type('bpm')
    #cors = oc_ins.get_all_cors()[1:10]
    oc_ins.gen_dakota_input()
    oc_ins.run(mpi=True, np=4)


if __name__ == '__main__':
    test_dakotaoc_mpi()

