#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" General optimization module by utilizing DAKOTA

* orbit correction: ``DakotaOC``

Tong Zhang <zhangt@frib.msu.edu>

2016-10-23 14:26:13 PM EDT
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import multiprocessing
import subprocess
import time
from shutil import rmtree

from flame import Machine
import dakutils


class DakotaBase(object):
    """ Base class for general optimization, initialized parameters:

    """

    def __init__(self, **kws):
        self._dakexec = 'dakota'
        self._dakout = 'dakota.out'

    @property
    def dakexec(self):
        return self._dakexec

    @dakexec.setter
    def dakexec(self, dakexec):
        self._dakexec = dakexec


class DakotaOC(DakotaBase):
    """ Dakota optimization class with orbit correction driver

    :param lat_file: lattice file
    :param elem_bpm: list of element indice of BPMs
    :param elem_cor: list of element indice of correctors, always folders of 2
    :param elem_hcor: list of element indice of horizontal correctors
    :param elem_vcor: list of element indice of vertical correctors
    :param model: simulation model, 'flame' or 'impact'
    :param optdriver: analysis driver for optimization, 'flamedriver' by default
    :param workdir: directory that dakota input/output files should be in
    :param kws: keywords parameters for additional usage, defined in ``DakotaBase`` class
                valid keys:
                
                    * dakexec: full path of dakota executable
    """

    def __init__(self,
                 lat_file=None,
                 elem_bpm=None,
                 elem_cor=None,
                 elem_hcor=None,
                 elem_vcor=None,
                 model=None,
                 optdriver=None,
                 workdir=None,
                 **kws):
        super(self.__class__, self).__init__(**kws)

        if lat_file is not None:
            self._lat_file = os.path.realpath(os.path.expanduser(lat_file))
        else:  # user example lattice file
            pass

        self._elem_bpm = elem_bpm
        self._elem_hcor, self._elem_vcor = elem_hcor, elem_vcor
        if elem_cor is not None:
            self._elem_hcor, self._elem_vcor = elem_cor[0::2], elem_cor[1::2]
        elif elem_hcor is not None:
            self._elem_hcor = elem_hcor
        elif elem_vcor is not None:
            self._elem_vcor = elem_vcor

        if model is None:
            self._model = 'FLAME'
        else:
            self._model = model.upper()

        if optdriver is None:
            self._opt_driver = 'flamedriver'

        if workdir is None:
            self._workdir = os.path.join('/tmp',
                                         'dakota_' + dakutils.random_string(6))

        if kws.get('dakexec') is not None:
            self._dakexec = kws.get('dakexec')

        self.set_model()
        self.create_machine(self._lat_file)
        self.set_bpms(self._elem_bpm)
        self.set_cors(self._elem_hcor, self._elem_vcor)

    @property
    def hcor(self):
        return self._elem_hcor

    @property
    def vcor(self):
        return self._elem_vcor

    @property
    def latfile(self):
        return self._lat_file

    @latfile.setter
    def latfile(self, latfile):
        self._lat_file = latfile

    @property
    def optdriver(self):
        return self._opt_driver

    @optdriver.setter
    def optdriver(self, driver):
        self._opt_driver = driver

    def __del__(self):
        pass
        #try:
        #    rmtree(self._workdir)
        #except:
        #    pass

    def create_machine(self, lat_file):
        """ create machine instance with model configuration
        
        * setup _machine
        * setup _elem_bpm, _elem_cor or (_elem_hcor and _elem_vcor)
        """
        if self._model == "FLAME":
            self._create_flame_machine(lat_file)
        elif self._model == "IMPACT":
            self._create_impact_machine(lat_file)

    def _create_flame_machine(self, lat_file):
        try:
            self._machine = Machine(open(lat_file, 'r'))
        except IOError as e:
            print("Failed to open {fn}: {info}".format(fn=e.filename,
                                                       info=e.args[-1]))
            sys.exit(1)
        except (RuntimeError, KeyError) as e:
            print("Cannot parse lattice, " + e.args[-1])
            sys.exit(1)
        except:
            print("Failed to create machine")
            sys.exit(1)

    def _create_impact_machine(self, lat_file):
        pass

    def set_bpms(self, bpm=None):
        """ set BPMs

        :param bpm: list of bpm indices, if None, use all BPMs
        """
        if bpm is None:
            self._elem_bpm = self.get_all_bpms()
        else:
            self._elem_bpm = bpm

    def set_cors(self, cor=None, hcor=None, vcor=None):
        """ set correctors, if cor, hcor and vcor are None, use all correctors
        if cor is not None, use cor, ignore hcor and vcor

        :param cor: list of corrector indices, hcor, vcor,...
        :param hcor: list of horizontal corrector indices
        :param vcor: list of vertical corrector indices
        """
        if cor is not None:
            self._elem_hcor = cor[0::2]
            self._elem_vcor = cor[1::2]
        else:
            if hcor is None and vcor is None:
                self._elem_hcor = self.get_all_cors(type='h')
                self._elem_vcor = self.get_all_cors(type='v')
            else:
                if hcor is not None:
                    self._elem_hcor = hcor
                if vcor is not None:
                    self._elem_vcor = vcor

    def set_model(self, **kws):
        """ configure model

        :param kws: only for impact, available keys:
            "execpath": path of impact executable
        """
        if self._model == 'flame':
            pass  # nothing more needs to do if model is 'flame'
        else:  # self._model is 'impact'
            execpath = kws.get('execpath')
            if execpath is not None:
                self._impexec = os.path.real(os.path.expanduser(execpath))
            else:  # just use impact as the default name
                self._impexec = "impact"

    def get_all_bpms(self):
        """ get list of all valid bpms indices

        :return: a list of bpm indices
        
        :Example:
        
        >>> dakoc = DakotaOC('test/test.lat')
        >>> print(dakoc.get_all_bpms())
        """
        return self.get_elem_by_type(type='bpm')

    def get_all_cors(self, type=None):
        """ get list of all valid correctors indices
        
        :param type: define corrector type, 'h': horizontal, 'v': vertical, 
            if not defined, return all correctors
        :return: a list of corrector indices

        :Example:

        >>> dakoc = DakotaOC('test/test.lat')
        >>> print(dakoc.get_all_cors())
        """
        all_cors = self.get_elem_by_type(type='orbtrim')
        if type is None:
            return all_cors
        elif type == 'h':
            return all_cors[0::2]
        elif type == 'v':
            return all_cors[1::2]
        else:
            print("warning: unrecongnized corrector type.")
            return all_cors

    def get_elem_by_name(self, name):
        """ get list of element(s) by name(s)

        :param name: tuple or list of name(s)
        :return: list of element indices

        :Example:

        >>> dakoc = DakotaOC('test/test.lat')
        >>> names = ('LS1_CA01:BPM_D1144', 'LS1_WA01:BPM_D1155')
        >>> idx = dakoc.get_elem_by_name(names)
        >>> print(idx)
        [18, 31]

        """
        if isinstance(name, str):
            name = (name, )
        retval = [self._machine.find(name=n)[0] for n in name]
        return retval

    def get_elem_by_type(self, type):
        """ get list of element(s) by type

        :param type: string name of element type
        :return: list of element indices

        :Example:

        >>> dakoc = DakotaOC('test/test.lat')
        >>> type = 'bpm'
        >>> idx = dakoc.get_elem_by_type(type)
        >>> print(idx)

        """
        retval = self._machine.find(type=type)
        return retval

    def gen_dakota_input(self, infile='dakota.in', debug=False):
        """ generate dakota input file

        :param infile: dakota input filename
        :param debug: if True, generate a simple test input file
        """
        if not debug:
            bpms = "'" + ' '.join(
                ['{0}'.format(i) for i in self._elem_bpm]) + "'"
            hcors = "'" + ' '.join(
                ['{0}'.format(i) for i in self._elem_hcor]) + "'"
            vcors = "'" + ' '.join(
                ['{0}'.format(i) for i in self._elem_vcor]) + "'"

            oc_interface = []
            oc_interface.append('fork')
            oc_interface.append(
                'analysis_driver = "{driver} {latfile} {bpms} {hcors} {vcors}"'.format(
                    driver=self.optdriver,
                    latfile=self.latfile,
                    bpms=bpms,
                    hcors=hcors,
                    vcors=vcors, ))
            oc_interface.append('deactivate = active_set_vector')

            dakinp = dakutils.DakotaInput()
            dakinp.set_template(name='oc')
            dakinp.interface = oc_interface
            dakinp.variables = self._oc_variables
        else:  # debug is True
            dakinp = dakutils.DakotaInput()

        if not os.path.isdir(self._workdir):
            os.mkdir(self._workdir)
        inputfile = os.path.join(self._workdir, infile)
        outputfile = inputfile.replace('.in', '.out')
        self._dakin = inputfile
        self._dakout = outputfile
        dakinp.write(inputfile)

    def set_variables(self, plist=None, initial=1e-4, lower=-0.01, upper=0.01):
        """ setup variables block, that is setup ``oc_variables``
        should be ready to invoke after ``set_cors()``

        :param plist: list of defined parameters (``DakotaParam`` object), 
            automatically setup if not defined
        :param initial: initial values for all variables, only valid when plist is None
        :param lower: lower bound for all variables, only valid when plist is None
        :param upper: upper bound for all variables, only valid when plist is None
        """
        if plist is None:
            if self._elem_hcor is None and self._elem_vcor is None:
                print("No corrector is selected, set_cors() first.")
                sys.exit(1)
            else:
                x_len = len(
                    self._elem_hcor) if self._elem_hcor is not None else 0
                y_len = len(
                    self._elem_vcor) if self._elem_vcor is not None else 0
                n = x_len + y_len
                oc_variables = []
                oc_variables.append('continuous_design = {0}'.format(n))
                oc_variables.append('  initial_point' + "{0:>14e}".format(
                    initial) * n)
                oc_variables.append('  lower_bounds ' + "{0:>14e}".format(
                    lower) * n)
                oc_variables.append('  upper_bounds ' + "{0:>14e}".format(
                    upper) * n)
                xlbls = ["'x{0:03d}'".format(i) for i in range(1, x_len + 1)]
                ylbls = ["'y{0:03d}'".format(i) for i in range(1, y_len + 1)]
                oc_variables.append('  descriptors  ' + ''.join(
                    ["{0:>14s}".format(lbl) for lbl in xlbls + ylbls]))
                self._oc_variables = oc_variables

    def run(self, mpi=False, np=None):
        """ run optimization

        :param mpi: if True, run DAKOTA in parallel mode, False by default
        :param np: number of processes to use, only valid when ``mpi`` is True 
        """
        if mpi:
            max_core_num = multiprocessing.cpu_count()
            if np is None or int(np) > max_core_num:
                np = max_core_num
            run_command = "mpirun -np {np} {dakexec} -i {dakin} -o {dakout}".format(
                np=np,
                dakexec=self._dakexec,
                dakin=self._dakin,
                dakout=self._dakout)
        else:  # mpi is False
            run_command = "{dakexec} -i {dakin} -o {dakout}".format(
                dakexec=self._dakexec,
                dakin=self._dakin,
                dakout=self._dakout)
        subprocess.call(run_command.split())

    def get_opt_results(self, outfile=None, rtype='dict'):
        """ extract optimized results from dakota output

        :param outfile: file name of dakota output file, 
            'dakota.out' by default
        :param rtype: type of returned results, 'dict' or 'list', 
            'dict' by default
        :return: by default return a dict of optimized results with each item
            of the format like "x1":0.1, etc., or if rtype='list', return a 
            list of values, when the keys are ascend sorted.

        :Example:
        
        >>> opt_vars = get_optresults(outfile='flame_oc.out', rtype='dict'):
        >>> print(opt_vars)
        {'x2': 0.0020782814353, 'x1': -0.0017913264033}
        >>> opt_vars = get_optresults(outfile='flame_oc.out', rtype='list'):
        >>> print(opt_vars)
        [-0.0017913264033, 0.0020782814353]
        """
        if outfile is None:
            return dakutils.get_opt_results(outfile=self._dakout, rtype=rtype)
        else:
            return dakutils.get_opt_results(outfile=outfile, rtype=rtype)

    def plot(self, outfile=None, figsize=(10, 8), dpi=120, **kws):
        """ show orbit

        :param outfile: output file of dakota
        :param figsize: figure size, (h, w)
        :param dpi: figure dpi
        """
        if outfile is None:
            opt_vars = self.get_opt_results()
        else:
            opt_vars = self.get_opt_results(outfile=outfile)

        idx_h, idx_v = self._elem_hcor, self._elem_vcor
        zpos, x, y = self.get_orbit((idx_h, idx_v), opt_vars)

        fig = plt.figure(figsize=figsize, dpi=dpi, **kws)
        ax = fig.add_subplot(111)
        linex, = ax.plot(zpos, x, 'r-', label='$x$', lw=2)
        liney, = ax.plot(zpos, y, 'b-', label='$y$', lw=2)
        ax.set_xlabel('$z\,\mathrm{[m]}$', fontsize=20)
        ax.set_ylabel('$\mathrm{Orbit\;[mm]}$', fontsize=20)
        ax.legend(loc=3)

        plt.show()

    def get_orbit(self, idx=None, val=None, outfile=None):
        """ calculate the orbit with given configurations

        :param idx: (idx_hcor, idx_vcor), tuple of list of indices of h/v cors
        :param val: values for each correctos, h/v
        :param outfile: filename to save the data
        """
        if idx is None:
            idx_x, idx_y = self._elem_hcor, self._elem_vcor
        else:
            idx_x, idx_y = idx
        if val is None:
            val = self.get_opt_results()
        else:
            val = val

        m = self._machine
        val_x = [v for (k, v) in sorted(val.items()) if k.startswith('x')]
        val_y = [v for (k, v) in sorted(val.items()) if k.startswith('y')]
        [m.reconfigure(eid, {'theta_x': eval})
         for (eid, eval) in zip(idx_x, val_x)]
        [m.reconfigure(eid, {'theta_y': eval})
         for (eid, eval) in zip(idx_y, val_y)]
        s = m.allocState({})
        r = m.propagate(s, 0, len(m), observe=range(len(m)))
        zpos = np.array([r[i][1].pos for i in self._elem_bpm])
        x, y = np.array(
            [[r[i][1].moment0_env[j] for i in self._elem_bpm] for j in [0, 2]])

        if outfile is not None:
            np.savetxt(outfile,
                       np.vstack((zpos, x, y)).T,
                       fmt="%22.14e",
                       comments='# orbit data saved at ' + time.ctime() + '\n',
                       header="#{0:^22s} {1:^22s} {2:^22s}".format(
                           "zpos [m]", "x [mm]", "y [mm]"),
                       delimiter=' ')

        return zpos, x, y


def test_dakotaoc1():
    latfile = 'test/test.lat'
    oc_ins = DakotaOC(lat_file=latfile)
    oc_ins.gen_dakota_input(debug=True)
    #oc_ins.run(mpi=True, np=2)
    oc_ins.run()
    print oc_ins.get_opt_results()


def test_dakotaoc2():
    latfile = 'test_392.lat'
    oc_ins = DakotaOC(lat_file=latfile)
    #names = ('LS1_CA01:BPM_D1144', 'LS1_WA01:BPM_D1155')
    #names = 'LS1_CA01:BPM_D1144'
    names = 'LS1_CB06:DCH_D1574'
    idx = oc_ins.get_elem_by_name(names)
    print idx

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
    oc_ins.plot()
    #print oc_ins.get_opt_results()


if __name__ == '__main__':
    #test_dakotaoc1()
    test_dakotaoc2()
