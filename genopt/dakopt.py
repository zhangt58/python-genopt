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
import tempfile

from flame import Machine
import dakutils
from dakutils import generate_latfile
#from flamtutils import generate_latfile


class DakotaBase(object):
    """ Base class for general optimization, initialized parameters:
        valid keyword parameters:

        * workdir: root dir for dakota input/output files,
          the defualt one should be created in /tmp, or define some dir path
        * dakexec: full path of dakota executable,
          the default one should be *dakota*, or define the full path
        * dakhead: prefixed name for input/output files of *dakota*, 
          the default one is *dakota*
        * keep: if keep the working directory (i.e. defined by *workdir*), default is False
    """

    def __init__(self, **kws):
        # workdir
        wdir = kws.get('workdir')
        if wdir is None:
            #self._workdir = os.path.join('/tmp', 'dakota_' + dakutils.random_string(6))
            self._workdir = tempfile.mkdtemp(prefix='dakota_')
        else:
            self._workdir = wdir
            if not os.path.isdir(wdir):
                os.makedirs(wdir)

        # keep working data?
        keepflag = kws.get('keep')
        if keepflag is not None and keepflag:
            self._keep = True
        else:
            self._keep = False

        # dakexec
        if kws.get('dakexec') is not None:
            self._dakexec = kws.get('dakexec')
        else:
            self._dakexec = 'dakota'

        # dakhead
        if kws.get('dakhead') is not None:
            self._dakhead = kws.get('dakhead')
        else:
            self._dakhead = 'dakota'

        self._dakin = None
        self._dakout = None

    @property
    def dakexec(self):
        return self._dakexec

    @dakexec.setter
    def dakexec(self, dakexec):
        self._dakexec = dakexec

    @property
    def workdir(self):
        return self._workdir

    @workdir.setter
    def workdir(self, wdir):
        self._workdir = wdir

    @property
    def dakhead(self):
        return self._dakhead

    @dakhead.setter
    def dakhead(self, nprefix):
        self._dakhead = nprefix

    @property
    def keep(self):
        return self._keep

    @keep.setter
    def keep(self, f):
        self._keep = f

    def __del__(self):
        if not self._keep:
            try:
                rmtree(self._workdir)
            except:
                pass
        else:
            print("work files are kept in: %s" % (self._workdir))
        

class DakotaOC(DakotaBase):
    """ Dakota optimization class with orbit correction driver

    :param lat_file: lattice file
    :param elem_bpm: list of element indice of BPMs
    :param elem_cor: list of element indice of correctors, always folders of 2
    :param elem_hcor: list of element indice of horizontal correctors
    :param elem_vcor: list of element indice of vertical correctors
    :param ref_x0: reference orbit in x, list of BPM readings
    :param ref_y0: reference orbit in y, list of BPM readings
    :param ref_flag: string flag for objective functions:

           1. "x": :math:`\sum \Delta x^2`, :math:`\Delta x = x-x_0`;
           2. "y": :math:`\sum \Delta y^2`, :math:`\Delta y = y-y_0`;
           3. "xy": :math:`\sum \Delta x^2 + \sum \Delta y^2`.

    :param model: simulation model, 'flame' or 'impact'
    :param optdriver: analysis driver for optimization, 'flamedriver_oc' by default
    :param kws: keywords parameters for additional usage, defined in ``DakotaBase`` class
                valid keys:
               * *workdir*: root dir for dakota input/output files,
                 the defualt one should be created in /tmp, or define some dir path
               * *dakexec*: full path of dakota executable,
                 the default one should be *dakota*, or define the full path
               * *dakhead*: prefixed name for input/output files of *dakota*, 
                 the default one is *dakota*
               * *keep*: if keep the working directory (i.e. defined by *workdir*), 
                 default is False
    """

    def __init__(self,
                 lat_file=None,
                 elem_bpm=None,
                 elem_cor=None,
                 elem_hcor=None,
                 elem_vcor=None,
                 ref_x0=None,
                 ref_y0=None,
                 ref_flag=None,
                 model=None,
                 optdriver=None,
                 **kws):
        super(self.__class__, self).__init__(**kws)

        if lat_file is not None:
            self._lat_file = os.path.realpath(os.path.expanduser(lat_file))
        else:  # use example lattice file
            pass

        self._elem_bpm = elem_bpm
        self._elem_hcor, self._elem_vcor = elem_hcor, elem_vcor
        if elem_cor is not None:
            self._elem_hcor, self._elem_vcor = elem_cor[0::2], elem_cor[1::2]
        elif elem_hcor is not None:
            self._elem_hcor = elem_hcor
        elif elem_vcor is not None:
            self._elem_vcor = elem_vcor

        self._ref_x0 = ref_x0
        self._ref_y0 = ref_y0
        self._ref_flag = "xy" if ref_flag is None else ref_flag

        if model is None:
            self._model = 'FLAME'
        else:
            self._model = model.upper()

        if optdriver is None:
            self._opt_driver = 'flamedriver_oc'

        self.set_model()
        self.create_machine(self._lat_file)
        self.set_bpms(self._elem_bpm)
        self.set_cors(self._elem_hcor, self._elem_vcor)
        self.set_ref_x0(self._ref_x0)
        self.set_ref_y0(self._ref_y0)

    @property
    def hcors(self):
        return self._elem_hcor

    @property
    def vcors(self):
        return self._elem_vcor

    @property
    def latfile(self):
        return self._lat_file

    @property
    def ref_x0(self):
        return self._ref_x0

    @property
    def ref_y0(self):
        return self._ref_y0

    @property
    def ref_flag(sef):
        return self._ref_flag

    @ref_flag.setter
    def ref_flag(self, s):
        self._ref_flag = s

    @property
    def bpms(self):
        return self._elem_bpm

    @latfile.setter
    def latfile(self, latfile):
        self._lat_file = latfile

    @property
    def optdriver(self):
        return self._opt_driver

    @optdriver.setter
    def optdriver(self, driver):
        self._opt_driver = driver

    def get_machine(self):
        """ get flame machine object for potential usage

        :return: flame machine object or None
        """
        try:
            return self._machine
        except:
            return None
    
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

    def set_ref_x0(self, ref_arr=None):
        """ set reference orbit in x, if not set, use 0s

        :param ref_arr: array of reference orbit values
                        size should be the same number as selected BPMs
        """
        if ref_arr is None:
            self._ref_x0 = [0]*len(self._elem_bpm)
        else:
            self._ref_x0 = ref_arr

    def set_ref_y0(self, ref_arr=None):
        """ set reference orbit in y, if not set, use 0s

        :param ref_arr: array of reference orbit values
                        size should be the same number as selected BPMs
        """
        if ref_arr is None:
            self._ref_y0 = [0]*len(self._elem_bpm)
        else:
            self._ref_y0 = ref_arr
    
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

    def gen_dakota_input(self, infile=None, debug=False):
        """ generate dakota input file

        :param infile: dakota input filename
        :param debug: if True, generate a simple test input file
        """
        if not debug:
            dakinp = dakutils.DakotaInput()
            #dakinp.set_template(name='oc')
            dakinp.interface = self._oc_interface
            dakinp.variables = self._oc_variables
            dakinp.model = self._oc_model
            dakinp.responses = self._oc_responses
            dakinp.method = self._oc_method
            dakinp.environment = self._oc_environ
        else:  # debug is True
            dakinp = dakutils.DakotaInput()

        if infile is None:
            infile = self._dakhead + '.in'
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
        else:  # plist = [p1, p2, ...]
            n = len(plist)
            initial_point_string = ' '.join(["{0:>14e}".format(p.initial) for p in plist])
            lower_bounds_string = ' '.join(["{0:>14e}".format(p.lower) for p in plist])
            upper_bounds_string = ' '.join(["{0:>14e}".format(p.upper) for p in plist])
            descriptors_string = ' '.join(["{0:>14s}".format(p.label) for p in plist])
            oc_variables = []
            oc_variables.append('continuous_design = {0}'.format(n))
            oc_variables.append('  initial_point' + initial_point_string)
            oc_variables.append('  lower_bounds ' + lower_bounds_string)
            oc_variables.append('  upper_bounds ' + upper_bounds_string)
            oc_variables.append('  descriptors  ' + descriptors_string)
            self._oc_variables = oc_variables

    def set_interface(self, interface=None, **kws):
        """ setup interface block, that is setup ``oc_interface``
        should be ready to invoke after ``set_cors`` and ``set_bpms``

        :param interface: ``DakotaInterface`` object, automatically setup if not defined
        """
        if interface is None:
            oc_interface = dakutils.DakotaInterface(mode='fork', latfile=self._lat_file,
                                                    driver='flamedriver_oc',
                                                    bpms=self._elem_bpm,
                                                    hcors=self._elem_hcor,
                                                    vcors=self._elem_vcor,
                                                    ref_x0=self._ref_x0,
                                                    ref_y0=self._ref_y0,
                                                    ref_flag=self._ref_flag,
                                                    deactivate='active_set_vector')
        else:
            oc_interface = interface
        self._oc_interface = oc_interface.get_config()

    def set_model(self, model=None, **kws):
        """ setup model block, that is setup ``oc_model``

        :param model: ``DakotaModel`` object, automatically setup if not defined
        """
        if model is None:
            oc_model = dakutils.DakotaModel()
        else:
            oc_model = model
        self._oc_model = oc_model.get_config()

    def set_responses(self, responses=None, **kws):
        """ setup responses block, that is setup ``oc_responses``

        :param responses: ``DakotaResponses`` object, automatically setup if not defined
        """
        if responses is None:
            oc_responses = dakutils.DakotaResponses(gradient='numerical')
        else:
            oc_responses = responses
        self._oc_responses = oc_responses.get_config()

    def set_environ(self, environ=None):
        """ setup environment block, that is setup ``oc_environ``

        :param environ: ``DakotaEnviron`` object, automatically setup if not defined
        """
        if environ is None:
            oc_environ = dakutils.DakotaEnviron(tabfile='dakota.dat')
        else:
            oc_environ = environ
        self._oc_environ= oc_environ.get_config()

    def set_method(self, method=None):
        """ setup method block, that is setup ``oc_method``

        :param method: ``DakotaMethod`` object, automatically setup if not defined
        """
        if method is None:
            oc_method = dakutils.DakotaMethod(method='cg')
        else:
            oc_method = method
        self._oc_method = oc_method.get_config()

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

    def get_opt_results(self, outfile=None, rtype='dict', label='plain'):
        """ extract optimized results from dakota output

        :param outfile: file name of dakota output file, 
            'dakota.out' by default
        :param rtype: type of returned results, 'dict' or 'list', 
            'dict' by default
        :param label: label types for returned variables, only valid when rtype 'dict', 
            'plain' by default:

                * *'plain'*: variable labels with format of ``x1``, ``x2``, ``y1``, ``y2``, etc.
                  e.g. ``{'x1': v1, 'y1': v2}``
                * *'fancy'*: variable labels with the name defined in lattice file,
                  e.g. ``'LS1_CA01:DCH_D1131'``, dict returned sample: 
                  ``{'LS1_CA01:DCH_D1131': {'id':9, 'config':{'theta_x':v1}}}``

        .. note:: The ``fancy`` option will make re-configuring flame machine in a more 
            convenient way, such as:

            >>> opt_cors = get_opt_results(label='fancy')
            >>> for k,v in opt_cors.items():
            >>>     m.reconfigure(v['id'], v['config'])
            >>> # here m is an instance of flame.Machine class
            >>> 

        :return: by default return a dict of optimized results with each item
            of the format like "x1":0.1 or more fancy format by set label with 'fancy', etc.,
            if rtype='list', return a list of values, when the keys are ascend sorted.

        :Example:
        
        >>> opt_vars = get_optresults(outfile='flame_oc.out', rtype='dict'):
        >>> print(opt_vars)
        {'x2': 0.0020782814353, 'x1': -0.0017913264033}
        >>> opt_vars = get_optresults(outfile='flame_oc.out', rtype='list'):
        >>> print(opt_vars)
        [-0.0017913264033, 0.0020782814353]
        """
        if outfile is None:
            outfile=self._dakout
        if rtype == 'list':
            return dakutils.get_opt_results(outfile=outfile, rtype=rtype)
        else:
            rdict = dakutils.get_opt_results(outfile=outfile, rtype=rtype)
            if label == 'plain':
                return rdict
            else:  # label = 'fancy'
                val_x = [v for (k,v) in sorted(rdict.items()) if k.startswith('x')]
                val_y = [v for (k,v) in sorted(rdict.items()) if k.startswith('y')]
                vx = [{'id': i, 'config':{'theta_x': v}} for i,v in zip(self._elem_hcor, val_x)]
                vy = [{'id': i, 'config':{'theta_y': v}} for i,v in zip(self._elem_vcor, val_y)]
                kx = [self._machine.conf(i)['name'] for i in self._elem_hcor]
                ky = [self._machine.conf(i)['name'] for i in self._elem_vcor]
                return dict(zip(kx+ky, vx+vy))

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
        zpos, x, y, mtmp = self.get_orbit((idx_h, idx_v), opt_vars)

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
        :return: tuple of zpos, env_x, env_y, machine
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

        return zpos, x, y, m

    def simple_run(self, method='cg', mpi=None, np=None, **kws):
        """ run optimization after ``set_bpms()`` and ``set_cors()``,
        by using default configuration and make full use of computing resources.

        :param method: optimization method, 'cg', 'ps', 'cg' by default
        :param mpi: if True, run DAKOTA in parallel mode, False by default
        :param np: number of processes to use, only valid when ``mpi`` is True 
        :param kws: keyword parameters
            valid keys:
                * step: gradient step, 1e-6 by default
                * iternum: max iteration number, 20 by default
                * evalnum: max function evaulation number, 1000 by default
        """
        if method == 'cg':
            max_iter_num = kws.get('iternum', 20)
            step = kws.get('step', 1e-6)
            md = dakutils.DakotaMethod(method='cg', 
                                       max_iterations=max_iter_num)
            self.set_method(method=md)
            re = dakutils.DakotaResponses(gradient='numerical', step=step)
            self.set_responses(responses=re)
        else: # 'ps'
            max_eval_num = kws.get('evalnum', 1000)
            md = dakutils.DakotaMethod(method='ps', 
                                       max_function_evaluations=max_eval_num)
            self.set_method(method=md)
            re = dakutils.DakotaResponses()
            self.set_responses(responses=re)

        self.set_environ()
        self.set_model()
        self.set_variables()
        self.set_interface()
        self.gen_dakota_input()
        if mpi:
            max_core_num = multiprocessing.cpu_count()
            if np is None or int(np) > max_core_num:
                np = max_core_num
            self.run(mpi=mpi, np=np)
        else:
            self.run()

    def get_opt_latfile(self, outfile='out.lat'):
        """ get optimized lattice file for potential next usage,
        ``run()`` or ``simple_run()`` should be evoked first to get the 
        optimized results.
        
        :param outfile: file name for generated lattice file
        :return: lattice file name or None if failed
        """
        try:
            z, x, y, m = self.get_orbit()
            rfile = generate_latfile(m, latfile=outfile)
        except:
            print("Failed to generate latfile.")
            rfile = None
        finally:
            return rfile


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
