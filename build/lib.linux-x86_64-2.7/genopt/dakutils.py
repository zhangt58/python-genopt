#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" module contains utilities:

* generate dakota input files
* extract data from output files

Tong Zhang <zhangt@frib.msu.edu>

2016-10-17 09:19:25 AM EDT
"""

import os
import sys
import random
import string
import numpy as np
from numpy import ndarray


class DakotaInput(object):
    """ template of dakota input file, field could be overriden by
    providing additional keyword arguments, 
    
    :param kws: keyword arguments, valid keys are dakota directives

    :Example:
    
    >>> dak_inp = DakotaInput(method=["max_iterations = 500",
                                      "convergence_tolerance = 1e-7",
                                      "conmin_frcg",])
    >>> 
    """

    def __init__(self, **kws):
        self.dakota_directive = ('environment',
                                 'method',
                                 'model',
                                 'variables',
                                 'interface',
                                 'responses', )
        self.environment = ["tabular_data", "  tabular_data_file 'dakota.dat'"]

        self.method = [
            "multidim_parameter_study",
            "  partitions = 4 4",
        ]

        self.model = ["single", ]

        self.variables = [
            "continuous_design = 2",
            "  lower_bounds -1   -1",
            "  upper_bounds  1    1",
            "  descriptors 'x1' 'x2'",
        ]

        self.interface = [
            "analysis_driver = 'rosenbrock'",
            "  direct",
        ]

        self.responses = [
            "num_objective_functions = 1",
            "no_gradients",
            "no_hessians",
        ]

        for k in kws:
            setattr(self, k, kws[k])

    def write(self, infile=None):
        """ write all the input into file, as dakota input file

        :param infile: fullname of input file, if not defined, infile will
            be assigned as 'dakota.in' in current working directory
        """
        if infile is None:
            cur_dir = os.path.curdir
            infile = os.path.expanduser(os.path.realpath(os.path.join(
                cur_dir, 'dakota.in')))
        else:
            infile = os.path.expanduser(infile)

        out = open(infile, 'w')
        for k in self.dakota_directive:
            out.write(k)
            out.write('\n')
            out.write('  ' + '\n  '.join(getattr(self, k)))
            out.write('\n\n')
        out.close()

    def set_template(self, name='oc'):
        self._template_oc()

    def _template_oc(self):
        """ template input file for orbit correction
        """
        entry_interface = []
        entry_interface.append('fork')
        entry_interface.append('analysis_driver = "{driver} {argv}"'.format(
            driver='flamedriver_oc',
            argv='/home/tong1/work/FRIB/projects/flame_github/optdrivers/oc/test_392.lat'))
        entry_interface.append('deactivate = active_set_vector')

        entry_method = []
        entry_method.append('conmin_frcg')
        entry_method.append('  max_iterations=30')
        entry_method.append('  convergence_tolerance=1e-4')

        entry_variables = []
        entry_variables.append('continuous_design = 2')
        entry_variables.append('  initial_point 1e-4 1e-4')
        entry_variables.append('  lower_bounds -0.01 -0.01')
        entry_variables.append('  upper_bounds  0.01  0.01')
        entry_variables.append('  descriptors   "x1" "x2"')

        entry_responses = []
        entry_responses.append('num_objective_functions=1')
        entry_responses.append('numerical_gradients')
        entry_responses.append('  method_source dakota')
        entry_responses.append('  interval_type forward')
        entry_responses.append('  fd_gradient_step_size 1.0e-6')
        entry_responses.append('no_hessians')

        self.interface = entry_interface
        self.method = entry_method
        self.variables = entry_variables
        self.responses = entry_responses
                

class DakotaParam(object):
    """ create dakota variable for ``variables`` block

    :param label: string to represent itself, e.g. ``x001``,
                  it is recommended to annotate the number with the format of ``%03d``,
                  i.e. ``1 --> 001``, ``10 --> 010``, ``100 --> 100``, etc.
    :param initial: initial value, 0.0 by default
    :param lower: lower bound, -1.0e10 by default
    :param upper: upper bound, 1.0e10 by default
    """

    def __init__(self, label, initial=0.0, lower=-1.0e10, upper=1.0e10):
        self._label = "'{label}'".format(label=label)
        self._initial = initial
        self._lower = lower
        self._upper = upper

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, s):
        self._label = "'{label}'".format(label=s)

    @property
    def initial(self):
        return self._initial

    @initial.setter
    def initial(self, x):
        self._initial = x

    @property
    def lower(self):
        return self._lower

    @lower.setter
    def lower(self, x):
        self._lower = x

    @property
    def upper(self):
        return self._upper

    @upper.setter
    def upper(self, x):
        self._upper = x

    def __repr__(self):
        return "Parameter {label:>3s}: initial value is {initial:>.3g}, limit range is [{lower:>.3g}:{upper:>.3g}]".format(
            label=self._label,
            initial=self._initial,
            lower=self._lower,
            upper=self._upper)


class DakotaInterface(object):
    """ create dakota interface for ``interface`` block

    :param mode: 'fork' or 'direct' (future usage)
    :param driver: analysis driver, external ('fork') executable file
                   internal ('direct') executable file
    :param latfile: file name of (flame) lattice file
    :param bpms: array of selected BPMs' id
    :param hcors: array of selected horizontal (x) correctors' id
    :param vcors: array of selected vertical (y) correctors' id
    :param ref_x0: array of BPM readings for reference orbit in x, if not defined, use 0s
    :param ref_y0: array of BPM readings for reference orbit in y, if not defined, use 0s
    :param ref_flag: string flag for objective functions:

           1. "x": :math:`\sum \Delta x^2`, :math:`\Delta x = x-x_0`;
           2. "y": :math:`\sum \Delta y^2`, :math:`\Delta y = y-y_0`;
           3. "xy": :math:`\sum \Delta x^2 + \sum \Delta y^2`.

    :param kws: keyword parameters, valid keys: 
        e.g.:
        * deactivate, possible value: 'active_set_vector'

    .. note:: ``mode`` should be set to be 'direct' when the analysis drivers are
        built with dakota library, presently, 'fork' is used.

    :Example:
    
    >>> # for orbit correction
    >>> bpms = [1,2,3] # just for demonstration
    >>> hcors, vcors = [1,3,5], [2,4,6]
    >>> latfile = 'test.lat'
    >>> oc_interface = DakotaInterface(mode='fork', 
    >>>                                driver='flamedriver_oc',
    >>>                                latfile=latfile
    >>>                                bpms=bpms, hcors=hcors, vcors=vcors,
    >>>                                ref_x0=None, ref_y0=None,
    >>>                                ref_flag=None,
    >>>                                deactivate='active_set_vector')
    >>> # add extra configurations
    >>> oc_interface.set_extra(p1='v1', p2='v2')
    >>> # get configuration
    >>> config_str = oc_interface.get_config()
    >>> 
    """
    def __init__(self, mode='fork', driver='flamedriver_oc', latfile=None,
                 bpms=None, hcors=None, vcors=None, ref_x0=None, ref_y0=None, ref_flag=None, **kws):
        self._mode, self._driver = mode, driver
        self._latfile = latfile
        self._bpms = bpms
        self._hcors, self._vcors = hcors, vcors
        if bpms is not None and bpms != 'all':
            self._ref_x0 = [0]*len(bpms) if ref_x0 is None else ref_x0
            self._ref_y0 = [0]*len(bpms) if ref_y0 is None else ref_y0
        else:
            self._ref_x0, self._ref_y0 = ref_x0, ref_y0
        self._ref_flag = "xy" if ref_flag is None else ref_flag
        self._kws = kws
        for k in kws:
            setattr(self, k, kws.get(k))

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        self._mode = mode
        
    @property
    def driver(self):
        return self._driver

    @driver.setter
    def driver(self, driver):
        self._driver = driver

    @property
    def latfile(self):
        return self._latfile

    @latfile.setter
    def latfile(self, fn):
        self._latfile = fn

    @property
    def bpms(self):
        return self._bpms

    @bpms.setter
    def bpms(self, nlist):
        self._bpms = nlist

    @property
    def hcors(self):
        return self._hcors

    @hcors.setter
    def hcors(self, nlist):
        self._hcors = nlist

    @property
    def vcors(self):
        return self._vcors

    @vcors.setter
    def vcors(self, nlist):
        self._vcors = nlist

    @property
    def ref_x0(self):
        return self._ref_x0

    @ref_x0.setter
    def ref_x0(self, vlist):
        self._ref_x0 = vlist

    @property
    def ref_y0(self):
        return self._ref_y0

    @ref_y0.setter
    def ref_y0(self, vlist):
        self._ref_y0 = vlist

    @property
    def ref_flag(self):
        return self._ref_flag

    @ref_flag.setter
    def ref_flag(self, s):
        self._ref_flag = s

    def set_extra(self, **kws):
        """ add extra configurations
        """
        self._kws.update(kws)

    def __repr__(self):
        s = "Configurations: \n"
        for k,v in self.__dict__.items():
            s += "{0:<10s} ==> {1:<10s}\n".format(str(k), str(v))
        return s

    def get_config(self, rtype='list'):
        """ get interface configuration for dakota input block

        :param rtype: 'list' or 'string'
        :return: dakota interface input
        """
        if self._bpms != 'all':
            bpms = "'" + ' '.join(
                ['{0}'.format(i) for i in self._bpms]) + "'"
        else:  # pseudo_all is True
            bpms = self._bpms
        
        hcors = "'" + ' '.join(
            ['{0}'.format(i) for i in self._hcors]) + "'"
        vcors = "'" + ' '.join(
            ['{0}'.format(i) for i in self._vcors]) + "'"
        ref_x0 = "'" + ' '.join(
            ['{0}'.format(i) for i in self._ref_x0]) + "'"
        ref_y0 = "'" + ' '.join(
            ['{0}'.format(i) for i in self._ref_y0]) + "'"

        oc_interface = []
        oc_interface.append(self._mode)
        oc_interface.append(
            'analysis_driver = "{driver} {latfile} {bpms} {hcors} {vcors} {ref_x0} {ref_y0} {ref_flag}"'.format(
                driver=self._driver,
                latfile=self._latfile,
                bpms=bpms,
                hcors=hcors,
                vcors=vcors, 
                ref_x0=ref_x0,
                ref_y0=ref_y0,
                ref_flag=self._ref_flag,
                ))
        for k,v in self._kws.items():
            oc_interface.append('{0} = {1}'.format(str(k), str(v)))
        if rtype == 'string':
            return '\n'.join(oc_interface)
        else: # list
            return oc_interface


class DakotaModel(object):
    """ create dakota model for ``model`` block
    """
    def __init__(self, **kws):
        self._model = 'single'
        self._kws = kws

    def __repr__(self):
        s = "Configurations: \n"
        for k,v in self.__dict__.items():
            s += "{0:<10s} ==> {1:<10s}\n".format(str(k), str(v))
        return s

    def get_config(self, rtype='list'):
        """ get model configuration for dakota input block

        :param rtype: 'list' or 'string'
        :return: dakota model input
        """
        oc_model = []
        oc_model.append(self._model)
        for k,v in self._kws.items():
            oc_model.append('{0} = {1}'.format(str(k), str(v)))
        if rtype == 'string':
            return '\n'.join(oc_model)
        else: # list
            return oc_model


class DakotaResponses(object):
    """ create dakota responses for ``responses`` block

    :param nfunc: num of objective functions
    :param gradient: gradient type: 'analytic' or 'numerical'
    :param hessian: hessian configuration
    :param kws: keyword parameters for gradients and hessians
        valid keys: any available for responses
        among which key name of 'grad' is for gradients configuration, 
        the value should be a dict (future)

    :Example:

    >>> # default responses:
    >>> response = DakotaResponses()
    >>> print response.get_config()
    ['num_objective_functions = 1', 'no_gradients', 'no_hessians']
    >>> 
    >>> # responses with analytic gradients:
    >>> response = DakotaResponses(gradient='analytic')
    >>> print response.get_config()
    ['num_objective_functions = 1', 'analytic_gradients', 'no_hessians']
    >>> 
    >>> # responses with numerical gradients, default configuration:
    >>> oc_responses = DakotaResponses(gradient='numerical')
    >>> print oc_responses.get_config()
    ['num_objective_functions = 1', 'numerical_gradients', ' method_source dakota', 
     ' interval_type forward', ' fd_gradient_step_size 1e-06', 'no_hessians']
    >>>
    >>> # responses with numerical gradients, define step:
    >>> oc_responses = DakotaResponses(gradient='numerical', step=2.0e-7)
    >>> print oc_responses.get_config()
    ['num_objective_functions = 1', 'numerical_gradients', ' method_source dakota', 
     ' interval_type forward', ' fd_gradient_step_size 2e-07', 'no_hessians']
    >>> 
    >>> # given other keyword parameters:
    >>> oc_responses = DakotaResponses(gradient='numerical', step=2.0e-7, k1='v1', k2='v2')
    >>> print oc_responses.get_config()
    ['num_objective_functions = 1', 'numerical_gradients', ' method_source dakota', 
     ' interval_type forward', ' fd_gradient_step_size 2e-07', 'no_hessians', 'k2 = v2', 'k1 = v1']
    >>> 
    """
    def __init__(self, nfunc=1, gradient=None, hessian=None, **kws):
        self._nfunc = nfunc
        self._gradients = 'no_gradients' if gradient is None else gradient
        self._hessians = 'no_hessians' if hessian is None else hessian
        self._kws = kws

    def gradients(self, type=None, step=1.0e-6, **kws):
        """ generate gradients configuration

        :param type: 'numerical' or 'analytic' (default)
        :param step: gradient step size, only valid when type is numerical
        :param kws: other keyword parameters
        :return: list of configuration
        """
        # replace with keyword parameter 

        g = {}
        g['config'] = {}
        if type is None or type == 'analytic':  # type: analytic
            g['type'] = 'analytic_gradients'
        else: # type: numerical
            g['type'] = 'numerical_gradients'
            g['config']['method_source'] = 'dakota'
            g['config']['interval_type'] = 'forward'
            g['config']['fd_gradient_step_size'] = step

        g_config = [' {0} {1}'.format(k,v) for k,v in g['config'].items()]

        retval = []
        retval.append('{0}'.format(g['type']))
        for i in g_config:
            retval.append(i)

        for k,v in kws.items():
            retval.append('{0} {1}'.format(k,v))

        return retval

    def __repr__(self):
        s = "Configurations: \n"
        for k,v in self.__dict__.items():
            s += "{0:<10s} ==> {1:<10s}\n".format(str(k), str(v))
        return s

    def get_config(self, rtype='list'):
        """ get responses configuration for dakota input block

        :param rtype: 'list' or 'string'
        :return: dakota responses input
        """
        oc_responses = []
        oc_responses.append('num_objective_functions = {0}'.format(self._nfunc))
        if self._gradients == 'no_gradients':
            oc_responses.append('no_gradients')
        else:
            step = 1e-6 if self._kws.get('step') is None else self._kws.pop('step')
            grad_dict = self._kws.get('grad') if self._kws.get('grad') is not None else {}
            gradients = self.gradients(type=self._gradients, step=step, **grad_dict)
            oc_responses.extend(gradients)

        if self._hessians == 'no_hessians':
            oc_responses.append('no_hessians')
        else:
            pass

        for k,v in self._kws.items():
            oc_responses.append('{0} = {1}'.format(str(k), str(v)))
        if rtype == 'string':
            return '\n'.join(oc_responses)
        else: # list
            return oc_responses


class DakotaMethod(object):
    """ create dakota method for ``method`` block

    :param method: method name, 'cg' by default, all possible choices: 'cg', 'ps'
    :param iternum: max iteration number, 20 by default
    :param tolerance: convergence tolerance, 1e-4 by default
    :param kws: other keyword parameters

    :Example:

    >>> # default
    >>> oc_method = DakotaMethod()
    >>> print oc_method.get_config()
    ['conmin_frcg', ' convergence_tolerance 0.0001', ' max_iterations 20']
    >>> # define method with pattern search
    >>> oc_method = DakotaMethod(method='ps')
    >>> print oc_method.get_config()
    ['coliny_pattern_search', ' contraction_factor 0.75', ' max_function_evaluations 500', 
     ' solution_accuracy 0.0001', ' exploratory_moves basic_pattern', 
     ' threshold_delta 0.0001', ' initial_delta 0.5', ' max_iterations 100']
    >>> # modify options of pattern search method
    >>> oc_method = DakotaMethod(method='ps', max_iterations=200, contraction_factor=0.8)
    >>> print oc_method.get_config()
    ['coliny_pattern_search', ' contraction_factor 0.8', ' max_function_evaluations 500', 
     ' solution_accuracy 0.0001', ' exploratory_moves basic_pattern', 
     ' threshold_delta 0.0001', ' initial_delta 0.5', ' max_iterations 200']
    >>> # conmin_frcg method
    >>> oc_method = DakotaMethod(method='cg')
    >>> print oc_method.get_config()
    ['conmin_frcg', ' convergence_tolerance 0.0001', ' max_iterations 20']
    >>> # modify options
    >>> oc_method = DakotaMethod(method='cg', max_iterations=100)
    >>> print oc_method.get_config()
    ['conmin_frcg', ' convergence_tolerance 0.0001', ' max_iterations 100']
    >>> 
    """
    def __init__(self, method='cg', iternum=20, tolerance=1e-4, **kws):
        self._method = method
        self._iternum = iternum
        self._tol = tolerance
        self._kws = {k:v for k,v in kws.items()}

        ps_k = ['solution_accuracy', 'initial_delta', 'threshold_delta', 
                'exploratory_moves', 'contraction_factor', 'max_iterations', 
                'max_function_evaluations']
        ps_v = [1e-4, 0.5, 1e-4, 'basic_pattern', 0.75, 100, 500]
        self._ps_d = dict(zip(ps_k, ps_v))

        cg_k = ['max_iterations', 'convergence_tolerance']
        cg_v = [iternum, tolerance]
        self._cg_d = dict(zip(cg_k, cg_v))

        if method == 'cg':
            self._cg_d.update(kws)
            [self._kws.pop(k) for k in kws if k in cg_k]
        elif method == 'ps':
            self._ps_d.update(kws)
            [self._kws.pop(k) for k in kws if k in ps_k]


    def get_default_method(self, method):
        """ get default configuration of some method

        :param method: method name, 'cg' or 'ps'
        :return: dict of configuration
        """
        if method == 'cg':
            for k,v in self._cg_d.items():
                return "{k:<10s}:{v:<10s}".format(str(k),str(v))
        elif method == 'ps':
            for k,v in self._ps_d.items():
                return "{k:<10s}:{v:<10s}".format(str(k),str(v))

    def method(self, method):
        """ return method configuration
        
        :param method: method stirng name, 'cg' or 'ps'
        :return: list of method configuration
        """
        mdict = {}
        mdict['config'] = {}
        if method == 'cg':
            mdict['type'] = 'conmin_frcg'
            mdict['config'] = self._cg_d
        elif method == 'ps':
            mdict['type'] = 'coliny_pattern_search'
            mdict['config'] = self._ps_d
        
        retval = []
        retval.append('{0}'.format(mdict['type']))
        retval.extend([' {0} {1}'.format(k,v) for k,v in mdict['config'].items()])

        return retval


    def __repr__(self):
        s = "Configurations: \n"
        for k,v in self.__dict__.items():
            s += "{0:<10s} ==> {1:<10s}\n".format(str(k), str(v))
        return s

    def get_config(self, rtype='list'):
        """ get method configuration for dakota input block

        :param rtype: 'list' or 'string'
        :return: dakota method input
        """
        oc_method = self.method(self._method)

        for k,v in self._kws.items():
            oc_method.append('{0} = {1}'.format(str(k), str(v)))
        if rtype == 'string':
            return '\n'.join(oc_method)
        else: # list
            return oc_method


class DakotaEnviron(object):
    """ create datako environment for ``environment`` block

    :param tabfile: tabular file name, by default not save tabular data
    :param kws: other keyword parameters

    :Example:

    >>> # default
    >>> oc_environ = DakotaEnviron()
    >>> print oc_environ.get_config()
    []
    >>> # define name of tabular file
    >>> oc_environ = DakotaEnviron(tabfile='tmp.dat')
    >>> print oc_environ.get_config()
    ['tabular_data', " tabular_data_file 'tmp.dat'"]
    >>> 
    """
    def __init__(self, tabfile=None, **kws):
        self._tabfile = tabfile
        self._kws = kws
    
    def __repr__(self):
        s = "Configurations: \n"
        for k,v in self.__dict__.items():
            s += "{0:<10s} ==> {1:<10s}\n".format(str(k), str(v))
        return s

    def get_config(self, rtype='list'):
        """ get responses configuration for dakota input block

        :param rtype: 'list' or 'string'
        :return: dakota responses input
        """
        oc_environ = []
        if self._tabfile is not None:
            oc_environ.append('tabular_data')
            oc_environ.append(" tabular_data_file '{0}'".format(self._tabfile))

        for k,v in self._kws.items():
            oc_environ.append('{0} = {1}'.format(str(k), str(v)))
        if rtype == 'string':
            return '\n'.join(oc_environ)
        else: # list
            return oc_environ


def generate_latfile(machine, latfile='out.lat'):
    """ Generate lattice file for the usage of FLAME code

    :param machine: flame machine object
    :param latfile: file name for generated lattice file, 'out.lat' by default
    :return: None if failed to generate lattice file, or the out file name

    :Example:

    >>> from flame import Machine
    >>> latfile = 'test.lat'
    >>> m = Machine(open(latfile))
    >>> outfile1 = generate_latfile(m, 'out1.lat')
    >>> m.reconfigure(80, {'theta_x': 0.1})
    >>> outfile2 = generate_latfile(m, 'out2.lat')
    >>> 

    .. warning:: To get element configuration only by ``m.conf(i)`` method,
        where ``m`` is ``flame.Machine`` object, ``i`` is element index,
        when some re-configuring operation is done, ``m.conf(i)`` will be update,
        but ``m.conf()["elements"]`` remains with the initial value.
    """
    m = machine
    try:
        mconf = m.conf()
        mks = mconf.keys()
    except:
        print("Failed to load FLAME machine object.")
        return None

    try:
        mconf_ks = mconf.keys()
        [mconf_ks.remove(i) for i in ['elements', 'name'] if i in mconf_ks]

        #
        lines = []
        for k in mconf_ks:
            v = mconf[k]
            if isinstance(v, ndarray):
                v = v.tolist()
            if isinstance(v, str):
                v = '"{0}"'.format(v)
            line = '{k} = {v};'.format(k=k, v=v)
            lines.append(line)

        mconfe = mconf['elements']

        # element configuration
        elem_num = len(mconfe)
        elem_name_list = []
        for i in range(0, elem_num):
            elem_i = m.conf(i)
            ename, etype = elem_i['name'], elem_i['type']
            if ename in elem_name_list:
                continue
            elem_name_list.append(ename)
            ki = elem_i.keys()
            elem_k = set(ki).difference(mks)
            if etype == 'stripper':
                elem_k.add('IonChargeStates')
                elem_k.add('NCharge')
            p = []
            for k, v in elem_i.items():
                if k in elem_k and k not in ['name', 'type']:
                    if isinstance(v, ndarray):
                        v = v.tolist()
                    if isinstance(v, str):
                        v = '"{0}"'.format(v)
                    p.append('{k} = {v}'.format(k=k, v=v))
            pline = ', '.join(p)

            line = '{n}: {t}, {p}'.format(n=ename, t=etype, p=pline)

            line = line.strip(', ') + ';'
            lines.append(line)

        dline = '(' + ', '.join(([e['name'] for e in mconfe])) + ')'

        blname = mconf['name']
        lines.append('{0}: LINE = {1};'.format(blname, dline))
        lines.append('USE: {0};'.format(blname))
    except:
        print("Failed to generate lattice file.")
        return None

    try:
        if latfile != sys.stdout:
            fout = open(latfile, 'w')
            fout.writelines('\n'.join(lines))
            fout.close()
        else:
            sys.stdout.writelines('\n'.join(lines))
    except:
        print("Failed to write to %s" % (latfile))
        return None

    return latfile


def get_opt_results(outfile='dakota.out', rtype='dict'):
    """ extract optimized results from dakota output

    :param outfile: file name of dakota output file, 
        'dakota.out' by default
    :param rtype: type of returned results, 'dict' or 'list', 
        'dict' by default
    :return: by default return a dict of optimized results with each item
        of the format like "x1":0.1, etc., or if rtype='list', return a 
        list of values, when the keys are ascend sorted.

    :Example:
    
    >>> opt_vars = get_opt_results(outfile='flame_oc.out', rtype='dict'):
    >>> print(opt_vars)
    {'x2': 0.0020782814353, 'x1': -0.0017913264033}
    >>> opt_vars = get_opt_results(outfile='flame_oc.out', rtype='list'):
    >>> print(opt_vars)
    [-0.0017913264033, 0.0020782814353]

    """
    try:
        append_flag = False
        varlines = []
        for line in open(outfile, 'r'):
            line = line.strip('< >\n\t')
            if append_flag:
                varlines.append(line)
            if line.startswith('Best parameters'):
                append_flag = True
            if line.startswith('Best objective'):
                append_flag = False

        retdict = {
            k: float(v)
            for v, s, k in [line.partition(' ') for line in varlines[:-1]]
        }

        if rtype == 'dict':
            retval = retdict
        else:  #
            sorted_tuple = sorted(
                retdict.items(),
                key=lambda x: '{0:0>20}'.format(x[0]).lower())
            retval = [v for k, v in sorted_tuple]

        return retval

    except IOError:
        print("Cannot open %s" % (outfile))
        sys.exit(1)


def random_string(length=8):
    """ generate random string with given length

    :param length: string length, 8 by default
    :return: random strings with defined length
    """
    return ''.join(
        [random.choice(string.letters + string.digits) for _ in range(length)])


def test_one_element(x):
    """ test if all the elements are the same

    :param x: list, tuple, or numpy array
    :return: True if all are the same, else False
    """
    retval = np.alltrue(np.array(x) == np.ones(len(x))*x[0])
    return retval


def test_dakotainput():
    entry_interface = []
    entry_interface.append('fork')
    entry_interface.append('analysis_driver = "{driver} {argv}"'.format(
        driver='flamedriver_oc',
        argv='/home/tong1/work/FRIB/projects/flame_github/optdrivers/oc/test_392.lat'))
    entry_interface.append('deactivate = active_set_vector')

    entry_method = []
    entry_method.append('conmin_frcg')
    entry_method.append('  max_iterations=500')
    entry_method.append('  convergence_tolerance=1e-7')

    entry_variables = []
    entry_variables.append('continuous_design = 2')
    entry_variables.append('  initial_point 1e-4 1e-4')
    entry_variables.append('  lower_bounds -0.01 -0.01')
    entry_variables.append('  upper_bounds  0.01  0.01')
    entry_variables.append('  descriptors   "x1" "x2"')

    entry_responses = []
    entry_responses.append('num_objective_functions=1')
    entry_responses.append('numerical_gradients')
    entry_responses.append('  method_source dakota')
    entry_responses.append('  interval_type forward')
    entry_responses.append('  fd_gradient_step_size 1.0e-6')
    entry_responses.append('no_hessians')

    dak_inp = DakotaInput(interface=entry_interface,
                          method=entry_method,
                          variables=entry_variables,
                          responses=entry_responses, )
    dak_inp.write('./test/test_oc.in')

def test_get_opt_results():
    opt_vars = get_opt_results('test/flame_oc.out', rtype='dict')
    print opt_vars

    opt_vars = get_opt_results('test/flame_oc.out', rtype='list')
    print opt_vars

def test_dakotaparam():
    p1 = DakotaParam('x1', 0.1, -10, 10)
    p2 = DakotaParam('x2', 0.2, -20, 20)
    print p1
    print p2
    plist = [p1, p2]
    print ' '.join(["{0:>14s}".format(p.label) for p in plist])
    print ' '.join(["{0:>14e}".format(p.initial) for p in plist])
    print ' '.join(["{0:>14e}".format(p.lower) for p in plist])
    print ' '.join(["{0:>14e}".format(p.upper) for p in plist])

def test_dakotainterface():
    if1 = DakotaInterface()
    print if1

    if2 = DakotaInterface(mode='fork')
    print if2

    if3 = DakotaInterface(mode='fork', driver='flamedriver_oc')
    print if3

    if4 = DakotaInterface(mode='fork', driver='flamedriver_oc', deactivate='active_set_vector')
    print if4


    bpms = [1,2,3] # just for demonstration
    hcors, vcors = [1,3,5], [2,4,6]
    latfile = 'test.lat'
    oc_interface = DakotaInterface(mode='fork', latfile=latfile,
                                   driver='flamedriver_oc',
                                   bpms=bpms, hcors=hcors, vcors=vcors,
                                   deactivate='active_set_vector')
    oc_interface.set_extra(p1='k1', p2='k2')
    oc_interface.set_extra(p1='kk1', p3='kk3')
    print oc_interface
    print oc_interface.get_config()

    bpms = range(1242) # just for demonstration
    hcors, vcors = range(1242)[0:-1:2], range(1242)[1:-1:2]
    latfile = 'test.lat'
    oc_interface = DakotaInterface(mode='fork', latfile=latfile,
                                   driver='flamedriver_oc',
                                   bpms=bpms, hcors=hcors, vcors=vcors,
                                   deactivate='active_set_vector')
    oc_interface.set_extra(p1='k1', p2='k2')
    oc_interface.set_extra(p1='kk1', p3='kk3')
    print oc_interface
    print oc_interface.get_config()[1]


def test_dakotamodel():
    oc_model = DakotaModel()
    print oc_model
    print oc_model.get_config()

def test_dakotaresponses():
    oc_responses = DakotaResponses()
    print oc_responses.get_config()

    oc_responses = DakotaResponses(gradient='analytic')
    print oc_responses.get_config()

    oc_responses = DakotaResponses(gradient='numerical')
    print oc_responses.get_config()

    oc_responses = DakotaResponses(gradient='numerical', step=2.0e-7)
    print oc_responses.get_config()

    oc_responses = DakotaResponses(gradient='numerical', step=2.0e-7, k1='v1', k2='v2')
    print oc_responses.get_config()
    print oc_responses

def test_dakotamethod():
    oc_method = DakotaMethod()
    print oc_method.get_config()

    oc_method = DakotaMethod(method='ps')
    print oc_method.get_config()

    oc_method = DakotaMethod(method='ps', max_iterations=200, contraction_factor=0.8)
    print oc_method.get_config()

    oc_method = DakotaMethod(method='cg')
    print oc_method.get_config()

    oc_method = DakotaMethod(method='cg', max_iterations=100)
    print oc_method.get_config()

def test_dakotaenviron():
    oc_environ = DakotaEnviron()
    print oc_environ.get_config()

    oc_environ = DakotaEnviron(tabfile='tmp.dat')
    print oc_environ.get_config()


if __name__ == '__main__':
    #test_dakotainput()
    #test_get_opt_results()
    #test_dakotaparam()
    test_dakotainterface()
    #test_dakotamodel()
    #test_dakotaresponses()
    #test_dakotamethod()
    #test_dakotaenviron()
    
