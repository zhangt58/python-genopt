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
            driver='flamedriver',
            argv='/home/tong1/work/FRIB/projects/flame_github/optdrivers/oc/test_392.lat'))
        entry_interface.append('deactivate = active_set_vector')

        entry_method = []
        entry_method.append('conmin_frcg')
        entry_method.append('  max_iterations=100')
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

    :param label: string to represent itself, e.g. 'x1'
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


def test_dakotainput():
    entry_interface = []
    entry_interface.append('fork')
    entry_interface.append('analysis_driver = "{driver} {argv}"'.format(
        driver='flamedriver',
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


if __name__ == '__main__':
    test_dakotainput()
    test_get_opt_results()
    test_dakotaparam()
