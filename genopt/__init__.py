from .dakutils import DakotaInput, DakotaParam
from .dakutils import get_opt_results
from .dakopt   import DakotaBase, DakotaOC

__version__ = "0.0.1"
__author__ = "Tong Zhang"

__doc__ = """General multi-dimensional optimization package built by Python,
incorporating optimization algorithms provided by DAKOTA.

:version: %s
:author: Tong Zhang <zhangt@frib.msu.edu>

:Example:

>>> # This is a ordinary example to do orbit correction by 
>>> # multi-dimensional optimization approach.
>>>  
>>> # import package
>>> import genopt
>>> 
>>> # lattice file name
>>> latfile = './contrib/test_392.lat'
>>> 
>>> # create optimization object
>>> oc_ins = genopt.DakotaOC(lat_file=latfile)
>>> 
>>> # get indices of BPMs and correctors
>>> bpms = oc_ins.get_elem_by_type('bpm')
>>> cors = oc_ins.get_all_cors()[45:61]
>>> 
>>> # set BPMs and correctors
>>> oc_ins.set_bpms(bpm=bpms)
>>> oc_ins.set_cors(cor=cors)
>>> 
>>> # set parameters
>>> oc_ins.set_variables()
>>> 
>>> # generate dakota input file
>>> oc_ins.gen_dakota_input()
>>> 
>>> # run optimization, enable MPI
>>> oc_ins.run(mpi=True, np=4)
>>> 
>>> # get optimized results:
>>> opt_vars = oc_ins.get_opt_results()
>>> 
>>> # or show orbit after correction
>>> oc_ins.plot()
>>> 
>>> # or save the orbit data (to file)
>>> oc_ins.get_orbit((oc_ins.hcor, oc_ins.vcor), opt_vars, outfile='orbit.dat')
>>> 
""" % (__version__)

__all__ = ["DakotaInput", "DakotaParam", "DakotaBase", "DakotaOC",
           "get_opt_results"]
