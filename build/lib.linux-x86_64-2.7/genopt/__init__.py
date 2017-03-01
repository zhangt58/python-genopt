from .dakutils import DakotaInput
from .dakutils import DakotaParam
from .dakutils import DakotaInterface
from .dakutils import DakotaMethod
from .dakutils import DakotaModel
from .dakutils import DakotaResponses
from .dakutils import DakotaEnviron
from .dakutils import get_opt_results
from .dakopt   import DakotaBase, DakotaOC

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

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
>>> # run optimization, enable MPI, 
>>> # with optimization of CG, 20 iterations
>>> oc_ins.simple_run(method='cg', mpi=True, np=4, iternum=20)
>>> 
>>> # get optimized results:
>>> opt_vars = oc_ins.get_opt_results()
>>> 
>>> # or show orbit after correction
>>> oc_ins.plot()
>>> 
>>> # or save the orbit data (to file)
>>> oc_ins.get_orbit(outfile='orbit.dat')
>>> 
""" % (__version__)

__all__ = ["DakotaInput", "DakotaParam", "DakotaBase", "DakotaOC",
           "DakotaEnviron", "DakotaInterface", "DakotaMethod",
           "DakotaModel", "DakotaResponses",
           "get_opt_results"]
