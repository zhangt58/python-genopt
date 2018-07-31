.. _simplest_approach:

Getting started
===============

This approach requires fewest input of code to complete the orbit 
correction optimization task, which also means you only has very few
options to adjust to the optimization model. Hopefully, this approach 
could be used as an ordinary template to fulfill most of the orbit 
correction tasks. Below is the demo code:

.. literalinclude:: ../../snippets/demo1.py
    :language: python

The lattice file used here could be found from 
:download:`here <../../snippets/test_392.lat>`, or from `https://github.com/archman/genopt/blob/master/lattice/test_392.lat <https://github.com/archman/genopt/blob/master/lattice/test_392.lat>`_.

For this approach, the following default configuration is applied:

1. Selected all BPMs and correctors (both horizontal and vertical types);
2. Set the reference orbit with all BPMs' readings of ``x=0`` and ``y=0``;
3. Set the objective function with the sum of all the square of orbit deviations w.r.t. reference orbit.

By default, ``conmin_frcg`` optimization method is used, possible options
for ``simple_run()`` could be:

* common options:
    1. ``mpi``: if True, run in parallel mode; if False, run in serial mode;
    2. ``np``: number of cores to use if ``mpi`` is True;
    3. ``echo``: if False, will not generate output when optimizing, the same for :func:`run()`;
* gradient descent, i.e. ``method=cg``:
    1. ``iternum``: max iteration number, 20 by default;
    2. ``step``: forward gradient step size, 1e-6 by default;
* pattern search, i.e. ``method=ps``:
    1. ``iternum``: max iteration number, 20 by default;
    2. ``evalnum``: max function evaulation number, 1000 by default;

There are two options for ``DakotaOC`` maybe useful sometimes:

1. ``workdir``: root directory for dakota input and output files
2. ``keep``: if keep working files, True or False

After run this script, beam orbit data could be saved into file, e.g. 
:download:`orbit.dat <../../snippets/orbit.dat>`:

which could be used to generate figures, the following figure is a typical
one could be generated from the optimized results:

.. image:: ../../images/oc_x0y0.png
    :width: 500px
    :align: center
