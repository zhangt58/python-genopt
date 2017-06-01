Introduction
============

``genopt`` is a python package, trying to serve as a solution of general 
multi-dimensional optimization. The core optimization algorithms employed
inside are mainly provided by ``DAKOTA``, which is the brief for 
*Design Analysis Kit for Optimization and Terascale Applications*, 
another tool written in C++.

The following image illustrates the general optimization framework
by properly utilizing ``DAKOTA``.

.. image:: ../images/dakota-sys-workflow_2.png
    :align: center
    :width: 600px

To apply this optimization framework, specific ``analysis drivers`` should
be created first, e.g. ``flamedriver1``, ``flamedriver2``... indicate the
dedicated executable drivers built from C++, for the application in 
accelerator commissioning, e.g. FRIB.

.. image:: ../images/dakota-genopt-framework.png
    :align: center
    :width: 600px

.. note:: ``flame`` is an particle envolope tracking code developed by C++,
    with the capbility of multi-charge particle states momentum space 
    tracking, it is developed by FRIB; ``flamedriver(s)`` are 
    user-customized executables by linking the flame core library 
    (``libflame_core.so``) to accomplish various different requirements.

The intention of ``genopt`` is to provide a uniform interface to do the
multi-dimensional optimization tasks. It provides interfaces to let the
users to customize the optimization drivers, optimization methods, 
variables, etc. The optimized results are returned by clean interface.
Dedicated analysis drivers should be created and tell the package to use.
``DakotaOC`` is a dedicated class designed for orbit correction for 
accelerator, which uses ``flame`` as the modeling tool.
