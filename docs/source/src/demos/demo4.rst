Setup optimization engine
=========================

The simplest approach, (see :ref:`simplest_approach`), just covers detail 
of the more specific configurations, especially for the optimization engine
itself, however ``genopt`` provides different interfaces to make customized
adjustment.

Method
------

``DakotaMethod`` is designed to handle ``method`` block, which is essential
to define the optimization method, e.g.

.. code-block:: python

    oc_method = genopt.DakotaMethod(method='ps', max_iterations=200,
                contraction_factor=0.8)
    # other options could be added, like max_function_evaluations=2000
    oc_ins.set_method(oc_method)

Interface
---------

``DakotaInterface`` is designed to handle ``interface`` block, for the
general optimization regime, ``fork`` mode is the common case, only if
the analysis driver is compile into dakota, ``direct`` could be used.

Here is an example of user-defined interface:

.. code-block:: python

    bpms = [10,20,30]
    hcors, vcors = [5, 10, 20], [7, 12, 30]
    latfile = 'test.lat'
    oc_inter = genopt.DakotaInterface(mode='fork',
                        driver='flamedriver_oc',
                        latfile=latfile,
                        bpms=bpms, hcors=hcors, vcors=vcors,)
    # set interface
    oc_ins.set_interface(oc_inter)

.. note:: Extra parameters could be added by this way:
    oc_inter.set_extra(deactivate="active_set_vector")

Responses
---------

Objective function(s) and gradients/hessians could be set in 
``responses`` block, which is handled by ``DakotaResponses`` class.

Typical example:

.. code-block:: python

    oc_responses = DakotaResponses(gradient='numerical', step=2.0e-7)
    oc_ins.set_responses(oc_responses)

Environment
-----------

Dakota environment block could be adjusted by instantiating class 
``DakotaEnviron``, e.g. 

.. code-block:: python
    
    datfile = 'dakota1.dat'
    e = genopt.DakotaEnviron(tabfile=datfile)
    oc_ins.set_environ(e)

``tabfile`` option could be used to define where the dakota tabular data 
should go, will not generate tabular file if not set.

Model
-----

``DakotaModel`` is designed to handle ``model`` block, recently, just use
the default configuration, i.e:

.. code-block:: python

    oc_ins.set_model()
    # or:
    m = genopt.DakotaModel()
    oc_ins.set_model(m)

