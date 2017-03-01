Run optimization
================

If running optimization not by ``simple_run()`` method, another approach
should be utilized.

.. code-block:: python

    # generate input file for optimization
    oc_ins.gen_dakota_input()

    # run optimization
    oc_ins.run(mpi=True, np=4)


Below is a typical user customized script to find the optimized correctors 
configurations.

.. literalinclude:: ../../snippets/demo5.py
    :language: python


The following figure shows correct the orbit to different reference orbits.

.. image:: ../../images/oc_015.png
    :width: 500px
    :align: center

