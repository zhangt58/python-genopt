Setup variables
===============

By default the variables to be optimized is setup with the following 
parameters:

+-----------------+------------------+----------------+
|  initial value  |   lower bound    |   upper bound  |
+=================+==================+================+
|     1e-4        |      -0.01       |      0.01      |
+-----------------+------------------+----------------+

However, subtle configuration could be achieved by using ``set_variables()``
method of ``DakotaOc`` class, here is how to do it:

Parameter could be created by using ``DakotaParam`` class, here is the code:

.. literalinclude:: ../../snippets/demo3.py
    :language: python
    :emphasize-lines: 9

``plist_y`` could be created in the same way, then issue ``set_variables()``
with ``set_variables(plist=plist_x+plist_y)``.

.. note::
    The emphasized line is to setup the variable labels, it is recommended
    that all parameters' label with the format like ``x001``, ``x002``, etc.
    
