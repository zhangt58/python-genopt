Setup BPMs, correctors and reference orbit
==========================================

For more general cases, ``genopt`` provides interfaces to setup
BPMs, correctors, reference orbit and objective function type, etc.,
leaving more controls to the user side, to fulfill specific task.


Here is an exmaple to show how to use these capabilities.

.. literalinclude:: ../../snippets/demo2.py
    :language: python
    :emphasize-lines: 7-22

The highlighted code block is added for controlling all these
abovementioned properties.

.. warning::
    1. BPMs and correctos are distinguished by the element index, which
       could be get by proper method, e.g. ``get_all_cors()``;
    2. The array size of selected BPMs and reference orbit must be the same;
    3. ``bpms``, ``hcors``, ``vcors`` are properties of ``DakotaOC`` instance.

.. warning::
    All elements could be treated as `BPMs`, see :func:`set_bpms()`, set ``pseudo_all=True``
    option will use all elements as monitors.

.. note::
    Objective functions could be chosen from three types according to the value
    of ``ref_flag``:
        1. ``ref_flag="xy"``: :math:`\sum \Delta x^2 + \sum \Delta y^2`
        2. ``ref_flag="x"``: :math:`\sum \Delta x^2`
        3. ``ref_flag="y"``: :math:`\sum \Delta y^2`
    where :math:`\Delta x = x - x_0`, :math:`\Delta y = y - y_0`.
