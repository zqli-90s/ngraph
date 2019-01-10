.. dyn_reshape.rst:

##########
DynReshape
##########

.. code-block:: cpp

   DynReshape  // Operation that reshapes a tensor to a shape computed at runtime


Description
===========

.. warning:: This op is experimental and subject to change without notice.

This is a dynamic variant of the `Reshape` operation. TODO: examples.

Inputs
------

+------------------+-------------------------+---------------------------------+
| Name             | Element Type            | Shape                           |
+==================+=========================+=================================+
| ``arg``          | Any                     | Any                             |
+------------------+-------------------------+---------------------------------+
| ``output_shape`` | Any                     | Any with rank=1                 |
+------------------+-------------------------+---------------------------------+

Outputs
-------

+-----------------+-----------------+------------------+
| Name            | Element Type    | Shape            |
+=================+=================+==================+
| ``output``      | Same as ``arg`` | ``output_shape`` |
+-----------------+-----------------+------------------+


Mathematical Definition
=======================

TODO


C++ Interface
=============

.. doxygenclass:: ngraph::op::DynReshape
   :project: ngraph
   :members:
