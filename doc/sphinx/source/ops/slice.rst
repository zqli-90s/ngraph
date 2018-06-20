.. slice.rst:

######
Slice
######

.. code-block:: cpp

   Slice  //  Construct a tensor slice operation.


Description
===========

Takes a slice of an input tensor; for example, the sub-tensor that resides within a bounding box, optionally with stride.


Inputs
------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``arg``         | Any                     | Any                            |
+-----------------+-------------------------+--------------------------------+

Outputs
-------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``output``      | Same as ``arg``         | Same as ``arg``                |
+-----------------+-------------------------+--------------------------------+


Mathematical Definition
=======================

.. math::



Backprop
========

.. math::


C++ Interface
=============

.. doxygenclass:: ngraph::op::Slice
   :project: ngraph
   :members:
