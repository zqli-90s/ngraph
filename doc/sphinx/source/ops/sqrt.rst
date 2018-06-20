.. sqrt.rst:

####
Sqrt
####

.. code-block:: cpp

   Sqrt  //  Elementwise square root operation.


Description
===========

Constructs a square operation.


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

.. doxygenclass:: ngraph::op::Sqrt
   :project: ngraph
   :members:
