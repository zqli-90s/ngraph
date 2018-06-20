.. sinh.rst:

####
Sinh
####

.. code-block:: cpp

   Sinh  //  Elementwise hyperbolic sine (sinh) operation.


Description
===========

Constructs a hyperbolic sine operation.



Inputs
------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``arg``         |                         |                                |
+-----------------+-------------------------+--------------------------------+

Outputs
-------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``output``      |                         |                                |
+-----------------+-------------------------+--------------------------------+


Mathematical Definition
=======================

.. math::



Backprop
========

.. math::


C++ Interface
=============

.. doxygenclass:: ngraph::op::Sinh
   :project: ngraph
   :members:
