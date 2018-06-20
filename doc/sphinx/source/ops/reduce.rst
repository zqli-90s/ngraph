.. reduce.rst:

######
Reduce
######

.. code-block:: cpp

   Reduce  //  Tensor reduction operation.


Description
===========

Reduces the input tensor, eliminating the specified reduction axes, 
given a reduction function that maps two scalars to a scalar.

.. For example, if the reduction function \f$f(x,y) = x+y\f$:

.. \f[
    \mathit{reduce}\left(f,\{0\},
        \left[ \begin{array}{ccc}
               1 & 2 \\ 3 & 4 \\ 5 & 6 \end{array} \right]\right) =
    \left[ (1 + 3 + 5), (2 + 4 + 6) \right] =
    \left[ 9, 12 \right]~~~\text{(dimension 0 (rows) is eliminated)}
.. \f]

.. \f[
    \mathit{reduce}\left(f,\{1\},
        \left[ \begin{array}{ccc}
               1 & 2 \\ 3 & 4 \\ 5 & 6 \end{array} \right]\right) =
    \left[ (1 + 2), (3 + 4), (5 + 6) \right] =
    \left[ 3, 7, 11 \right]~~~\text{(dimension 1 (columns) is eliminated)}
.. \f]

.. \f[
    \mathit{reduce}\left(f,\{0,1\},
        \left[ \begin{array}{ccc}
               1 & 2 \\ 3 & 4 \\ 5 & 6 \end{array} \right]\right) =
     (1 + 2) + (3 + 4) + (5 + 6) =
     21~~~\text{(both dimensions (rows and columns) are eliminated)}
.. \f]

It is assumed that \f$f\f$ is associative. In other words, the order of operations 
is undefined. In the case where a collapsed dimension is `0`, the value of `arg_init` 
will be substituted.

Note that the parameter `reduction_axes` specifies which axes are to be *eliminated*, 
which can be a bit counterintuitive. For example, as seen above, eliminating the column 
dimension results in the *rows* being summed, not the columns.


Inputs
------

+-----------------+-------------------------+-----------------------------------------+
| Name            | Element Type            | Shape                                   |
+=================+=========================+=========================================+
| ``arg0``        | any                     |                                         |
+-----------------+-------------------------+-----------------------------------------+
| ``arg1``        | same as ``arg0``        |                                         |
+-----------------+-------------------------+-----------------------------------------+

Attributes
----------

+------------------------+---------------+--------------------------------------------------+
| Name                   |               |                                                  |
+========================+===============+==================================================+
| reduction_axes_count   | ``size_t``    |                                                  |
+------------------------+---------------+--------------------------------------------------+

Outputs
-------

+-----------------+-------------------------+----------------------------------------+
| Name            | Element Type            | Shape                                  |
+=================+=========================+========================================+
| ``output``      | same as ``arg0``        |                                        |
+-----------------+-------------------------+----------------------------------------+


Mathematical Definition
=======================

.. math::






C++ Interface
=============

.. doxygenclass:: ngraph::op::Reduce
   :project: ngraph
   :members:
