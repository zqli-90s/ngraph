
Local Response Normalization
############################

.. code-block:: cpp

   LRN  // Elementwise Local Response Normalization operation


Description
===========


.. TODO


Inputs
------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``arg``         | Any                     | Any                            |
+-----------------+-------------------------+--------------------------------+

Attributes
----------
+-----------------+----------------------------------------------------------------+
| Name            | Description                                                    |
+=================+================================================================+
| ``axes``        | The axis positions (0-based) on which to calculate the softmax |
+-----------------+----------------------------------------------------------------+

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

   \texttt{output}_{i} = \frac{\exp(\texttt{arg}_{i})}{\sum_{j} \exp(\texttt{arg}_{j})}


C++ Interface
=============

.. doxygenclass:: ngraph::op::LRN
   :project: ngraph
   :members: m_axes



..      ///
        /// ## Inputs
        ///
        /// |       | Type                              | Description                                     |
        /// | ----- | --------------------------------- | ----------------------------------------------- |
        /// | `arg` | \f$N[n, c, d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape and numeric element type. |
        ///
        /// ## Output
        ///
        /// | Type                   | Description                                                                          |
        /// | ---------------------- | ------------------------------------------------------------------------------------ |
        /// | \f$N[n, c, d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[n, c, d_1,\dots,d_n] = \frac{N[n,i,d_1,\dots,d_n]}{ (bias + alpha * (\sum_{i=max(0,(nsize-1)/2)}^{min(C, (nsize-1)/2)+1} N[n,i,d_1,\dots,d_n]^{2}) ^ {2})}\f$ |
        class LRN : public util::UnaryElementwiseArithmetic
        {

