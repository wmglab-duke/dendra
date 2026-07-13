dendra.utils
============

Interpolation
-------------

.. autoclass:: dendra.utils.PreparedInterp1d
   :members:
   :special-members: __init__
.. autoclass:: dendra.utils.PreparedInterp1dUniform
   :members:
.. autoclass:: dendra.utils.PreparedInterp3dRect
   :members:
.. autoclass:: dendra.utils.PreparedInterp3dRectUniform
   :members:
.. autoclass:: dendra.utils.PreparedInterp3dScattered
   :members:
.. autofunction:: dendra.utils.interp1d
.. autofunction:: dendra.utils.interp1d_uniform


Parameter sweeps and tensor shapes
----------------------------------

These helpers construct flattened parameter grids and align one-dimensional
parameters with model tensors without materializing an intermediate meshgrid.

.. autofunction:: dendra.utils.tensor_ops.cartesian_product
.. autofunction:: dendra.utils.tensor_ops.add_dims_as_necessary


NEURON morphology paths
-----------------------

The path helpers operate on connected NEURON ``Section`` trees. They return
sections in traversal order and return ``None`` when no qualifying path exists.

.. autofunction:: dendra.utils.neuron.path_sections
.. autofunction:: dendra.utils.neuron.path_via
