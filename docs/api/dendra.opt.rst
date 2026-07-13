dendra.opt
==========

Gradient hygiene
----------------

Call :func:`dendra.opt.sanitize_grad` after ``loss.backward()`` and before
``optimizer.step()`` to replace non-finite gradients with zero. Optional value
clipping is applied element by element before optional global L2-norm clipping
across every parameter group in the optimizer. The helper supports dense,
strided gradients; unsupported sparse layouts raise an error before any
gradient is changed.

.. autofunction:: dendra.opt.sanitize_grad
