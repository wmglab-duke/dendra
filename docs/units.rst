Units used in AxonML
====================

===========     =======================
Quantity        Default / Assumed Units
===========     =======================
Voltage         mV
Current         mA
Resistivity     Ω·cm
Capacitance     µF/cm²
Length          µm
Time            ms
Frequency       kHz
===========     =======================

Internally, AxonML assumes the above units for all physical quantities.
However, it may be natural to use different units under certain circumstances. For example, you may want to specify a length in millimeters (mm) instead of micrometers (µm) (e.g., when speciying the length of an axon), or a time in seconds (s) instead of milliseconds (ms). For intracellular current injection in particular, it is more natural to express the amplitude in units of nA. To facilitate this, AxonML provides a set of unit conversion factors that you can use to convert between different units in :mod:`axonml.units`:

.. code-block:: python

    from axonml.units import mm, s, nA

    length_mm = 1.0 * mm   # 1 mm in micrometers
    time_s = 0.5 * s       # 0.5 seconds in milliseconds
    current_nA = 2.0 * nA  # 2 nA in mA

This allows you to write code that is more readable and easier to understand, while still ensuring that the internal calculations are performed using the correct units.
