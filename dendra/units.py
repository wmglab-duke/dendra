"""
Scalar unit conversions for Dendra's public numerical conventions.

These names are ordinary :class:`float` conversion factors, not unit-carrying
quantities. Multiply a value by one of them to express it in the base unit
documented by the receiving API. For example, model time is expressed in
milliseconds, voltage in millivolts, absolute injected current in milliamperes,
and morphology length in micrometers.

Do not construct compound units by algebraically combining these scalars. A
distributed mechanism conductance, for example, is already entered as a number
in ``S/cm²``; it is not written as ``value * S / cm**2``. Point-process method
bodies use a separate local convention (current in nA and conductance in µS),
described in :class:`dendra.models.mechanisms.PointProcess` and in the units
guide.

Attributes
----------
ms : float
    Millisecond time unit (base unit for time in Dendra).
s : float
    Second (``1000 * ms``).
minutes : float
    Minute (``60 * s``).
hours : float
    Hour (``60 * minutes``).

mV : float
    Millivolt voltage unit (base unit for voltage in Dendra).
V : float
    Volt (``1e3 * mV``).

mA : float
    Milliampere conversion for APIs that accept absolute current, including
    intracellular stimulation and public ``i_membrane`` values. Distributed
    mechanism currents instead use mA/cm²; point-process method bodies use a
    local bare-nA coordinate.
A : float
    Ampere (``1e3 * mA``).
uA : float
    Microampere (``1e-6 * A``).
nA : float
    Nanoampere (``1e-9 * A``).
pA : float
    Picoampere (``1e-12 * A``).
fA : float
    Femtoampere (``1e-15 * A``).

ohm : float
    Ohm resistance unit.
S : float
    Siemens conductance unit (``1 / ohm``).
mS : float
    Millisiemens (``1e-3 * S``).
uS : float
    Microsiemens (``1e-6 * S``).
nS : float
    Nanosiemens (``1e-9 * S``).
pS : float
    Picosiemens (``1e-12 * S``).
fS : float
    Femtosiemens (``1e-15 * S``).

uF : float
    Microfarad capacitance unit (base unit for capacitance in Dendra).
F : float
    Farad (``1e6 * uF``).
nF : float
    Nanofarad (``1e-9 * F``).
pF : float
    Picofarad (``1e-12 * F``).
fF : float
    Femtofarad (``1e-15 * F``).

um : float
    Micrometer length unit (base unit for length in Dendra).
m : float
    Meter (``1e6 * um``).
cm : float
    Centimeter (``1e-2 * m``).
mm : float
    Millimeter (``1e-3 * m``).
nm : float
    Nanometer (``1e-9 * m``).

kHz : float
    Kilohertz frequency unit (base unit for frequency in Dendra).
Hz : float
    Hertz (``1e-3 * kHz``).
MHz : float
    Megahertz (``1e3 * kHz``).
GHz : float
    Gigahertz (``1e6 * kHz``).

Examples
--------
Use these conversions with APIs whose documented base units match them:

>>> from dendra import units as U
>>> length = 100 * U.um
>>> dt = 0.025 * U.ms
>>> holding_potential = -65 * U.mV
>>> injected_current = 2 * U.nA  # converted to Dendra's absolute-current unit, mA
"""

# time
ms = 1.0
s = 1000.0 * ms
minutes = 60.0 * s
hours = 60.0 * minutes

# voltage
mV = 1.0
V = 1.0e3 * mV

# current
mA = 1.0
A = 1.0e3 * mA
uA = 1.0e-6 * A
nA = 1.0e-9 * A
pA = 1.0e-12 * A
fA = 1.0e-15 * A

# conductance
ohm = 1.0
S = 1.0 / ohm
mS = 1.0e-3 * S
uS = 1.0e-6 * S
nS = 1.0e-9 * S
pS = 1.0e-12 * S
fS = 1.0e-15 * S

# capacitance
uF = 1.0
F = 1.0e6 * uF
nF = 1.0e-9 * F
pF = 1.0e-12 * F
fF = 1.0e-15 * F

# length
um = 1.0
m = 1.0e6 * um
cm = 1.0e-2 * m
mm = 1.0e-3 * m
nm = 1.0e-9 * m

# frequency
kHz = 1.0
Hz = 1.0e-3 * kHz
MHz = 1.0e3 * kHz
GHz = 1.0e6 * kHz
