from source import BPModel
from tools import *
import numpy as np
import matplotlib.pyplot as plt

def StepIndexRI(X, Y, P):
    nClad = P.n0
    nCore = np.sqrt(nClad**2 + 0.22**2)
    RI = nClad * np.ones_like(X)
    RI[np.sqrt(X**2 + Y**2) < 52.5e-6] = nCore
    return RI

def RI2(X, Y, P):
    nClad = P.n0
    RI = nClad * np.ones_like(X)
    nCore = np.sqrt(nClad**2 + 0.22**2)
    RI[np.sqrt(X**2 + Y**2) < 32.5e-6] = nCore
    return RI

P = BPModel()
P.xmax = 200e-6
P.ymax = 200e-6
P.N = 300
P.Lz = 10e-2
P.dz = 1e-3
P.n0 = silica_index(P.lambda0)
P.set_RI_from_function(StepIndexRI)

P.solve_modes(Nmodes=10, plotModes=False)

P.set_initial_field_from_mode('all', combination=True)
P.shift_field(0,0)
P.show_field()
P.set_RI_from_function(RI2)

P.propagate_full(animate = True)
print(P.initPower)
print(P.remainingPower)