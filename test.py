from source import BPModel
from tools import *
import numpy as np

def StepIndexRI(X, Y, P):
    nClad = silica_index(P.lambda0)
    nCore = np.sqrt(nClad**2 + 0.22**2)
    RI = nClad * np.ones_like(X)
    RI[np.sqrt(X**2 + Y**2) < 52.5e-6] = nCore
    return RI

P = BPModel()
P.xmax = 150e-6
P.ymax = 150e-6
P.N = 256
P.set_RI_from_function(StepIndexRI)
print(np.max(P.RI))
P.solve_modes(Nmodes=25, plotModes=False)

P.set_initial_field_from_mode('all', combination=True)
P.show_field()
